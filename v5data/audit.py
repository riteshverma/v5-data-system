"""Independent audit: re-derive every claim from artifacts on disk.

The auditor trusts nothing the training processes *said*; it re-reads shards,
manifests, ledgers, checkpoints and reports, recomputes hashes, re-materializes
batches, re-derives OPUS decisions, reconciles the mixture, and re-computes the
performance numbers. Each requirement gets a list of named checks; the
requirement passes only if every check passes. Output: evidence.json + .md.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

from . import checkpoint as ckpt
from . import config as C
from .build import Env, paths, raw_doc_index
from .firewall import HELD_OUT_SPLITS
from .ledger import parse_orphans, scan
from .model import TinyGPT
from .opus import ACCEPT, DEFER, OVERRIDE, REJECT, OpusGate
from .packing import attention_mask, batch_fingerprint, check_invariants, materialize
from .perf import packing_metrics, throughput_metrics
from .shards import validate_all
from .tokenizer import FrozenTokenizer
from .trainer import LEDGERS, Run, layout_sha256, sequences_from_record
from .mixture import verify_plan
from .util import (atomic_write_json, read_json, read_jsonl, rel, sha256_bytes, sha256_file,
                   sha256_json)


class Req:
    def __init__(self, rid: str, title: str, evidence_label: str):
        self.rid, self.title, self.label = rid, title, evidence_label
        self.checks, self.evidence, self.metrics = [], [], {}

    def check(self, name: str, ok, detail="") -> bool:
        ok = bool(ok)
        self.checks.append({"name": name, "passed": ok, "detail": detail})
        return ok

    def cite(self, path: str, what: str) -> None:
        self.evidence.append({"path": path, "what": what})

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c["passed"] for c in self.checks)

    def to_dict(self) -> dict:
        return {"id": self.rid, "title": self.title, "result": "PASS" if self.passed else "FAIL",
                "evidence_label": self.label, "checks": self.checks,
                "evidence": self.evidence, "metrics": self.metrics}


class Auditor:
    def __init__(self, art: str):
        self.art = art
        self.p = paths(art)
        self.env = Env(art)
        self.R = Run(art, "main")
        self.events = read_jsonl(os.path.join(art, "ledgers", "events.jsonl"))
        self.scans = {n: scan(self.R.ledger_path(n)) for n in LEDGERS}
        self.scans["lifecycle"] = scan(self.R.lifecycle_path)
        self.cons = [r["data"] for r in self.scans["consumption"]["records"]]
        self.learn = [r["data"] for r in self.scans["learning"]["records"]]
        self.opus = [r["data"] for r in self.scans["opus_decisions"]["records"]]
        self.cons_by = {c["step"]: c for c in self.cons}
        self.learn_by = {c["step"]: c for c in self.learn}
        self.lifecycle = [r["data"] for r in self.scans["lifecycle"]["records"]]
        self.reports = self.p["reports"]
        self._arrays: dict = {}

    # ------------------------------------------------------------ helpers
    def ev(self, name: str, level: str = "PASS") -> list:
        return [e for e in self.events if e["event"] == name and e["level"] == level]

    def arrays(self, step: int) -> tuple:
        if step not in self._arrays:
            seqs = sequences_from_record(self.cons_by[step])
            self._arrays[step] = (seqs, materialize(seqs, self.env.catalog, C.TRAIN["seq_len"]))
        return self._arrays[step]

    def report(self, name: str):
        p = os.path.join(self.reports, name)
        return read_json(p) if os.path.exists(p) else None

    def ckpt_metas(self, root=None) -> dict:
        root = root or self.R.cdir
        out = {}
        for d in sorted(glob.glob(os.path.join(root, "step_*"))):
            if os.path.isdir(d) and not d.endswith(".tmp"):
                out[int(os.path.basename(d)[5:])] = (d, read_json(os.path.join(d, "meta.json")))
        return out

    # ======================================================== requirements
    def end_to_end(self) -> Req:
        r = Req("end_to_end", "End-to-end execution", "run.log phase events + contiguous ledger")
        phases = ["shards_created", "manifests_validated", "eval_shard_blocked", "mixture_compiled",
                  "batches_packed", "opus_decisions_recorded", "checkpoint_saved",
                  "crash_simulated", "run_resumed", "historical_stream_replayed",
                  "branch_forked", "performance_measured"]
        seen = {e["event"] for e in self.events}
        missing = [p for p in phases if p not in seen]
        r.check("all_pipeline_phases_logged", not missing, f"missing={missing}")
        fails = [e["event"] for e in self.events if e["level"] == "FAIL"]
        r.check("no_FAIL_events_in_run_log", not fails, f"fails={fails}")
        steps = [c["step"] for c in self.cons]
        r.check("main_run_completed_all_steps", steps == list(range(1, C.TRAIN["total_steps"] + 1)),
                f"{len(steps)} committed steps")
        r.cite("run.log", "complete ordered event log from all processes")
        r.cite("ledgers/events.jsonl", "machine-readable version of run.log")
        return r

    def tokenizer_integrity(self) -> Req:
        r = Req("tokenizer_integrity", "Tokenizer integrity", "Manifest record")
        lock = read_json(self.p["tokenizer_lock"])
        try:
            tok = FrozenTokenizer.load_frozen(self.p["tokenizer"], self.p["tokenizer_lock"])
            ok = tok.sha256 == lock["tokenizer_sha256"]
        except Exception as e:  # noqa: BLE001
            ok, tok = False, None
            r.check("load_error", False, str(e))
        r.check("tokenizer_file_matches_lock_hash", ok, lock["tokenizer_sha256"])
        r.check("lock_marks_frozen", lock.get("frozen") is True)
        r.check("trained_on_train_split_only", lock["trained_on_splits"] == ["train"],
                str(lock["trained_on_splits"]))
        bad = [sid for sid, m in self.env.catalog.manifests.items()
               if m["tokenizer_sha256"] != lock["tokenizer_sha256"]]
        r.check("every_shard_manifest_pins_frozen_tokenizer", not bad,
                f"{len(self.env.catalog.manifests)} manifests, mismatched={bad}")
        metas = self.ckpt_metas()
        r.check("every_checkpoint_pins_frozen_tokenizer",
                all(m["tokenizer_sha256"] == lock["tokenizer_sha256"] for _, m in metas.values()),
                f"{len(metas)} checkpoints")
        raw = raw_doc_index(self.art)
        mism = 0
        for m in self.env.catalog.manifests.values():
            for row in m["docs"]:
                ids, ls = tok.encode_document(raw[row["doc_id"]])
                if sha256_bytes(np.asarray(ids, np.uint16).tobytes()) != row["token_sha256"] \
                        or ls != row["loss_start"]:
                    mism += 1
        r.metrics["documents_retokenized"] = len(raw)
        r.check("retokenizing_every_raw_document_reproduces_shard_tokens", mism == 0,
                f"{len(raw)} docs, {mism} mismatches")
        r.check("tamper_detection_demonstrated", self.ev("tokenizer_tamper_detected"))
        r.check("roundtrip_demonstrated", self.ev("tokenizer_roundtrip_verified"))
        r.cite("tokenizer/tokenizer.lock.json", "frozen tokenizer hash + training provenance")
        r.cite("manifests/shards/", "tokenizer_sha256 field of every shard manifest")
        r.cite("manifests/validation_report.json", "re-tokenization check per shard")
        return r

    def shards_manifests(self) -> Req:
        r = Req("shards_manifests", "Immutable shards and manifests", "Shard manifests + validation")
        rep = validate_all(self.art, self.env.tok, self.env.tok_sha, raw_doc_index(self.art))
        sh = rep["shards"].values()
        r.check("file_hashes_match_manifests", all(s["file_hashes"] for s in sh))
        r.check("manifest_self_hashes_valid", all(s["manifest_self_hash"] for s in sh))
        r.check("catalog_pins_each_manifest_hash", all(s["catalog_pin"] for s in sh))
        r.check("catalog_self_hash_valid", rep["catalog_self_hash"], rep["catalog_sha256"])
        r.check("shard_files_read_only", all(s["read_only"] for s in sh))
        r.check("overwrite_refused_demonstrated", self.ev("shard_overwrite_refused"))
        r.check("tamper_detection_demonstrated", self.ev("shard_tamper_detected"))
        r.metrics = {"shards": len(rep["shards"]),
                     "documents": sum(s["num_docs"] for s in sh),
                     "tokens": sum(m["num_tokens"] for m in self.env.catalog.manifests.values())}
        r.cite("manifests/catalog.json", "catalog hash over all shard manifest hashes")
        r.cite("manifests/shards/", "per-shard manifest (file hashes, per-doc table)")
        r.cite("shards/", "immutable (read-only) token/index files")
        return r

    def _all_consumption(self) -> list:
        """Every consumption record ever written: live main, orphans, branches."""
        recs = list(self.cons)
        for f in glob.glob(os.path.join(self.p["ledgers"], "**", "consumption.jsonl"),
                           recursive=True):
            if os.path.normpath(f) == os.path.normpath(self.R.ledger_path("consumption")):
                continue
            recs += [x["data"] for x in parse_orphans(f)]
        return recs

    def eval_firewall(self) -> Req:
        r = Req("eval_firewall", "Evaluation firewall", "Blocked-shard event")
        r.check("eval_shard_registration_blocked", self.ev("eval_shard_blocked"))
        r.check("val_shard_registration_blocked", self.ev("val_shard_blocked"))
        r.check("held_out_materialization_as_training_blocked",
                self.ev("held_out_batch_blocked"))
        cat = self.env.catalog
        allrecs = self._all_consumption()
        bad = [(c["step"], s[0]) for c in allrecs for sq in c["sequences"] for s in sq["spans"]
               if cat.split_of(s[0]) != "train"]
        r.check("no_held_out_token_in_any_loss_bearing_batch", not bad,
                f"{len(allrecs)} consumption records incl. orphans and branches; violations={bad[:3]}")
        flagged = self.env.contamination["flagged"]
        consumed_keys = {f"{s[0]}/{s[1]}" for c in allrecs for sq in c["sequences"]
                         for s in sq["spans"]}
        leaked = sorted(set(flagged) & consumed_keys)
        r.check("no_contaminated_document_consumed", not leaked, f"leaked={leaked}")
        fl_dec = [d for d in self.opus if d["doc_key"] in flagged]
        r.check("every_contaminated_candidate_hard_rejected",
                fl_dec and all(d["decision"] == REJECT and d["reason"].startswith("eval_contamination")
                               for d in fl_dec), f"{len(fl_dec)} contaminated candidates seen")
        held_keys = {f"{sid}/" for sid in cat.manifests if cat.split_of(sid) in HELD_OUT_SPLITS}
        r.check("held_out_docs_never_offered_to_opus",
                not [d for d in self.opus if any(d["doc_key"].startswith(h) for h in held_keys)])
        vals = [x["data"] for x in self.scans["validation"]["records"]]
        evs = [x["data"] for x in scan(os.path.join(self.R.ldir, "evaluation.jsonl"))["records"]]
        ho = vals + evs
        r.check("held_out_scoring_is_gradient_free_and_non_loss_bearing",
                ho and all(v["grad"] is False and v["loss_bearing"] is False for v in ho),
                f"{len(vals)} validation + {len(evs)} evaluation records")
        r.check("held_out_scoring_reads_only_held_out_shards",
                all(cat.split_of(s) in HELD_OUT_SPLITS for v in ho for s in v["shards"]))
        r.metrics = {"contaminated_train_docs_flagged": len(flagged),
                     "contaminated_candidates_rejected": len(fl_dec),
                     "by_reason": _count([v["reason"] for v in flagged.values()])}
        r.cite("ledgers/events.jsonl", "eval_shard_blocked / val_shard_blocked events")
        r.cite("manifests/contamination_report.json", "train docs flagged against the eval set")
        r.cite("ledgers/opus_decisions.jsonl", "REJECT eval_contamination records")
        r.cite("ledgers/validation.jsonl", "held-out scoring records (grad=false)")
        return r

    def packing(self) -> Req:
        r = Req("packing", "Packing correctness", "Packed-batch report")
        errs, per_policy = [], {}
        for c in self.cons:
            seqs, arr = self.arrays(c["step"])
            e = check_invariants(seqs, arr, self.env.catalog)
            errs += [f"step {c['step']}: {x}" for x in e]
            if batch_fingerprint(seqs, arr) != c["batch_hash"]:
                errs.append(f"step {c['step']}: batch hash differs on re-materialization")
            for sq in seqs:
                d = per_policy.setdefault(sq["policy"], {"sequences": 0, "pad": 0, "split_docs": 0,
                                                         "multi_doc_sequences": 0})
                d["sequences"] += 1; d["pad"] += sq["pad"]
                d["split_docs"] += sum(1 for g in sq["segments"] if g["start"] > 0)
                d["multi_doc_sequences"] += len(sq["segments"]) > 1
        r.check("all_batches_rematerialize_to_recorded_hash_and_pass_mask_invariants", not errs,
                f"{len(self.cons)} batches; first errors={errs[:3]}")
        cs = per_policy.get("concat_split_docmask", {})
        r.check("concat_policy_zero_padding_and_splits_docs",
                cs and cs["pad"] == 0 and cs["split_docs"] > 0, str(cs))
        nosplit = {k: v for k, v in per_policy.items() if k.startswith("bestfit")}
        r.check("bestfit_policies_never_split_documents",
                nosplit and all(v["split_docs"] == 0 for v in nosplit.values()), str(nosplit))
        r.check("documents_are_actually_packed_together",
                all(v["multi_doc_sequences"] > 0 for v in per_policy.values()))
        iso = self._attention_isolation()
        r.check("trained_model_attention_isolated_between_packed_documents",
                iso["max_abs_logit_change_other_docs"] < 1e-5 and iso["max_abs_logit_change_same_doc"] > 1e-4,
                str(iso))
        seqs, arr = self.arrays(1)
        sample = []
        for row in range(len(seqs)):
            if len(seqs[row]["segments"]) >= 2 and len(sample) < 3:
                n = C.TRAIN["seq_len"] - seqs[row]["pad"]
                sample.append({"row": row, "lane": seqs[row]["lane"], "policy": seqs[row]["policy"],
                               "spans": seqs[row]["segments"],
                               "tokens": arr["tokens"][row, :n].tolist(),
                               "text": self.env.tok.decode(arr["tokens"][row, :n]),
                               "position_ids": arr["position_ids"][row, :n].tolist(),
                               "segment_ids": arr["segment_ids"][row, :n].tolist(),
                               "loss_mask": arr["loss_mask"][row, :n].tolist(),
                               "pad": seqs[row]["pad"]})
        rep = {"batches_checked": len(self.cons), "violations": errs, "per_policy": per_policy,
               "attention_isolation_test": iso, "sample_step1_rows": sample}
        atomic_write_json(os.path.join(self.reports, "packing_report.json"), rep)
        r.metrics = {"per_policy": per_policy, "attention_isolation": iso}
        r.cite("reports/packing_report.json", "invariant results, isolation test, decoded sample rows")
        r.cite("ledgers/consumption.jsonl", "recorded spans that every batch was rebuilt from")
        return r

    def _attention_isolation(self) -> dict:
        """On the final trained model: perturb one packed document and verify that
        logits of the other documents in the same row do not move."""
        d, meta = self.ckpt_metas()[max(self.ckpt_metas())]
        _, blob, _ = ckpt.load(d)
        m = TinyGPT(self.env.tok.vocab_size, C.TRAIN["seq_len"], **C.TRAIN["model"])
        m.load_state_dict(blob["model"]); m.eval()
        seqs, arr = self.arrays(1)
        row = next(i for i, s in enumerate(seqs) if len(s["segments"]) >= 3)
        seg = arr["segment_ids"][row:row + 1]
        tok = torch.from_numpy(arr["tokens"][row:row + 1]).clone()
        pos = torch.from_numpy(arr["position_ids"][row:row + 1])
        mask = torch.from_numpy(attention_mask(seg))
        with torch.no_grad():
            base = m(tok, pos, mask)
            target = seg[0] == 2
            first = int(np.argmax(target))
            tok2 = tok.clone(); tok2[0, first] = (tok2[0, first] + 7) % self.env.tok.vocab_size
            pert = m(tok2, pos, mask)
        diff = (pert - base).abs().amax(-1)[0].numpy()
        other = (seg[0] != 2) & (seg[0] > 0)
        return {"step": 1, "row": row, "perturbed_segment": 2, "perturbed_position": first,
                "max_abs_logit_change_other_docs": float(diff[other].max()),
                "max_abs_logit_change_same_doc": float(diff[target].max())}

    def mixture(self) -> Req:
        r = Req("mixture", "Mixture compliance", "Planned versus actual shares")
        plan = self.env.plan
        r.check("compiled_plan_hash_valid", not verify_plan(plan), plan["plan_sha256"][:16])
        metas = self.ckpt_metas()
        r.check("every_checkpoint_pins_the_plan",
                all(m["plan_sha256"] == plan["plan_sha256"] for _, m in metas.values()))
        r.check("every_batch_records_the_plan",
                all(c["plan_sha256"] == plan["plan_sha256"] for c in self.cons))
        lanes, floors = plan["lane_order"], plan["floors_seqs"]
        recon_err, floor_err, stages = [], [], {}
        for c in self.cons:
            ps = plan["steps"][c["step"] - 1]
            actual = _count([sq["lane"] for sq in c["sequences"]])
            short = sum(c["lanes"][l]["shortfall"] for l in lanes)
            for l in lanes:
                lr = c["lanes"][l]
                want = ps["seqs"][l] - lr["shortfall"] + lr["backfill_received"]
                if actual.get(l, 0) != want or lr["planned"] != ps["seqs"][l]:
                    recon_err.append((c["step"], l))
                if actual.get(l, 0) < floors[l]:
                    floor_err.append((c["step"], l))
            if sum(c["lanes"][l]["backfill_received"] for l in lanes) != short:
                recon_err.append((c["step"], "backfill"))
            st = stages.setdefault(c["stage"], {"planned": {l: 0 for l in lanes},
                                                "actual": {l: 0 for l in lanes},
                                                "loss_tokens": {l: 0 for l in lanes},
                                                "floor_overrides": {l: 0 for l in lanes},
                                                "shortfall": {l: 0 for l in lanes}})
            for l in lanes:
                st["planned"][l] += ps["seqs"][l]
                st["actual"][l] += actual.get(l, 0)
                st["floor_overrides"][l] += c["lanes"][l]["overrides"]
                st["shortfall"][l] += c["lanes"][l]["shortfall"]
            for s in self.learn_by[c["step"]]["segments"]:
                st["loss_tokens"][s[4]] += s[7]
        table, maxdev = [], 0.0
        for name, st in stages.items():
            tp, ta, tt = (sum(st[k].values()) for k in ("planned", "actual", "loss_tokens"))
            target = next(s["weights"] for s in plan["stages"] if s["name"] == name)
            for l in lanes:
                row = {"stage": name, "lane": l, "target_weight": target[l],
                       "planned_seq_share": st["planned"][l] / tp,
                       "actual_seq_share": st["actual"][l] / ta,
                       "actual_loss_token_share": st["loss_tokens"][l] / tt,
                       "floor_seqs_per_step": floors[l],
                       "protected_floor_overrides": st["floor_overrides"][l],
                       "opus_shortfall_seqs": st["shortfall"][l]}
                maxdev = max(maxdev, abs(row["actual_seq_share"] - row["planned_seq_share"]))
                table.append(row)
        r.check("every_step_reconciles_plan_minus_shortfall_plus_backfill", not recon_err,
                f"errors={recon_err[:5]}")
        r.check("protected_floors_held_on_every_step", not floor_err,
                f"floors={floors}; violations={floor_err[:5]}")
        r.check("planned_vs_actual_share_within_tolerance", maxdev <= 0.10,
                f"max |actual-planned| stage share = {maxdev:.4f} (tol 0.10)")
        r.check("curriculum_stages_executed_in_order",
                [c["stage"] for c in self.cons] == [s["stage"] for s in plan["steps"]])
        rep = {"plan_sha256": plan["plan_sha256"], "floors_seqs_per_step": floors,
               "max_stage_share_deviation": maxdev, "table": table}
        atomic_write_json(os.path.join(self.reports, "mixture_report.json"), rep)
        r.metrics = {"max_stage_share_deviation": maxdev, "table": table}
        r.cite("manifests/mixture_plan.json", "compiled per-step plan (stages, weights, floors)")
        r.cite("reports/mixture_report.json", "planned vs actual shares per stage and lane")
        return r

    def opus_trail(self) -> Req:
        r = Req("opus", "OPUS audit trail", "Candidate decision records")
        plan, cat = self.env.plan, self.env.catalog
        gate = OpusGate(C.OPUS, self.env.contamination["flagged"])
        defers, admitted, rejected, mism = {}, {}, set(), []
        for i, d in enumerate(self.opus):
            sid, idx = d["doc_key"].rsplit("/", 1)
            meta = cat.doc_meta(sid, int(idx))
            if meta["quality"] != d["quality"] or meta["content_sha256"] != d["content_sha256"]:
                mism.append((i, "record inputs differ from manifest"))
            if d["thresholds"] != plan["steps"][d["step"] - 1]["opus"]:
                mism.append((i, "thresholds differ from plan"))
            dec, reason, score = gate.evaluate(d["doc_key"], meta, d["defer_count"], d["thresholds"])
            if d["decision"] == OVERRIDE:
                if dec != DEFER or not C.LANES[d["lane"]]["protected"]:
                    mism.append((i, "override of a non-deferrable candidate"))
            else:
                if (dec, reason) != (d["decision"], d["reason"]) or score != d["score"]:
                    mism.append((i, f"re-derived {dec}/{reason} != {d['decision']}/{d['reason']}"))
                if d["defer_count"] != defers.get(d["doc_key"], 0):
                    mism.append((i, "defer count discontinuity"))
            if d["decision"] in (ACCEPT, OVERRIDE):
                gate.admit(meta)
                admitted.setdefault(d["doc_key"], d["step"])
            elif d["decision"] == DEFER:
                defers[d["doc_key"]] = defers.get(d["doc_key"], 0) + 1
            else:
                rejected.add(d["doc_key"])
        r.check("every_decision_rederived_from_manifest_and_plan", not mism,
                f"{len(self.opus)} decisions; mismatches={mism[:3]}")
        kinds = _count([d["decision"] for d in self.opus])
        r.check("accept_reject_defer_override_all_exercised",
                all(kinds.get(k, 0) > 0 for k in (ACCEPT, REJECT, DEFER, OVERRIDE)), str(kinds))
        bad_link, spans = [], {}
        for c in self.cons:
            for sq in c["sequences"]:
                for s in sq["spans"]:
                    k = f"{s[0]}/{s[1]}"
                    if k not in admitted or admitted[k] > c["step"]:
                        bad_link.append((c["step"], k))
                    spans.setdefault(k, []).append((s[3], s[4], sq["policy"]))
        r.check("every_consumed_document_has_prior_admission_record", not bad_link,
                f"violations={bad_link[:3]}")
        r.check("no_rejected_document_consumed", not (rejected & set(spans)))
        overlap = []
        for k, lst in spans.items():
            lst.sort()
            if lst[0][0] != 0 or any(a[1] != b[0] for a, b in zip(lst, lst[1:])):
                overlap.append(k)
            if lst[0][2] != "concat_split_docmask" and len(lst) > 1:
                overlap.append(k)
        r.check("no_document_span_consumed_twice_or_skipped", not overlap, f"violations={overlap[:3]}")
        by_step = {}
        for d in self.opus:
            by_step.setdefault(d["step"], []).append(
                {k: v for k, v in d.items() if k not in ("batch_id", "branch")})
        r.check("decision_stream_hash_matches_consumption_record_per_step",
                all(sha256_json(by_step.get(c["step"], [])) == c["opus"]["sha256"] and
                    len(by_step.get(c["step"], [])) == c["opus"]["decisions"] for c in self.cons))
        r.check("overrides_only_in_protected_lanes",
                all(C.LANES[d["lane"]]["protected"] for d in self.opus if d["decision"] == OVERRIDE))
        by = {}
        for d in self.opus:
            key = f"{d['stage']}|{d['lane']}|{d['decision']}|{d['reason'].split(':')[0]}"
            by[key] = by.get(key, 0) + 1
        examples = {}
        for d in self.opus:
            examples.setdefault(f"{d['decision']}:{d['reason'].split(':')[0]}", d)
        rep = {"decisions": len(self.opus), "by_decision": kinds, "by_stage_lane_decision_reason": by,
               "examples": examples, "admitted_docs": len(admitted), "rejected_docs": len(rejected)}
        atomic_write_json(os.path.join(self.reports, "opus_report.json"), rep)
        r.metrics = {"by_decision": kinds, "admitted_docs": len(admitted)}
        r.cite("ledgers/opus_decisions.jsonl", "one hash-chained record per candidate decision")
        r.cite("reports/opus_report.json", "counts by stage/lane/decision/reason + examples")
        return r

    def ledgers(self) -> Req:
        r = Req("ledgers", "Consumption and learning ledgers", "Hash-chained ledgers")
        files = [self.R.ledger_path(n) for n in LEDGERS] + [self.R.lifecycle_path,
                                                            os.path.join(self.R.ldir, "evaluation.jsonl")]
        files += glob.glob(os.path.join(self.p["ledgers"], "branches", "*", "*.jsonl"))
        bad = [rel(f, self.art) for f in files
               if not (s := scan(f))["ok"] or s["torn_tail_bytes"]]
        r.check("all_live_ledgers_hash_chains_valid_no_torn_tail", not bad,
                f"{len(files)} ledgers; bad={bad}")
        T = C.TRAIN["total_steps"]
        r.check("consumption_has_each_step_exactly_once",
                [c["step"] for c in self.cons] == list(range(1, T + 1)))
        r.check("learning_has_each_step_exactly_once",
                [c["step"] for c in self.learn] == list(range(1, T + 1)))
        r.check("learning_records_bind_to_consumption_batches",
                all(self.learn_by[s]["batch_hash"] == self.cons_by[s]["batch_hash"] and
                    self.learn_by[s]["batch_id"] == self.cons_by[s]["batch_id"]
                    for s in range(1, T + 1)))
        r.check("batch_ids_unique", len({c["batch_id"] for c in self.cons}) == T)
        metas, off_bad = self.ckpt_metas(), []
        for step, (d, m) in metas.items():
            try:
                ckpt.load(d)
            except Exception as e:  # noqa: BLE001
                off_bad.append((step, str(e)))
            for n in LEDGERS:
                off = m["ledger_offsets"][n]
                recs = self.scans[n]["records"]
                if off["records"] > len(recs) or (off["records"] and recs[off["records"] - 1]["hash"] != off["head"]):
                    off_bad.append((step, n, "head"))
                with open(self.R.ledger_path(n), "rb") as f:
                    f.seek(off["bytes"] - 1 if off["bytes"] else 0)
                    if off["bytes"] and f.read(1) != b"\n":
                        off_bad.append((step, n, "byte offset not on record boundary"))
            if self.cons[m["ledger_offsets"]["consumption"]["records"] - 1]["step"] != step:
                off_bad.append((step, "consumption offset step"))
        r.check("checkpoints_bound_to_exact_ledger_offsets", metas and not off_bad,
                f"{sorted(metas)}; errors={off_bad[:3]}")
        life = [x["event"] for x in self.lifecycle]
        r.check("lifecycle_ledger_records_checkpoints_crash_and_recovery",
                life.count("checkpoint_saved") >= len(metas) and "crash_recovered" in life
                and "process_exited" in life)
        r.metrics = {"ledger_records": {n: self.scans[n]["count"] for n in self.scans},
                     "checkpoints": sorted(metas)}
        for n in LEDGERS:
            r.cite(f"ledgers/{n}.jsonl", f"{self.scans[n]['count']} records, head {self.scans[n]['head'][:16]}")
        r.cite("ledgers/lifecycle.jsonl", "checkpoint / crash / recovery transaction log")
        r.cite("checkpoints/", "meta.json with ledger offsets per checkpoint")
        return r

    def learning_trace(self) -> Req:
        r = Req("learning_trace", "Learning trace", "Loss linked to source data")
        errs = []
        for l in self.learn:
            c = self.cons_by[l["step"]]
            seqs, arr = self.arrays(l["step"])
            want = [(r_, k, f"{g['shard']}/{g['doc_idx']}", g["start"], g["end"])
                    for r_, sq in enumerate(seqs) for k, g in enumerate(sq["segments"], 1)]
            got = [(s[0], s[1], s[2], s[5], s[6]) for s in l["segments"]]
            if want != got:
                errs.append((l["step"], "segments != consumption spans"))
            f = os.path.join(self.art, l["token_loss_file"])
            if sha256_file(f) != l["token_loss_sha256"]:
                errs.append((l["step"], "token-loss file hash"))
                continue
            tl = np.load(f)
            if np.any(tl[arr["loss_mask"] == 0] != 0):
                errs.append((l["step"], "loss outside loss mask"))
            ntok = sum(s[7] for s in l["segments"])
            if ntok != int(arr["loss_mask"].sum()) or ntok != l["loss_tokens"]:
                errs.append((l["step"], "loss token count"))
            seg_sum = sum(s[8] for s in l["segments"])
            if abs(seg_sum / ntok - l["loss"]) > 1e-4 * max(1, l["loss"]):
                errs.append((l["step"], "segment losses do not sum to step loss"))
            p_row = {}
            for s in l["segments"]:
                p0 = p_row.get(s[0], 0); n = s[6] - s[5]
                if abs(float(tl[s[0], p0:p0 + n].astype(np.float64).sum()) - s[8]) > 1e-6:
                    errs.append((l["step"], "segment loss != token-loss file")); break
                p_row[s[0]] = p0 + n
            del c
        r.check("every_step_loss_decomposes_into_per_document_losses", not errs, f"errors={errs[:3]}")
        losses = [l["loss"] for l in self.learn]
        first, last = float(np.mean(losses[:10])), float(np.mean(losses[-10:]))
        r.check("model_learned_loss_decreased", last < first, f"first10={first:.3f} last10={last:.3f}")
        vals = [x["data"]["loss"] for x in self.scans["validation"]["records"]]
        r.check("validation_loss_decreased", len(vals) >= 2 and vals[-1] < vals[0], str([round(v, 3) for v in vals]))
        docs, lane_stage = {}, {}
        for l in self.learn:
            for s in l["segments"]:
                d = docs.setdefault(s[2], {"doc_id": s[3], "lane": s[4], "spans": []})
                d["spans"].append({"step": l["step"], "batch_id": l["batch_id"], "row": s[0],
                                   "start": s[5], "end": s[6], "loss_tokens": s[7],
                                   "mean_loss": s[8] / s[7] if s[7] else None})
                ls = lane_stage.setdefault(f"{l['stage']}|{s[4]}", [0.0, 0])
                ls[0] += s[8]; ls[1] += s[7]
        split_docs = {k: v for k, v in docs.items() if len(v["spans"]) > 1}
        rep = {"documents_trained": len(docs),
               "per_stage_lane_mean_loss": {k: v[0] / v[1] for k, v in lane_stage.items() if v[1]},
               "loss_curve": [[l["step"], l["loss"]] for l in self.learn],
               "validation_curve": [[x["data"]["step"], x["data"]["loss"]]
                                    for x in self.scans["validation"]["records"]],
               "example_split_document_trace": dict(list(split_docs.items())[:3]),
               "doc_trace": docs}
        atomic_write_json(os.path.join(self.reports, "learning_trace.json"), rep, indent=None)
        r.metrics = {"documents_trained": len(docs), "first10_loss": first, "last10_loss": last,
                     "validation_curve": rep["validation_curve"]}
        r.cite("ledgers/learning.jsonl", "per-step loss + per-document (row, span) loss sums")
        r.cite("ledgers/token_losses/", "token-level loss arrays, hash-pinned by the learning ledger")
        r.cite("reports/learning_trace.json", "document -> (step, span, loss) index")
        return r

    def crash_recovery(self) -> Req:
        r = Req("crash_recovery", "Crash recovery", "Expected and resumed batch ids")
        life = self.lifecycle
        exits = [x for x in life if x["event"] == "process_exited"]
        crash = [x for x in exits if x["exit_code"] != 0]
        r.check("training_process_actually_died", crash and crash[0]["exit_code"] == 137,
                str([(x["attempt"], x["exit_code"]) for x in exits]))
        rec = next((x for x in life if x["event"] == "crash_recovered"), None)
        r.check("recovery_event_recorded", rec is not None)
        if rec is None:
            return r
        ck = rec["checkpoint_step"]
        d, meta = self.ckpt_metas()[ck]
        exp = meta["next_batch_expected"]
        got = self.cons_by[exp["step"]]
        r.check("resumed_first_batch_equals_checkpoint_expected_batch",
                exp["batch_id"] == got["batch_id"] and exp["batch_hash"] == got["batch_hash"]
                and exp["layout_sha256"] == layout_sha256(sequences_from_record(got)),
                f"expected {exp['batch_id']} / resumed {got['batch_id']}")
        orphan_dir = os.path.join(self.R.ldir, "orphans", f"attempt{rec['attempt'] - 1}")
        orphans = {x["data"]["step"]: x["data"] for x in
                   parse_orphans(os.path.join(orphan_dir, "consumption.jsonl"))}
        crash_step = crash[0].get("crash_at")
        r.check("uncommitted_tail_preserved_as_orphans",
                sorted(orphans) == list(range(ck + 1, crash_step + 1)),
                f"orphan steps {sorted(orphans)}")
        r.check("resumed_batches_equal_crashed_attempt_batches",
                all(orphans[s]["batch_hash"] == self.cons_by[s]["batch_hash"] for s in orphans))
        ol = {x["data"]["step"]: x["data"] for x in
              parse_orphans(os.path.join(orphan_dir, "learning.jsonl"))}
        r.check("resumed_model_reproduces_crashed_attempt_losses_bitwise",
                ol and all(ol[s]["loss"] == self.learn_by[s]["loss"] for s in ol),
                f"{len(ol)} completed pre-crash steps compared")
        ref = self.report("reference_stream.json")
        live = [(c["step"], c["batch_id"], c["batch_hash"]) for c in self.cons]
        refl = [(x["step"], x["batch_id"], x["batch_hash"]) for x in ref["batches"]]
        r.check("whole_stream_equals_uninterrupted_reference_no_skip_no_repeat", live == refl,
                f"{len(live)} batches vs {len(refl)} in reference")
        attempts = _count([c["attempt"] for c in self.cons])
        r.check("steps_split_between_attempts_at_checkpoint",
                all(c["attempt"] == (1 if c["step"] <= ck else rec["attempt"]) for c in self.cons),
                str(attempts))
        r.check("resume_event_logged_pass", self.ev("resume_next_batch_matched"))
        r.metrics = {"checkpoint_step": ck, "crash_step": crash_step,
                     "expected_next_batch": exp, "resumed_batch": {"batch_id": got["batch_id"],
                                                                   "batch_hash": got["batch_hash"]},
                     "rolled_back": rec["recoveries"], "attempts": attempts}
        atomic_write_json(os.path.join(self.reports, "resume_report.json"), r.metrics)
        r.cite(f"checkpoints/step_{ck:06d}/meta.json", "next_batch_expected (lookahead)")
        r.cite("ledgers/consumption.jsonl", f"step {exp['step']} record written by the resumed attempt")
        r.cite(rel(orphan_dir, self.art), "uncommitted records of the crashed attempt")
        r.cite("reports/reference_stream.json", "uninterrupted data-only reference stream")
        r.cite("reports/resume_report.json", "summary")
        return r

    def replay(self) -> Req:
        r = Req("replay", "Replay", "Original and replay hashes")
        rep = self.report("replay_report.json")
        r.check("replay_report_present", rep is not None)
        if rep is None:
            return r
        s0, s1 = rep["interval"]
        r.check("ledger_replay_hashes_match", rep["ledger_replay"]["all_match"])
        r.check("checkpoint_pipeline_replay_ids_spans_hashes_opus_state_match",
                rep["pipeline_replay"]["all_match"])
        r.check("training_replay_token_losses_bitwise_equal", rep["training_replay"]["all_match"],
                f"max |loss diff| = {rep['training_replay']['max_abs_loss_diff']}")
        # independent re-check of the report against the live ledger
        rows = rep["pipeline_replay"]["rows"]
        r.check("replay_report_consistent_with_live_ledger",
                [x["step"] for x in rows] == list(range(s0, s1 + 1)) and
                all(self.cons_by[x["step"]]["batch_hash"] == x["original_hash"] == x["replay_hash"]
                    for x in rows))
        r.check("replay_event_logged_pass", self.ev("replay_hash_matched"))
        r.metrics = {"interval": rep["interval"], "from_checkpoint": rep["from_checkpoint"],
                     "rows": [{"step": x["step"], "original": x["original_batch_id"],
                               "replay": x["replay_batch_id"]} for x in rows]}
        r.cite("reports/replay_report.json", "per-step original vs replayed ids / hashes / losses")
        return r

    def fork(self) -> Req:
        r = Req("fork", "Fork from earlier checkpoint", "Branch lineage + ledgers")
        rep = self.report("fork_report.json")
        r.check("fork_report_present", rep is not None)
        if rep is None:
            return r
        b = rep["branch"]
        lin = read_json(os.path.join(self.art, "manifests", "branches", b, "lineage.json"))
        fr = Run(self.art, b)
        fcons = {x["data"]["step"]: x["data"] for x in scan(fr.ledger_path("consumption"))["records"]}
        ok_prefix = True
        for n, pre in lin["ledger_prefixes"].items():
            with open(self.R.ledger_path(n), "rb") as f:
                a = f.read(pre["bytes"])
            with open(fr.ledger_path(n), "rb") as f:
                bb = f.read(pre["bytes"])
            ok_prefix &= sha256_bytes(a) == sha256_bytes(bb) == pre["prefix_sha256"]
        r.check("branch_shares_verified_history_prefix_with_parent", ok_prefix)
        F, D, E = lin["fork_step"], lin["diverge_at_step"], rep["end_step"]
        same = [s for s in range(F + 1, D) if fcons[s]["batch_hash"] == self.cons_by[s]["batch_hash"]]
        r.check("branch_reproduces_parent_batches_before_divergence", len(same) == D - F - 1,
                f"steps {F + 1}..{D - 1}")
        diff = [s for s in range(D, E + 1) if fcons[s]["batch_hash"] != self.cons_by[s]["batch_hash"]]
        r.check("branch_diverges_under_new_mixture", len(diff) == E - D + 1, f"steps {D}..{E}")
        r.check("branch_steps_contiguous", sorted(fcons) == list(range(1, E + 1)))
        r.check("parent_ledgers_untouched_by_fork",
                all(sha256_file(self.R.ledger_path(n)) == h for n, h in rep["parent_ledger_sha256_before"].items()))
        fplan = read_json(os.path.join(self.art, "manifests", "branches", b, "mixture_plan.json"))
        r.check("branch_consumption_follows_branch_plan",
                all(_count([sq["lane"] for sq in fcons[s]["sequences"]]).get(l, 0) ==
                    fplan["steps"][s - 1]["seqs"][l] - fcons[s]["lanes"][l]["shortfall"]
                    + fcons[s]["lanes"][l]["backfill_received"]
                    for s in range(F + 1, E + 1) for l in fplan["lane_order"]))
        r.check("branch_checkpoint_saved", os.path.exists(os.path.join(fr.cdir, f"step_{E:06d}", "meta.json")))
        r.metrics = {"branch": b, "fork_step": F, "diverge_at": D, "end_step": E,
                     "identical_steps": same, "diverged_steps": diff}
        r.cite(f"manifests/branches/{b}/lineage.json", "parent checkpoint + ledger prefix hashes")
        r.cite(f"ledgers/branches/{b}/", "branch ledgers (parent prefix + branch history)")
        r.cite("reports/fork_report.json", "per-step parent vs branch batch ids")
        return r

    def throughput(self) -> Req:
        r = Req("throughput", "Throughput", "Performance report")
        perf = read_json(os.path.join(self.art, "performance.json"))
        pk = packing_metrics(self.env, self.cons)
        tp = throughput_metrics(self.learn)
        r.check("packing_numbers_reconstructed_from_ledgers",
                _close(pk, perf["packing"]), "recomputed from re-materialized batches")
        r.check("throughput_numbers_reconstructed_from_ledger_timings",
                _close(tp, perf["throughput"]), "recomputed from learning-ledger timing_ms")
        r.check("packing_beats_naive_one_doc_per_sequence",
                pk["totals"]["utilization"] > pk["naive_one_doc_per_sequence"]["utilization"],
                f"{pk['totals']['utilization']:.3f} vs {pk['naive_one_doc_per_sequence']['utilization']:.3f}")
        r.check("useful_loss_bearing_tokens_per_sec_measured",
                tp["useful_loss_bearing_tokens_per_sec"] > 0,
                f"{tp['useful_loss_bearing_tokens_per_sec']:.0f} tok/s")
        r.metrics = {"utilization": pk["totals"]["utilization"],
                     "loss_bearing_fraction": pk["totals"]["loss_bearing_fraction"],
                     "naive_utilization": pk["naive_one_doc_per_sequence"]["utilization"],
                     "useful_loss_bearing_tokens_per_sec": tp["useful_loss_bearing_tokens_per_sec"],
                     "slot_tokens_per_sec": tp["slot_tokens_per_sec"]}
        r.cite("performance.json", "utilization, loss-bearing fraction, tokens/s, naive baseline")
        return r

    # ================================================================ run
    def run(self) -> list:
        fns = [(self.end_to_end, "end_to_end", "End-to-end execution"),
               (self.tokenizer_integrity, "tokenizer_integrity", "Tokenizer integrity"),
               (self.shards_manifests, "shards_manifests", "Immutable shards and manifests"),
               (self.eval_firewall, "eval_firewall", "Evaluation firewall"),
               (self.packing, "packing", "Packing correctness"),
               (self.mixture, "mixture", "Mixture compliance"),
               (self.opus_trail, "opus", "OPUS audit trail"),
               (self.ledgers, "ledgers", "Consumption and learning ledgers"),
               (self.learning_trace, "learning_trace", "Learning trace"),
               (self.crash_recovery, "crash_recovery", "Crash recovery"),
               (self.replay, "replay", "Replay"),
               (self.fork, "fork", "Fork from earlier checkpoint"),
               (self.throughput, "throughput", "Throughput")]
        out = []
        for fn, rid, title in fns:
            try:
                out.append(fn())
            except Exception as e:  # noqa: BLE001 - an audit that crashes is a failed audit
                r = Req(rid, title, "audit error")
                r.check("audit_completed_without_error", False, f"{type(e).__name__}: {e}")
                out.append(r)
        return out


def _count(xs) -> dict:
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def _close(a, b, tol=1e-9) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= tol * max(1.0, abs(a))
    return a == b


# ------------------------------------------------------------- evidence
TABLE_ORDER = [("tokenizer_integrity", "Tokenizer integrity"), ("eval_firewall", "Evaluation firewall"),
               ("packing", "Packing correctness"), ("mixture", "Mixture compliance"),
               ("opus", "OPUS audit trail"), ("crash_recovery", "Crash recovery"),
               ("replay", "Replay"), ("learning_trace", "Learning trace"),
               ("throughput", "Throughput")]


def write_evidence(art: str, reqs: list, meta: dict) -> dict:
    d = {r.rid: r.to_dict() for r in reqs}
    overall = all(r.passed for r in reqs)
    ev = {"schema": "v5.evidence/1", "overall": "PASS" if overall else "FAIL",
          "generated_by": "v5data.audit.Auditor (re-derived from artifacts on disk)",
          **meta, "requirements": d}
    atomic_write_json(os.path.join(art, "evidence.json"), ev)

    def cell(rid):
        x = d[rid]
        return x["evidence_label"] + " — " + ", ".join(f"`{e['path']}`" for e in x["evidence"][:2])

    lines = ["# V5 Training Data Execution System — Evidence", "",
             f"Overall: **{ev['overall']}** — {sum(r.passed for r in reqs)}/{len(reqs)} requirements, "
             f"{sum(len(r.checks) for r in reqs)} checks. Generated by the auditor from the artifacts "
             "in this directory; see `evidence.json` for every check and its detail.", "",
             "| Requirement | Result | Evidence |", "|---|---|---|"]
    for rid, title in TABLE_ORDER:
        lines.append(f"| {title} | {d[rid]['result']} | {cell(rid)} |")
    lines += ["", "Additional requirements:", "", "| Requirement | Result | Evidence |", "|---|---|---|"]
    for r in reqs:
        if r.rid not in dict(TABLE_ORDER):
            lines.append(f"| {r.title} | {d[r.rid]['result']} | {cell(r.rid)} |")
    m = {r.rid: r.metrics for r in reqs}
    cr, rp, tp = m["crash_recovery"], m["replay"], m["throughput"]
    lines += ["", "## Key facts", ""]
    if cr:
        lines.append(f"- **Crash/resume:** checkpoint at step {cr['checkpoint_step']}, process killed at step "
                     f"{cr['crash_step']} (exit 137). Checkpoint expected `{cr['expected_next_batch']['batch_id']}`; "
                     f"resumed run consumed `{cr['resumed_batch']['batch_id']}`.")
    if rp:
        a, b = rp["rows"][0], rp["rows"][-1]
        lines.append(f"- **Replay:** steps {rp['interval'][0]}–{rp['interval'][1]} from `{rp['from_checkpoint']}`: "
                     f"first `{a['original']}` → `{a['replay']}`, last `{b['original']}` → `{b['replay']}`.")
    lines.append(f"- **Packing:** utilization {tp['utilization']:.1%} (naive one-doc-per-sequence "
                 f"{tp['naive_utilization']:.1%}); loss-bearing fraction {tp['loss_bearing_fraction']:.1%}; "
                 f"{tp['useful_loss_bearing_tokens_per_sec']:.0f} useful loss-bearing tokens/s.")
    op = m["opus"]
    lines.append(f"- **OPUS:** {op['by_decision']}.")
    fk = m["fork"]
    if fk:
        lines.append(f"- **Fork:** `{fk['branch']}` from step {fk['fork_step']}: identical to parent for steps "
                     f"{fk['identical_steps'][0]}–{fk['identical_steps'][-1]}, diverged for "
                     f"{fk['diverged_steps'][0]}–{fk['diverged_steps'][-1]} under the new mixture.")
    fails = [(r.title, c["name"], c["detail"]) for r in reqs for c in r.checks if not c["passed"]]
    if fails:
        lines += ["", "## Failed checks", ""] + [f"- {t}: `{n}` {dt}" for t, n, dt in fails]
    with open(os.path.join(art, "evidence.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return ev


def main(argv=None) -> int:
    """``python -m v5data.audit --art DIR``: print {requirement: passed} as JSON."""
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    a = ap.parse_args(argv)
    res = {r.rid: r.passed for r in Auditor(a.art).run()}
    print(json.dumps(res))
    return 0 if all(res.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
