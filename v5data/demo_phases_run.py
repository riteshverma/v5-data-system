"""Demonstration phases, part 2: crash bookkeeping, replay, fork, performance, audit."""
from __future__ import annotations

import os
import time

from . import build
from . import config as C
from .audit import Auditor, write_evidence
from .ledger import Ledger, scan
from .perf import packing_metrics, throughput_metrics
from .replay import run_replay
from .trainer import LEDGERS, Run
from .util import atomic_write_json, sha256_file


def record_exit(art, attempt, rc, crash_at=None):
    """The supervisor, not the (possibly dead) worker, records how the process ended."""
    lc = Ledger(Run(art).lifecycle_path)
    lc.append({"event": "process_exited", "attempt": attempt, "exit_code": rc,
               "crash_at": crash_at})
    lc.commit()
    lc.close()


def after_crash(art, log, rc, crash):
    record_exit(art, 1, rc, crash)
    log.check(rc == 137, "crash_confirmed", f"training process exited with code {rc} at step {crash}")
    path = Run(art).ledger_path("opus_decisions")
    n = 0
    if os.path.exists(path):
        with open(path, "rb") as f:
            n = f.read().count(b"\n")
    log.info("opus_decisions_recorded", f"{n} candidate decisions in ledgers/opus_decisions.jsonl "
             "(including the not-yet-committed tail of the crashed step)")


def after_resume(art, log, rc):
    record_exit(art, 2, rc)
    log.check(rc == 0, "resume_completed", f"resumed process exit code {rc}")


def replay(art, log):
    R = C.REPLAY
    rp = run_replay(art, R["from_checkpoint_step"], R["start"], R["end"])
    atomic_write_json(os.path.join(build.paths(art)["reports"], "replay_report.json"), rp)
    a, b = rp["pipeline_replay"]["rows"][0], rp["pipeline_replay"]["rows"][-1]
    log.info("historical_stream_replayed", f"steps {R['start']}..{R['end']} (written by attempt(s) "
             f"{rp['original_attempts']}) replayed from {rp['from_checkpoint']}")
    log.check(rp["ledger_replay"]["all_match"], "replay_ledger_rebuild_matched",
              "every batch rebuilt from ledger spans + shards reproduces its recorded hash")
    log.check(rp["pipeline_replay"]["all_match"], "replay_hash_matched",
              f"ids/spans/hashes/OPUS decisions/pipeline state match; {a['original_batch_id']}=="
              f"{a['replay_batch_id']} ... {b['original_batch_id']}=={b['replay_batch_id']}")
    log.check(rp["training_replay"]["all_match"], "replay_learning_matched",
              "retrained interval: token losses bitwise equal, max |loss diff| "
              f"{rp['training_replay']['max_abs_loss_diff']}")


def parent_ledger_hashes(art) -> dict:
    R = Run(art)
    return {n: sha256_file(R.ledger_path(n)) for n in LEDGERS}


def after_fork(art, log, rc, before):
    F, R = C.FORK, Run(art)
    fr = Run(art, F["name"])
    fcons = {x["data"]["step"]: x["data"] for x in scan(fr.ledger_path("consumption"))["records"]}
    mcons = {x["data"]["step"]: x["data"] for x in scan(R.ledger_path("consumption"))["records"]}
    rows = [{"step": s, "parent_batch_id": mcons[s]["batch_id"],
             "branch_batch_id": fcons[s]["batch_id"] if s in fcons else None,
             "identical": s in fcons and mcons[s]["batch_hash"] == fcons[s]["batch_hash"]}
            for s in range(F["from_checkpoint_step"] + 1, F["end_step"] + 1)]
    atomic_write_json(os.path.join(build.paths(art)["reports"], "fork_report.json"), {
        "branch": F["name"], "fork_step": F["from_checkpoint_step"],
        "diverge_at": F["diverge_at_step"], "end_step": F["end_step"], "exit_code": rc,
        "branch_weights": F["weights"], "parent_ledger_sha256_before": before,
        "parent_ledger_sha256_after": parent_ledger_hashes(art), "rows": rows})
    pre = [x for x in rows if x["step"] < F["diverge_at_step"]]
    post = [x for x in rows if x["step"] >= F["diverge_at_step"]]
    log.check(rc == 0 and all(x["identical"] for x in pre), "fork_reproduced_parent_history",
              f"steps {pre[0]['step']}..{pre[-1]['step']} identical to parent")
    log.check(rc == 0 and all(not x["identical"] for x in post), "fork_diverged_under_new_mixture",
              f"steps {post[0]['step']}..{post[-1]['step']} follow {F['weights']}")


def performance(art, log, ref_secs):
    env, R, T = build.Env(art), Run(art), C.TRAIN["total_steps"]
    cons = [x["data"] for x in scan(R.ledger_path("consumption"))["records"]]
    learn = [x["data"] for x in scan(R.ledger_path("learning"))["records"]]
    slots_per_step = C.TRAIN["batch_size"] * C.TRAIN["seq_len"]
    perf = {"schema": "v5.performance/1",
            "definitions": {
                "utilization": "non-pad tokens / (sequences x seq_len)",
                "loss_bearing_fraction": "tokens with loss_mask=1 / slots (excludes pad, prompt "
                                         "tokens and each document's last token)",
                "useful_loss_bearing_tokens_per_sec": "sum(loss tokens) / sum(step time); step "
                "time = data + consumption/OPUS ledger fsync + fwd/bwd/opt",
                "naive_one_doc_per_sequence": "the same consumed documents, each alone in ceil(len/seq_len) padded sequences"},
            "packing": packing_metrics(env, cons),
            "throughput": throughput_metrics(learn),
            "data_pipeline_only": {"batches": T, "seconds": ref_secs,
                                   "batches_per_sec": T / ref_secs,
                                   "slot_tokens_per_sec": T * slots_per_step / ref_secs,
                                   "source": "reports/reference_stream.json"},
            "hardware": {"device": "cpu", "torch_threads": C.TRAIN["torch_threads"]}}
    atomic_write_json(os.path.join(art, "performance.json"), perf)
    pk, tp = perf["packing"], perf["throughput"]
    log.info("performance_measured",
             f"utilization {pk['totals']['utilization']:.1%} (naive "
             f"{pk['naive_one_doc_per_sequence']['utilization']:.1%}), loss-bearing "
             f"{pk['totals']['loss_bearing_fraction']:.1%}, "
             f"{tp['useful_loss_bearing_tokens_per_sec']:.0f} useful loss-bearing tok/s, "
             f"data pipeline alone {perf['data_pipeline_only']['slot_tokens_per_sec']:.0f} tok/s")


def audit(art, log, t0) -> bool:
    reqs = Auditor(art).run()
    for r in reqs:
        log.check(r.passed, f"audit:{r.rid}",
                  f"{sum(c['passed'] for c in r.checks)}/{len(r.checks)} checks")
    ev = write_evidence(art, reqs, {"artifacts_dir": os.path.basename(art)})
    log.check(ev["overall"] == "PASS", "audit_completed",
              f"{sum(r.passed for r in reqs)}/{len(reqs)} requirements PASS -> evidence.json, "
              f"evidence.md ({time.perf_counter() - t0:.1f}s total)")
    return ev["overall"] == "PASS"
