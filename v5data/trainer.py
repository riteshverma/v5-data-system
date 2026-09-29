"""Training process: pipeline -> ledgers -> model, with checkpoint / crash / resume / fork.

Run as a separate OS process by the demo (``python -m v5data.trainer ...``) so
the crash is a real process death (``os._exit(137)``) in the middle of a step,
not a Python exception that could be caught and cleaned up.

Write ordering per step (write-ahead):
  1. consumption record  (what is about to be trained on)   -> fsync
  2. OPUS decision records (why it was admitted)             -> fsync
  3. forward / backward / optimizer step
  4. token-loss file + learning record (what was learned)    -> fsync
A crash between 1 and 4 leaves a consumption record without a learning record;
recovery rolls every ledger back to the last checkpoint's offsets.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

from . import checkpoint as ckpt
from . import config as C
from .build import Env, compile_mixture, paths
from .firewall import assert_batch_trainable
from .ledger import Ledger, parse_orphans, recover_to, scan
from .model import TinyGPT, token_losses
from .packing import batch_fingerprint, materialize
from .pipeline import clone_pipeline, held_out_sequences
from .util import RunLog, atomic_write_json, read_json, rel, sha256_bytes, sha256_file, sha256_json

LEDGERS = ("consumption", "opus_decisions", "learning", "validation")


# ------------------------------------------------------------------ layout
class Run:
    def __init__(self, art: str, branch: str = "main"):
        self.art, self.branch = art, branch
        p = paths(art)
        if branch == "main":
            self.ldir, self.cdir = p["ledgers"], p["checkpoints"]
        else:
            self.ldir = os.path.join(p["ledgers"], "branches", branch)
            self.cdir = os.path.join(p["checkpoints"], "branches", branch)
        self.tdir = os.path.join(self.ldir, "token_losses")
        self.lifecycle_path = os.path.join(self.ldir, "lifecycle.jsonl")

    def ledger_path(self, name: str) -> str:
        return os.path.join(self.ldir, f"{name}.jsonl")

    def token_file(self, step: int) -> str:
        return os.path.join(self.tdir, f"step_{step:06d}.npy")


def sequences_from_record(rec_data: dict) -> list:
    """Rebuild pipeline-style sequences from a consumption ledger record."""
    return [{"lane": sq["lane"], "policy": sq["policy"], "pad": sq["pad"],
             "segments": [{"shard": s, "doc_idx": i, "doc_id": d, "start": a, "end": b}
                          for s, i, d, a, b in sq["spans"]]}
            for sq in rec_data["sequences"]]


def layout_sha256(sequences: list) -> str:
    return sha256_json([[sq["lane"], [[g["shard"], g["doc_idx"], g["start"], g["end"]]
                                      for g in sq["segments"]]] for sq in sequences])


# ------------------------------------------------------------------- model
def setup_torch() -> None:
    torch.set_num_threads(C.TRAIN["torch_threads"])
    torch.use_deterministic_algorithms(True)


def build_model(vocab: int):
    torch.manual_seed(C.SEED)
    m = TinyGPT(vocab, C.TRAIN["seq_len"], **C.TRAIN["model"])
    opt = torch.optim.AdamW(m.parameters(), lr=C.TRAIN["lr"], betas=(0.9, 0.95),
                            weight_decay=C.TRAIN["weight_decay"])
    return m, opt


def lr_at(step: int) -> float:
    T, W, lr = C.TRAIN["total_steps"], C.TRAIN["warmup_steps"], C.TRAIN["lr"]
    if step <= W:
        return lr * step / W
    frac = (step - W) / max(1, T - W)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))


def train_step(model, opt, arrays: dict, step: int) -> tuple:
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    tl = token_losses(model, arrays)
    n = int(arrays["loss_mask"].sum())
    loss = tl.sum() / n
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), C.TRAIN["grad_clip"])
    opt.step()
    return float(loss.item()), float(gn.item()), tl.detach().numpy().astype(np.float32)


def segment_losses(sequences: list, tl: np.ndarray, loss_mask: np.ndarray) -> list:
    """Sample-level learning trace: loss attributed to every packed document span."""
    out = []
    for r, sq in enumerate(sequences):
        p = 0
        for k, g in enumerate(sq["segments"], 1):
            n = g["end"] - g["start"]
            out.append([r, k, f"{g['shard']}/{g['doc_idx']}", g["doc_id"], sq["lane"],
                        g["start"], g["end"], int(loss_mask[r, p:p + n].sum()),
                        float(tl[r, p:p + n].astype(np.float64).sum())])
            p += n
    return out


@torch.no_grad()
def evaluate_held_out(model, env: Env, split: str) -> dict:
    """Loss on a held-out split. No gradient, no optimizer, never loss-bearing."""
    seqs = held_out_sequences(env.catalog, split, C.LANES, C.LANE_ORDER,
                              C.TRAIN["seq_len"], C.PACKING)
    was_training = model.training
    model.eval()
    per_lane, tot, ntok, hashes = {}, 0.0, 0, []
    for i in range(0, len(seqs), C.TRAIN["batch_size"]):
        chunk = seqs[i:i + C.TRAIN["batch_size"]]
        arrays = materialize(chunk, env.catalog, C.TRAIN["seq_len"])
        hashes.append(batch_fingerprint(chunk, arrays))
        tl = token_losses(model, arrays).numpy()
        for r, sq in enumerate(chunk):
            d = per_lane.setdefault(sq["lane"], [0.0, 0])
            d[0] += float(tl[r].sum()); d[1] += int(arrays["loss_mask"][r].sum())
        tot += float(tl.sum()); ntok += int(arrays["loss_mask"].sum())
    model.train(was_training)
    shards = sorted({g["shard"] for sq in seqs for g in sq["segments"]})
    return {"split": split, "loss": tot / max(1, ntok), "loss_tokens": ntok,
            "sequences": len(seqs), "shards": shards,
            "per_lane": {l: v[0] / max(1, v[1]) for l, v in per_lane.items()},
            "held_out_sha256": sha256_json(hashes), "grad": False, "loss_bearing": False}


# -------------------------------------------------------------------- loop
def run(art: str, mode: str, end_step: int, crash_at: int | None = None,
        branch: str = "main", fork_from: int | None = None, diverge_at: int | None = None) -> int:
    setup_torch()
    comp = {"train": "trainer", "resume": "resume", "fork": "fork"}[mode]
    log = RunLog(art, comp)
    R = Run(art, branch)
    plan_path = None
    if mode == "fork":
        fdir = os.path.join(art, "manifests", "branches", branch)
        stages = C.fork_stages(C.STAGES, diverge_at, C.FORK["weights"])
        plan_path = os.path.join(fdir, "mixture_plan.json")
        compile_mixture(art, stages=stages, out_path=plan_path)
    env = Env(art, plan_path)
    log.check(True, "tokenizer_hash_verified", f"process loaded frozen tokenizer "
              f"{env.tok_sha[:16]}; {len(env.catalog.manifests)} shard manifests verified")
    pipe = env.pipeline()
    model, opt = build_model(env.tok.vocab_size)
    for d in (R.ldir, R.cdir, R.tdir):
        os.makedirs(d, exist_ok=True)
    lifecycle = Ledger(R.lifecycle_path)
    prior_attempts = sum(1 for r in scan(R.lifecycle_path)["records"]
                         if r["data"]["event"] == "attempt_started")
    attempt = prior_attempts + 1
    expected_next, orphan_records = None, {}
    start = 1

    if mode == "resume":
        cpath = ckpt.latest(R.cdir)
        meta, blob, state = ckpt.load(cpath)
        _verify_ckpt_binding(meta, env, pipe)
        orphan_dir = os.path.join(R.ldir, "orphans", f"attempt{attempt - 1}")
        recs = [recover_to(R.ledger_path(n), meta["ledger_offsets"][n],
                           os.path.join(orphan_dir, f"{n}.jsonl")) for n in LEDGERS]
        moved = _orphan_token_files(R, meta["step"], orphan_dir)
        for r in recs:
            if r["orphan_file"]:
                r["orphan_file"] = rel(r["orphan_file"], art)
        for r in recs:
            log.info("ledger_rolled_back", f"{r['ledger']}: truncated {r['truncated_bytes']} B "
                     f"({r['orphaned_complete_records']} uncommitted records, "
                     f"{r['torn_bytes']} torn bytes) -> {r['orphan_file'] or '-'}")
        log.check(all(r["head_verified"] == meta["ledger_offsets"][r["ledger"][:-6]]["head"]
                      for r in recs), "ledger_rollback_to_checkpoint_offset",
                  f"all {len(recs)} ledgers end exactly at checkpoint step {meta['step']} heads")
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["optim"])
        torch.set_rng_state(blob["torch_rng"])
        pipe.set_state(state)
        ok = pipe.state_sha256() == meta["pipeline_state_sha256"]
        log.check(ok, "pipeline_state_restored", f"state sha {meta['pipeline_state_sha256'][:16]}")
        expected_next = meta["next_batch_expected"]
        for r in parse_orphans(os.path.join(orphan_dir, "consumption.jsonl")):
            orphan_records[r["data"]["step"]] = r["data"]
        lifecycle.append({"event": "crash_recovered", "attempt": attempt,
                          "from_checkpoint": rel(cpath, art), "checkpoint_step": meta["step"],
                          "recoveries": recs, "orphaned_token_files": moved})
        log.info("run_resumed", f"attempt {attempt} resumes from {rel(cpath, art)}; "
                 f"next step {meta['step'] + 1}")
        start = meta["step"] + 1

    elif mode == "fork":
        parent = Run(art, "main")
        cpath = ckpt.ckpt_dir(parent.cdir, fork_from)
        meta, blob, state = ckpt.load(cpath)
        lineage = {"schema": "v5.lineage/1", "branch": branch, "parent_branch": "main",
                   "parent_checkpoint": rel(cpath, art), "parent_checkpoint_meta_sha256":
                   meta["meta_sha256"], "fork_step": fork_from, "diverge_at_step": diverge_at,
                   "fork_plan_sha256": env.plan["plan_sha256"],
                   "parent_plan_sha256": meta["plan_sha256"], "ledger_prefixes": {}}
        for n in LEDGERS:
            off = meta["ledger_offsets"][n]
            src, dst = parent.ledger_path(n), R.ledger_path(n)
            if os.path.exists(dst):
                raise RuntimeError(f"branch ledger {dst} already exists")
            with open(src, "rb") as f:
                prefix = f.read(off["bytes"])
            with open(dst, "wb") as f:
                f.write(prefix)
            s = scan(dst)
            assert s["ok"] and s["head"] == off["head"] and s["count"] == off["records"]
            lineage["ledger_prefixes"][n] = {**off, "prefix_sha256": sha256_bytes(prefix)}
        atomic_write_json(os.path.join(art, "manifests", "branches", branch, "lineage.json"),
                          lineage)
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["optim"])
        torch.set_rng_state(blob["torch_rng"])
        pipe.set_state(state)
        lifecycle.append({"event": "forked", **lineage})
        log.check(True, "branch_forked", f"{branch} forked from {rel(cpath, art)}; shared "
                  f"ledger prefix verified; new mixture from step {diverge_at}")
        start = fork_from + 1

    ledgers = {n: Ledger(R.ledger_path(n)) for n in LEDGERS}
    lifecycle.append({"event": "attempt_started", "attempt": attempt, "mode": mode,
                      "start_step": start, "end_step": end_step, "pid": os.getpid(),
                      "crash_at": crash_at})
    lifecycle.commit()
    log.info("attempt_started", f"branch={branch} attempt={attempt} steps {start}..{end_step}"
             + (f" (fault injection armed at step {crash_at})" if crash_at else ""))

    for step in range(start, end_step + 1):
        t0 = time.perf_counter()
        b = pipe.next_batch(step)
        assert_batch_trainable(env.catalog, b["sequences"])     # firewall, last line
        t1 = time.perf_counter()

        if expected_next is not None and step == expected_next["step"]:
            ok = (b["batch_id"] == expected_next["batch_id"]
                  and b["batch_hash"] == expected_next["batch_hash"]
                  and layout_sha256(b["sequences"]) == expected_next["layout_sha256"])
            log.check(ok, "resume_next_batch_matched",
                      f"step {step}: expected {expected_next['batch_id']} "
                      f"got {b['batch_id']}")
            o = orphan_records.get(step)
            if o is not None:
                log.check(o["batch_hash"] == b["batch_hash"],
                          "resume_matches_crashed_attempt",
                          f"crashed attempt had already consumed {o['batch_id']} at step {step}")

        stream = {"branch": branch, "attempt": attempt, "step": step, "stage": b["stage"],
                  "batch_id": b["batch_id"], "batch_hash": b["batch_hash"],
                  "plan_sha256": env.plan["plan_sha256"]}
        ledgers["consumption"].append({
            **stream,
            "sequences": [{"lane": sq["lane"], "policy": sq["policy"], "pad": sq["pad"],
                           "spans": [[g["shard"], g["doc_idx"], g["doc_id"], g["start"], g["end"]]
                                     for g in sq["segments"]]} for sq in b["sequences"]],
            "lanes": b["lanes"], "stats": b["stats"],
            "opus": {"decisions": len(b["decisions"]), "sha256": sha256_json(b["decisions"])},
            "pipeline_state_sha256": pipe.state_sha256()})
        for d in b["decisions"]:
            ledgers["opus_decisions"].append({**d, "batch_id": b["batch_id"], "branch": branch})
        ledgers["consumption"].commit(); ledgers["opus_decisions"].commit()
        t2 = time.perf_counter()

        loss, gn, tl = train_step(model, opt, b["arrays"], step)
        t3 = time.perf_counter()

        tf = R.token_file(step)
        np.save(tf, tl)
        learn = {**stream, "loss": loss, "grad_norm": gn, "lr": lr_at(step),
                 "loss_tokens": b["stats"]["loss_tokens"],
                 "token_loss_file": rel(tf, art), "token_loss_sha256": sha256_file(tf),
                 "segments": segment_losses(b["sequences"], tl, b["arrays"]["loss_mask"]),
                 "timing_ms": {"data": (t1 - t0) * 1e3, "ledger": (t2 - t1) * 1e3,
                               "compute": (t3 - t2) * 1e3}}
        per_lane = {}
        for s in learn["segments"]:
            d = per_lane.setdefault(s[4], [0.0, 0]); d[0] += s[8]; d[1] += s[7]
        learn["per_lane"] = {l: {"loss": v[0] / max(1, v[1]), "tokens": v[1]}
                             for l, v in per_lane.items()}

        if crash_at is not None and step == crash_at:
            ledgers["learning"].write_torn(learn)
            log.warn("crash_simulated", f"process {os.getpid()} killed mid-step {step} "
                     f"(consumption+OPUS written, learning record torn); exit 137")
            os._exit(137)

        ledgers["learning"].append(learn)
        ledgers["learning"].commit()
        t4 = time.perf_counter()
        if step % 10 == 0 or step == start:
            log.info("batch_trained", f"step {step:3d} {b['batch_id']} stage={b['stage']:<11} "
                     f"loss={loss:.4f} loss_tokens={b['stats']['loss_tokens']} "
                     f"opus_decisions={len(b['decisions'])} ({(t4 - t0) * 1e3:.0f} ms)")

        if step % C.TRAIN["val_every"] == 0:
            v = evaluate_held_out(model, env, "val")
            ledgers["validation"].append({**stream, **v})
            ledgers["validation"].commit()
            log.info("validation", f"step {step}: val loss {v['loss']:.4f} on "
                     f"{v['loss_tokens']} held-out tokens (no grad)")

        if step % C.TRAIN["ckpt_every"] == 0 or step == end_step:
            _checkpoint(R, env, pipe, model, opt, ledgers, lifecycle, step, branch, attempt, log)

    if branch == "main" and end_step == C.TRAIN["total_steps"]:
        ev = Ledger(os.path.join(R.ldir, "evaluation.jsonl"))
        for split in ("val", "eval"):
            v = evaluate_held_out(model, env, split)
            ev.append({"branch": branch, "after_step": end_step, **v})
            log.info("held_out_scored", f"{split}: loss {v['loss']:.4f} over "
                     f"{v['loss_tokens']} tokens, shards {len(v['shards'])} (no grad)")
        ev.commit()
    lifecycle.append({"event": "attempt_finished", "attempt": attempt, "last_step": end_step})
    lifecycle.commit()
    return 0


def _verify_ckpt_binding(meta: dict, env: Env, pipe) -> None:
    for k, want in (("plan_sha256", env.plan["plan_sha256"]), ("tokenizer_sha256", env.tok_sha),
                    ("catalog_sha256", env.catalog.catalog["catalog_sha256"]),
                    ("order_sha256", pipe.order_sha256)):
        if meta[k] != want:
            raise ckpt.CheckpointError(f"checkpoint {k} {meta[k][:12]} != current {want[:12]}")


def _orphan_token_files(R: Run, ckpt_step: int, orphan_dir: str) -> list:
    moved = []
    if not os.path.isdir(R.tdir):
        return moved
    for fn in sorted(os.listdir(R.tdir)):
        if int(fn[5:11]) > ckpt_step:
            dst = os.path.join(orphan_dir, "token_losses", fn)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(os.path.join(R.tdir, fn), dst)
            moved.append(fn)
    return moved


def _checkpoint(R, env, pipe, model, opt, ledgers, lifecycle, step, branch, attempt, log):
    for l in ledgers.values():
        l.commit()
    expected = None
    if step < env.plan["total_steps"]:
        nb = clone_pipeline(pipe).next_batch(step + 1)       # lookahead on a copy
        expected = {"step": step + 1, "batch_id": nb["batch_id"],
                    "batch_hash": nb["batch_hash"], "layout_sha256": layout_sha256(nb["sequences"])}
    meta = ckpt.save(R.cdir, step, model, opt, pipe.get_state(), {
        "branch": branch, "attempt": attempt,
        "ledger_offsets": {n: l.offset() for n, l in ledgers.items()},
        "plan_sha256": env.plan["plan_sha256"], "tokenizer_sha256": env.tok_sha,
        "catalog_sha256": env.catalog.catalog["catalog_sha256"],
        "order_sha256": pipe.order_sha256, "next_batch_expected": expected})
    lifecycle.append({"event": "checkpoint_saved", "step": step, "attempt": attempt,
                      "dir": rel(ckpt.ckpt_dir(R.cdir, step), R.art),
                      "meta_sha256": meta["meta_sha256"],
                      "ledger_offsets": meta["ledger_offsets"],
                      "next_batch_expected": expected})
    lifecycle.commit()
    log.check(True, "checkpoint_saved",
              f"step {step} -> {rel(ckpt.ckpt_dir(R.cdir, step), R.art)}; consumption ledger "
              f"offset {meta['ledger_offsets']['consumption']['records']} records / "
              f"{meta['ledger_offsets']['consumption']['bytes']} B; next expected "
              f"{expected['batch_id'] if expected else '-'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--mode", choices=["train", "resume", "fork"], required=True)
    ap.add_argument("--end-step", type=int, required=True)
    ap.add_argument("--crash-at", type=int)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--fork-from", type=int)
    ap.add_argument("--diverge-at", type=int)
    a = ap.parse_args(argv)
    return run(a.art, a.mode, a.end_step, a.crash_at, a.branch, a.fork_from, a.diverge_at)


if __name__ == "__main__":
    sys.exit(main())
