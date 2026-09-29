"""Replay a historical interval three independent ways.

1. **ledger replay**      - rebuild every batch purely from the spans recorded in
                            the consumption ledger + the immutable shards.
                            Needs no pipeline state and no model.
2. **pipeline replay**    - restore the data-pipeline state from the checkpoint
                            preceding the interval and re-run lane streams, OPUS
                            and packing. Must reproduce batch ids, spans, hashes,
                            the OPUS decision stream and the pipeline state.
3. **training replay**    - additionally restore model + optimizer and retrain the
                            interval; per-step loss and per-token loss files must
                            reproduce the learning ledger.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from . import checkpoint as ckpt
from . import config as C
from .build import Env
from .firewall import assert_batch_trainable
from .ledger import scan
from .packing import batch_fingerprint, materialize
from .trainer import (Run, build_model, layout_sha256, sequences_from_record, setup_torch,
                      train_step)
from .util import sha256_bytes, sha256_json


def ledger_records(path: str, key: str = "step") -> dict:
    return {r["data"][key]: r["data"] for r in scan(path)["records"]}


def replay_from_ledger(env: Env, cons: dict, steps) -> list:
    rows = []
    for s in steps:
        rec = cons[s]
        seqs = sequences_from_record(rec)
        assert_batch_trainable(env.catalog, seqs)
        arrays = materialize(seqs, env.catalog, C.TRAIN["seq_len"])
        h = batch_fingerprint(seqs, arrays)
        rows.append({"step": s, "original_batch_id": rec["batch_id"],
                     "replay_batch_id": f"B{s:05d}-{h[:12]}",
                     "original_hash": rec["batch_hash"], "replay_hash": h,
                     "match": h == rec["batch_hash"] and f"B{s:05d}-{h[:12]}" == rec["batch_id"]})
    return rows


def replay_from_checkpoint(env: Env, ckpt_path: str, cons: dict, opus_by_step: dict,
                           steps) -> list:
    meta, _blob, state = ckpt.load(ckpt_path)
    pipe = env.pipeline()
    pipe.set_state(state)
    rows = []
    for s in steps:
        b = pipe.next_batch(s)
        rec = cons[s]
        orig_layout = layout_sha256(sequences_from_record(rec))
        orig_dec = opus_by_step.get(s, [])
        strip = lambda d: {k: v for k, v in d.items() if k not in ("batch_id", "branch")}
        rows.append({
            "step": s, "original_batch_id": rec["batch_id"], "replay_batch_id": b["batch_id"],
            "original_hash": rec["batch_hash"], "replay_hash": b["batch_hash"],
            "original_spans_sha256": orig_layout, "replay_spans_sha256": layout_sha256(b["sequences"]),
            "opus_decisions_match": sha256_json([strip(d) for d in orig_dec]) ==
            sha256_json(b["decisions"]),
            "pipeline_state_match": pipe.state_sha256() == rec["pipeline_state_sha256"],
            "match": (b["batch_id"] == rec["batch_id"] and b["batch_hash"] == rec["batch_hash"]
                      and layout_sha256(b["sequences"]) == orig_layout)})
    return rows


def replay_training(env: Env, art: str, ckpt_path: str, learn: dict, steps) -> list:
    setup_torch()
    meta, blob, state = ckpt.load(ckpt_path)
    model, opt = build_model(env.tok.vocab_size)
    model.load_state_dict(blob["model"]); opt.load_state_dict(blob["optim"])
    torch.set_rng_state(blob["torch_rng"])
    pipe = env.pipeline()
    pipe.set_state(state)
    rows = []
    for s in steps:
        b = pipe.next_batch(s)
        loss, _gn, tl = train_step(model, opt, b["arrays"], s)
        orig = learn[s]
        orig_tl = np.load(os.path.join(art, orig["token_loss_file"]))
        rows.append({"step": s, "original_loss": orig["loss"], "replay_loss": loss,
                     "abs_diff": abs(loss - orig["loss"]),
                     "token_losses_bitwise_equal": bool(np.array_equal(orig_tl, tl)),
                     "token_loss_sha256_match": sha256_bytes(tl.tobytes()) ==
                     sha256_bytes(orig_tl.tobytes())})
    return rows


def run_replay(art: str, from_step: int, start: int, end: int) -> dict:
    env = Env(art)
    R = Run(art, "main")
    cons = ledger_records(R.ledger_path("consumption"))
    learn = ledger_records(R.ledger_path("learning"))
    opus_by_step: dict = {}
    for r in scan(R.ledger_path("opus_decisions"))["records"]:
        opus_by_step.setdefault(r["data"]["step"], []).append(r["data"])
    steps = range(start, end + 1)
    cpath = ckpt.ckpt_dir(R.cdir, from_step)
    a = replay_from_ledger(env, cons, steps)
    b = replay_from_checkpoint(env, cpath, cons, opus_by_step, steps)
    c = replay_training(env, art, cpath, learn, steps)
    return {"interval": [start, end], "from_checkpoint": f"checkpoints/step_{from_step:06d}",
            "original_attempts": sorted({cons[s]["attempt"] for s in steps}),
            "ledger_replay": {"all_match": all(r["match"] for r in a), "rows": a},
            "pipeline_replay": {"all_match": all(r["match"] and r["opus_decisions_match"]
                                                 and r["pipeline_state_match"] for r in b),
                                "rows": b},
            "training_replay": {"all_match": all(r["token_losses_bitwise_equal"] for r in c),
                                "max_abs_loss_diff": max(r["abs_diff"] for r in c), "rows": c}}
