"""Atomic checkpoints bound to ledger offsets.

``checkpoints/step_000040/`` holds:

* ``model_optim.pt``  model + optimizer + torch RNG state
* ``pipeline_state.json``  the data pipeline state after step 40
* ``meta.json``  step, file hashes, pipeline-state hash, plan / tokenizer /
  catalog hashes, the byte/record/head offset of every ledger at step 40, and
  the fingerprint of the batch the run *must* consume next (lookahead).

The directory is written under a temporary name and renamed into place, then
``LATEST.json`` is atomically updated - a crash can never leave a
half-written checkpoint that looks valid.
"""
from __future__ import annotations

import os
import shutil

import torch

from .util import atomic_write_json, read_json, sha256_file, sha256_json


class CheckpointError(RuntimeError):
    pass


def ckpt_dir(root: str, step: int) -> str:
    return os.path.join(root, f"step_{step:06d}")


def save(root: str, step: int, model, opt, pipeline_state: dict, meta_extra: dict) -> dict:
    final = ckpt_dir(root, step)
    tmp = final + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                "torch_rng": torch.get_rng_state()}, os.path.join(tmp, "model_optim.pt"))
    atomic_write_json(os.path.join(tmp, "pipeline_state.json"), pipeline_state, indent=None)
    meta = {"schema": "v5.checkpoint/1", "step": step,
            "files": {fn: sha256_file(os.path.join(tmp, fn))
                      for fn in ("model_optim.pt", "pipeline_state.json")},
            "pipeline_state_sha256": sha256_json(pipeline_state), **meta_extra}
    meta["meta_sha256"] = sha256_json(meta)
    atomic_write_json(os.path.join(tmp, "meta.json"), meta)
    if os.path.exists(final):
        raise CheckpointError(f"checkpoint {final} already exists (checkpoints are immutable)")
    os.replace(tmp, final)
    atomic_write_json(os.path.join(root, "LATEST.json"),
                      {"step": step, "dir": os.path.basename(final),
                       "meta_sha256": meta["meta_sha256"]})
    return meta


def latest(root: str) -> str:
    p = os.path.join(root, "LATEST.json")
    if not os.path.exists(p):
        raise CheckpointError(f"no LATEST.json in {root}")
    return os.path.join(root, read_json(p)["dir"])


def load(path: str) -> tuple:
    meta = read_json(os.path.join(path, "meta.json"))
    if sha256_json({k: v for k, v in meta.items() if k != "meta_sha256"}) != meta["meta_sha256"]:
        raise CheckpointError(f"{path}: meta self-hash mismatch")
    for fn, h in meta["files"].items():
        if sha256_file(os.path.join(path, fn)) != h:
            raise CheckpointError(f"{path}: {fn} hash mismatch")
    blob = torch.load(os.path.join(path, "model_optim.pt"), weights_only=False)
    state = read_json(os.path.join(path, "pipeline_state.json"))
    if sha256_json(state) != meta["pipeline_state_sha256"]:
        raise CheckpointError(f"{path}: pipeline state hash mismatch")
    return meta, blob, state
