"""Phases of the end-to-end demonstration (driven by run_demo.py), part 1: build.

Each phase performs real work through the library and logs PASS/FAIL checks
whose outcome is computed, never asserted.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time

from . import build
from . import config as C
from .demo_phases_run import (after_crash, after_fork, after_resume, audit,  # noqa: F401
                              parent_ledger_hashes, performance, replay)
from .firewall import FirewallViolation, assert_batch_trainable
from .mixture import verify_plan
from .pipeline import DataPipeline, held_out_sequences
from .shards import Catalog, ShardExistsError, ShardIntegrityError, ShardReader, write_shard
from .tokenizer import FrozenTokenizer, TokenizerIntegrityError
from .trainer import layout_sha256
from .util import atomic_write_json


def documents_and_tokenizer(art, log):
    P = build.paths(art)
    corpus, _ = build.build_corpus(art)
    log.info("documents_written", ", ".join(f"{l}.{s}={len(d)}" for (l, s), d in sorted(corpus.items()))
             + " -> corpus/raw/ (includes planted duplicates, eval leaks and spam)")
    tok, lock = build.build_tokenizer(art, corpus)
    log.info("tokenizer_trained", f"byte-level BPE, vocab {lock['vocab_size']}, trained on "
             f"{lock['trained_on_splits']} only; frozen sha256 {lock['tokenizer_sha256'][:16]}")
    tok2 = FrozenTokenizer.load_frozen(P["tokenizer"], P["tokenizer_lock"])
    log.check(tok2.sha256 == lock["tokenizer_sha256"], "tokenizer_hash_verified",
              f"reloaded tokenizer hash == lock {lock['tokenizer_sha256'][:16]}")
    texts = [build.tokenizer_training_text(d) for docs in corpus.values() for d in docs[:50]]
    log.check(all(tok2.decode(tok2.encode(t)) == t for t in texts), "tokenizer_roundtrip_verified",
              f"decode(encode(x)) == x for {len(texts)} documents")
    with tempfile.TemporaryDirectory() as td:
        with open(P["tokenizer"], encoding="utf-8") as f:
            d = json.load(f)
        d["merges"][0] = d["merges"][1]
        bad = os.path.join(td, "tokenizer.json")
        with open(bad, "w", encoding="utf-8") as f:
            json.dump(d, f)
        try:
            FrozenTokenizer.load_frozen(bad, P["tokenizer_lock"])
            log.check(False, "tokenizer_tamper_detected", "modified tokenizer was accepted")
        except TokenizerIntegrityError as e:
            log.check(True, "tokenizer_tamper_detected", f"one edited merge rejected: {e}")
    return corpus, tok, lock


def shards_and_manifests(art, log, corpus, tok, lock):
    P = build.paths(art)
    mans = build.build_shards(art, corpus, tok, lock["tokenizer_sha256"])
    by_split = {}
    for m in mans:
        v = by_split.setdefault(m["split"], [0, 0])
        v[0] += 1
        v[1] += m["num_tokens"]
    log.info("shards_created", f"{len(mans)} shards (read-only) + manifests; "
             + ", ".join(f"{k}: {v[0]} shards / {v[1]} tokens" for k, v in sorted(by_split.items())))
    rep = build.validate(art, tok, lock["tokenizer_sha256"])
    log.check(rep["ok"], "manifests_validated",
              f"{len(rep['shards'])} manifests: file hashes, self-hashes, read-only bits, tokenizer "
              f"pin, re-tokenization of every document; catalog {rep['catalog_sha256'][:16]}")
    first = mans[0]
    try:
        write_shard(P["shards"], P["shard_manifests"], first["shard_id"], first["lane"],
                    first["split"], [], tok, lock["tokenizer_sha256"])
        log.check(False, "shard_overwrite_refused", "immutable shard was overwritten")
    except ShardExistsError:
        log.check(True, "shard_overwrite_refused", f"rewrite of {first['shard_id']} refused")
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "shards"))
        for k in ("tokens", "index"):
            shutil.copy(os.path.join(art, first["files"][k]["path"]),
                        os.path.join(td, first["files"][k]["path"]))
        tp = os.path.join(td, first["files"]["tokens"]["path"])
        os.chmod(tp, 0o666)
        with open(tp, "r+b") as f:
            f.seek(10)
            b = f.read(1)
            f.seek(10)
            f.write(bytes([b[0] ^ 1]))
        try:
            ShardReader(td, first, verify=True)
            log.check(False, "shard_tamper_detected", "flipped bit not detected")
        except ShardIntegrityError as e:
            log.check(True, "shard_tamper_detected",
                      f"1 flipped bit in a copy of {first['shard_id']}: {e}")


def firewall_and_mixture(art, log, lock):
    cat = Catalog(art, lock["tokenizer_sha256"])
    cont = build.build_contamination(art, cat)
    reasons = {}
    for v in cont["flagged"].values():
        reasons[v["reason"]] = reasons.get(v["reason"], 0) + 1
    log.info("contamination_scanned", f"{cont['train_docs_scanned']} train docs vs "
             f"{cont['eval_docs']} eval docs: {len(cont['flagged'])} flagged {reasons}")
    plan = build.compile_mixture(art)
    log.check(not verify_plan(plan), "mixture_compiled",
              f"{plan['total_steps']} steps, plan {plan['plan_sha256'][:16]}, floors/step "
              f"{plan['floors_seqs']}; " +
              "; ".join(f"{k}: {v['planned_seq_share']}" for k, v in plan["summary"].items()))
    probe = DataPipeline(cat, plan, C.LANES, C.OPUS, C.PACKING, C.TRAIN["seq_len"], C.SEED, cont,
                         auto_register=False)
    for split, ev in (("eval", "eval_shard_blocked"), ("val", "val_shard_blocked")):
        sid = cat.shards_where(lane="web", split=split)[0]
        try:
            probe.register_source("web", sid)
            log.check(False, ev, f"{sid} accepted as a training source")
        except FirewallViolation as e:
            log.check(True, ev, str(e))
    try:
        assert_batch_trainable(cat, held_out_sequences(cat, "eval", C.LANES, C.LANE_ORDER,
                                                       C.TRAIN["seq_len"], C.PACKING))
        log.check(False, "held_out_batch_blocked", "eval batch accepted for training")
    except FirewallViolation as e:
        log.check(True, "held_out_batch_blocked",
                  f"packed eval batch refused at the optimizer gate: {e}")


def reference_stream(art, log) -> float:
    """Uninterrupted data-only run: the stream every training attempt must reproduce."""
    env = build.Env(art)
    T = C.TRAIN["total_steps"]
    pipe, ref, t0 = env.pipeline(), [], time.perf_counter()
    for s in range(1, T + 1):
        ts = time.perf_counter()
        b = pipe.next_batch(s)
        ref.append({"step": s, "batch_id": b["batch_id"], "batch_hash": b["batch_hash"],
                    "layout_sha256": layout_sha256(b["sequences"]), "stats": b["stats"],
                    "ms": (time.perf_counter() - ts) * 1e3})
    secs = time.perf_counter() - t0
    atomic_write_json(os.path.join(build.paths(art)["reports"], "reference_stream.json"),
                      {"purpose": "expected batch stream of an uninterrupted run (pipeline only)",
                       "plan_sha256": env.plan["plan_sha256"], "batches": ref})
    util = sum(x["stats"]["real_tokens"] for x in ref) / sum(x["stats"]["slots"] for x in ref)
    log.info("batches_packed", f"{T} batches x {C.TRAIN['batch_size']} x {C.TRAIN['seq_len']} "
             f"packed in {secs:.2f}s; slot utilization {util:.1%} -> reports/reference_stream.json")
    return secs
