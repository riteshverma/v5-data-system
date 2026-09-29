"""Offline build stages and the verified environment loader.

documents -> frozen tokenizer -> immutable shards + manifests -> catalog
          -> contamination scan (firewall) -> compiled mixture plan
"""
from __future__ import annotations

import os

from . import config as C
from .corpus import generate, write_raw
from .firewall import build_contamination_report
from .mixture import compile_plan
from .shards import Catalog, validate_all, write_catalog, write_shard
from .tokenizer import FrozenTokenizer
from .util import atomic_write_json, read_json, read_jsonl, sha256_json


def paths(art: str) -> dict:
    return {
        "raw": os.path.join(art, "corpus", "raw"),
        "tokenizer": os.path.join(art, "tokenizer", "tokenizer.json"),
        "tokenizer_lock": os.path.join(art, "tokenizer", "tokenizer.lock.json"),
        "shards": os.path.join(art, "shards"),
        "shard_manifests": os.path.join(art, "manifests", "shards"),
        "catalog": os.path.join(art, "manifests", "catalog.json"),
        "validation": os.path.join(art, "manifests", "validation_report.json"),
        "contamination": os.path.join(art, "manifests", "contamination_report.json"),
        "plan": os.path.join(art, "manifests", "mixture_plan.json"),
        "ledgers": os.path.join(art, "ledgers"),
        "checkpoints": os.path.join(art, "checkpoints"),
        "reports": os.path.join(art, "reports"),
    }


def tokenizer_training_text(d: dict) -> str:
    return d["prompt"] + " " + d["response"] if d["lane"] == "instruct" else d["text"]


def build_corpus(art: str) -> tuple:
    corpus = generate(C.CORPUS)
    file_hashes = write_raw(corpus, paths(art)["raw"])
    atomic_write_json(os.path.join(art, "manifests", "corpus_manifest.json"), {
        "schema": "v5.corpus/1", "seed": C.CORPUS["seed"], "files": file_hashes,
        "counts": {f"{l}.{s}": len(d) for (l, s), d in sorted(corpus.items())},
        "origins": {f"{l}.{s}": _count(d, "origin") for (l, s), d in sorted(corpus.items())}})
    return corpus, file_hashes


def _count(docs, key):
    out = {}
    for d in docs:
        out[d[key]] = out.get(d[key], 0) + 1
    return out


def build_tokenizer(art: str, corpus: dict) -> tuple:
    texts = [tokenizer_training_text(d) for (l, s), docs in sorted(corpus.items())
             if s in C.TOKENIZER["train_splits"] for d in docs]
    tok = FrozenTokenizer.train(texts, C.TOKENIZER["n_merges"])
    p = paths(art)
    lock = tok.save_frozen(p["tokenizer"], p["tokenizer_lock"], {
        "trained_on_splits": C.TOKENIZER["train_splits"], "training_docs": len(texts),
        "training_text_sha256": sha256_json(texts), "n_merges": C.TOKENIZER["n_merges"]})
    return tok, lock


def build_shards(art: str, corpus: dict, tok, tok_sha: str) -> list:
    p, per = paths(art), C.CORPUS["docs_per_shard"]
    manifests = []
    for (lane, split), docs in sorted(corpus.items()):
        for k in range(0, len(docs), per):
            sid = f"{lane}-{split}-{k // per:03d}"
            manifests.append(write_shard(p["shards"], p["shard_manifests"], sid, lane, split,
                                         docs[k:k + per], tok, tok_sha))
    write_catalog(art, manifests, tok_sha)
    return manifests


def raw_doc_index(art: str) -> dict:
    out = {}
    raw = paths(art)["raw"]
    for fn in sorted(os.listdir(raw)):
        for d in read_jsonl(os.path.join(raw, fn)):
            out[d["doc_id"]] = d
    return out


def validate(art: str, tok, tok_sha: str) -> dict:
    rep = validate_all(art, tok, tok_sha, raw_doc_index(art))
    atomic_write_json(paths(art)["validation"], rep)
    return rep


def build_contamination(art: str, catalog) -> dict:
    rep = build_contamination_report(catalog, C.OPUS["contamination_ngram"],
                                     C.OPUS["contamination_threshold"])
    # n-gram overlap is only meaningful for prose lanes; templated lanes rely on hashes
    ngram_lanes = set(C.OPUS["ngram_lanes"])
    rep["flagged"] = {k: v for k, v in rep["flagged"].items()
                      if v["reason"] == "eval_exact_match"
                      or catalog.manifests[k.rsplit("/", 1)[0]]["lane"] in ngram_lanes}
    rep["ngram_lanes"] = sorted(ngram_lanes)
    atomic_write_json(paths(art)["contamination"], rep)
    return rep


def compile_mixture(art: str, stages=None, total_steps=None, out_path=None) -> dict:
    plan = compile_plan(stages or C.STAGES, C.LANES, C.LANE_ORDER, C.TRAIN["batch_size"],
                        total_steps or C.TRAIN["total_steps"], sha256_json(C.data_config()))
    atomic_write_json(out_path or paths(art)["plan"], plan)
    return plan


class Env:
    """Everything a training / replay / audit process needs, integrity-checked."""

    def __init__(self, art: str, plan_path: str | None = None):
        p = paths(art)
        self.art = art
        self.tok = FrozenTokenizer.load_frozen(p["tokenizer"], p["tokenizer_lock"])
        self.tok_sha = self.tok.sha256
        self.catalog = Catalog(art, self.tok_sha, verify=True)
        self.contamination = read_json(p["contamination"])
        self.plan = read_json(plan_path or p["plan"])
        from .mixture import verify_plan
        errs = verify_plan(self.plan)
        if errs:
            raise RuntimeError(f"mixture plan invalid: {errs[:3]}")

    def pipeline(self):
        from .pipeline import DataPipeline
        return DataPipeline(self.catalog, self.plan, C.LANES, C.OPUS, C.PACKING,
                            C.TRAIN["seq_len"], C.SEED, self.contamination)
