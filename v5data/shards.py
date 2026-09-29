"""Immutable tokenized shards, their manifests, and the validated catalog.

A shard is two binary files written exactly once and then made read-only:

    <shard_id>.tokens.bin   uint16 token ids of all documents, concatenated
    <shard_id>.index.bin    int64 document offsets (num_docs + 1 entries)

and a manifest ``manifests/shards/<shard_id>.json`` that records the tokenizer
hash, split, lane, file hashes and a per-document table (content hash of the
raw text, hash of the token ids, offset, length, loss region, quality score).
The manifest hashes itself (``manifest_sha256`` over every other field), so
tampering with either the data or the manifest is detectable.
"""
from __future__ import annotations

import os

import numpy as np

from .corpus import doc_text
from .quality import quality_score
from .util import (atomic_write_json, is_readonly, make_readonly,
                   read_json, sha256_bytes, sha256_file, sha256_json)


class ShardIntegrityError(RuntimeError):
    pass


class ShardExistsError(RuntimeError):
    pass


def _manifest_hash(m: dict) -> str:
    return sha256_json({k: v for k, v in m.items() if k != "manifest_sha256"})


def write_shard(shard_dir: str, manifest_dir: str, shard_id: str, lane: str, split: str,
                docs: list, tokenizer, tokenizer_sha256: str) -> dict:
    tok_path = os.path.join(shard_dir, f"{shard_id}.tokens.bin")
    idx_path = os.path.join(shard_dir, f"{shard_id}.index.bin")
    man_path = os.path.join(manifest_dir, f"{shard_id}.json")
    for p in (tok_path, idx_path, man_path):
        if os.path.exists(p):   # shards are write-once
            raise ShardExistsError(f"refusing to overwrite immutable shard file {p}")
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    all_ids, offsets, table = [], [0], []
    for d in docs:
        ids, loss_start = tokenizer.encode_document(d)
        arr = np.asarray(ids, dtype=np.uint16)
        table.append({
            "doc_idx": len(table), "doc_id": d["doc_id"],
            "content_sha256": d["content_sha256"],
            "token_sha256": sha256_bytes(arr.tobytes()),
            "offset": offsets[-1], "length": int(arr.size),
            "loss_start": loss_start,
            "quality": quality_score(doc_text(d)),
        })
        all_ids.append(arr)
        offsets.append(offsets[-1] + int(arr.size))
    tokens = np.concatenate(all_ids).astype(np.uint16)
    index = np.asarray(offsets, dtype=np.int64)
    for path, arr in ((tok_path, tokens), (idx_path, index)):
        with open(path, "wb") as f:
            f.write(arr.tobytes())
            f.flush()
            os.fsync(f.fileno())
        make_readonly(path)

    m = {
        "schema": "v5.shard/1", "shard_id": shard_id, "lane": lane, "split": split,
        "tokenizer_sha256": tokenizer_sha256, "dtype": "uint16",
        "num_docs": len(docs), "num_tokens": int(tokens.size),
        "files": {
            "tokens": {"path": f"shards/{shard_id}.tokens.bin",
                       "sha256": sha256_file(tok_path), "bytes": os.path.getsize(tok_path)},
            "index": {"path": f"shards/{shard_id}.index.bin",
                      "sha256": sha256_file(idx_path), "bytes": os.path.getsize(idx_path)},
        },
        "docs": table,
    }
    m["manifest_sha256"] = _manifest_hash(m)
    atomic_write_json(man_path, m, indent=None)
    make_readonly(man_path)
    return m


class ShardReader:
    """Verified, memory-mapped read access to one shard."""

    def __init__(self, art_dir: str, manifest: dict, verify: bool = True):
        self.m = manifest
        self.tok_path = os.path.join(art_dir, manifest["files"]["tokens"]["path"])
        self.idx_path = os.path.join(art_dir, manifest["files"]["index"]["path"])
        if verify:
            self.verify()
        self.tokens = np.memmap(self.tok_path, dtype=np.uint16, mode="r")
        self.index = np.fromfile(self.idx_path, dtype=np.int64)

    def verify(self) -> None:
        if _manifest_hash(self.m) != self.m["manifest_sha256"]:
            raise ShardIntegrityError(f"{self.m['shard_id']}: manifest self-hash mismatch")
        for key, path in (("tokens", self.tok_path), ("index", self.idx_path)):
            want = self.m["files"][key]["sha256"]
            got = sha256_file(path)
            if got != want:
                raise ShardIntegrityError(
                    f"{self.m['shard_id']}: {key} file hash {got[:12]} != manifest {want[:12]}")

    def doc_tokens(self, doc_idx: int) -> np.ndarray:
        a, b = int(self.index[doc_idx]), int(self.index[doc_idx + 1])
        return np.asarray(self.tokens[a:b])


class Catalog:
    """All shard manifests, validated against the frozen tokenizer.

    ``doc(key)`` gives O(1) access to a document's manifest row and tokens,
    where key = "<shard_id>/<doc_idx>".
    """

    def __init__(self, art_dir: str, tokenizer_sha256: str, verify: bool = True):
        self.art_dir = art_dir
        self.tokenizer_sha256 = tokenizer_sha256
        cat = read_json(os.path.join(art_dir, "manifests", "catalog.json"))
        self.catalog = cat
        self.manifests, self.readers = {}, {}
        for sid in cat["shards"]:
            m = read_json(os.path.join(art_dir, "manifests", "shards", f"{sid}.json"))
            if m["manifest_sha256"] != cat["shards"][sid]["manifest_sha256"]:
                raise ShardIntegrityError(f"{sid}: catalog pins a different manifest hash")
            if m["tokenizer_sha256"] != tokenizer_sha256:
                raise ShardIntegrityError(f"{sid}: built with tokenizer "
                                          f"{m['tokenizer_sha256'][:12]}, frozen is "
                                          f"{tokenizer_sha256[:12]}")
            self.manifests[sid] = m
            self.readers[sid] = ShardReader(art_dir, m, verify=verify)
        if catalog_hash(cat["shards"]) != cat["catalog_sha256"]:
            raise ShardIntegrityError("catalog self-hash mismatch")

    def shards_where(self, **kv) -> list:
        return [sid for sid, m in sorted(self.manifests.items())
                if all(m[k] == v for k, v in kv.items())]

    def doc_meta(self, shard_id: str, doc_idx: int) -> dict:
        return self.manifests[shard_id]["docs"][doc_idx]

    def doc_tokens(self, shard_id: str, doc_idx: int) -> np.ndarray:
        return self.readers[shard_id].doc_tokens(doc_idx)

    def split_of(self, shard_id: str) -> str:
        return self.manifests[shard_id]["split"]


def catalog_hash(shards: dict) -> str:
    return sha256_json({sid: v["manifest_sha256"] for sid, v in sorted(shards.items())})


def write_catalog(art_dir: str, manifests: list, tokenizer_sha256: str) -> dict:
    shards = {m["shard_id"]: {"lane": m["lane"], "split": m["split"],
                              "num_docs": m["num_docs"], "num_tokens": m["num_tokens"],
                              "manifest_sha256": m["manifest_sha256"]}
              for m in manifests}
    cat = {"schema": "v5.catalog/1", "tokenizer_sha256": tokenizer_sha256,
           "shards": shards, "catalog_sha256": catalog_hash(shards)}
    atomic_write_json(os.path.join(art_dir, "manifests", "catalog.json"), cat)
    return cat


def validate_all(art_dir: str, tokenizer, tokenizer_sha256: str, raw_docs: dict) -> dict:
    """Full validation: file hashes, self-hashes, read-only bits, tokenizer hash,
    and re-tokenization of every raw document reproducing the stored tokens."""
    cat = read_json(os.path.join(art_dir, "manifests", "catalog.json"))
    report = {"shards": {}, "ok": True}
    for sid in sorted(cat["shards"]):
        m = read_json(os.path.join(art_dir, "manifests", "shards", f"{sid}.json"))
        r = {"manifest_self_hash": _manifest_hash(m) == m["manifest_sha256"],
             "catalog_pin": cat["shards"][sid]["manifest_sha256"] == m["manifest_sha256"],
             "tokenizer_hash": m["tokenizer_sha256"] == tokenizer_sha256}
        try:
            reader = ShardReader(art_dir, m, verify=True)
            r["file_hashes"] = True
        except ShardIntegrityError:
            r["file_hashes"] = False
            reader = ShardReader(art_dir, m, verify=False)
        r["read_only"] = all(is_readonly(p) for p in (reader.tok_path, reader.idx_path))
        mismatch = 0
        for row in m["docs"]:
            raw = raw_docs[row["doc_id"]]
            ids, loss_start = tokenizer.encode_document(raw)
            stored = reader.doc_tokens(row["doc_idx"])
            if (sha256_bytes(np.asarray(ids, np.uint16).tobytes()) != row["token_sha256"]
                    or sha256_bytes(stored.tobytes()) != row["token_sha256"]
                    or loss_start != row["loss_start"]
                    or raw["content_sha256"] != row["content_sha256"]):
                mismatch += 1
        r["retokenize_mismatches"] = mismatch
        r["num_docs"] = m["num_docs"]
        r["ok"] = all(v for k, v in r.items() if isinstance(v, bool)) and mismatch == 0
        report["shards"][sid] = r
        report["ok"] &= r["ok"]
    report["catalog_self_hash"] = catalog_hash(cat["shards"]) == cat["catalog_sha256"]
    report["ok"] &= report["catalog_self_hash"]
    report["catalog_sha256"] = cat["catalog_sha256"]
    return report

