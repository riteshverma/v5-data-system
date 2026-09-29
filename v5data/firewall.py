"""Evaluation and validation firewall.

Two independent layers:

1. **Shard layer** - only shards whose manifest split is ``train`` may be
   registered as a training source. Anything else raises FirewallViolation
   (checked again on every materialized batch, right before the optimizer).
2. **Content layer** - training documents whose content hash equals an
   evaluation document, or whose token n-grams overlap an evaluation document
   above a threshold, are flagged. OPUS hard-rejects flagged documents and the
   protected-floor override can never promote them.

Validation (``val``) and evaluation (``eval``) data can only be read through
``HeldOutReader``, which runs the model under ``torch.no_grad`` and never
produces a loss that reaches an optimizer.
"""
from __future__ import annotations

TRAINABLE_SPLITS = frozenset({"train"})
HELD_OUT_SPLITS = frozenset({"val", "eval"})


class FirewallViolation(RuntimeError):
    pass


def assert_trainable(manifest: dict) -> None:
    if manifest["split"] not in TRAINABLE_SPLITS:
        raise FirewallViolation(
            f"shard {manifest['shard_id']} has split={manifest['split']!r}; "
            f"only {sorted(TRAINABLE_SPLITS)} may feed a loss-bearing batch")


def assert_batch_trainable(catalog, sequences: list) -> None:
    for seq in sequences:
        for seg in seq["segments"]:
            split = catalog.split_of(seg["shard"])
            if split not in TRAINABLE_SPLITS:
                raise FirewallViolation(f"segment {seg['shard']}/{seg['doc_idx']} is {split}")


def _ngrams(tokens, n: int) -> set:
    t = [int(x) for x in tokens]
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def build_contamination_report(catalog, n: int, threshold: float) -> dict:
    """Scan every training document against the held-out (eval) set."""
    eval_hashes, eval_grams = {}, set()
    for sid in catalog.shards_where(split="eval"):
        for row in catalog.manifests[sid]["docs"]:
            eval_hashes[row["content_sha256"]] = row["doc_id"]
            eval_grams |= _ngrams(catalog.doc_tokens(sid, row["doc_idx"])[:-1], n)
    flagged, overlaps = {}, []
    for sid in catalog.shards_where(split="train"):
        for row in catalog.manifests[sid]["docs"]:
            toks = catalog.doc_tokens(sid, row["doc_idx"])[:-1]
            grams = _ngrams(toks, n)
            ov = (len(grams & eval_grams) / len(grams)) if grams else 0.0
            key = f"{sid}/{row['doc_idx']}"
            overlaps.append(ov)
            if row["content_sha256"] in eval_hashes:
                flagged[key] = {"doc_id": row["doc_id"], "reason": "eval_exact_match",
                                "eval_doc_id": eval_hashes[row["content_sha256"]],
                                "overlap": round(ov, 4)}
            elif ov >= threshold:
                flagged[key] = {"doc_id": row["doc_id"], "reason": "eval_ngram_overlap",
                                "overlap": round(ov, 4)}
    clean = sorted(o for k, o in zip(range(len(overlaps)), overlaps))
    return {"schema": "v5.contamination/1", "ngram": n, "threshold": threshold,
            "eval_docs": len(eval_hashes), "eval_ngrams": len(eval_grams),
            "train_docs_scanned": len(overlaps), "flagged": flagged,
            "overlap_p99_all_train": clean[int(0.99 * (len(clean) - 1))] if clean else 0.0}
