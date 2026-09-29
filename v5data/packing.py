"""Packing policies, batch materialization, mask construction and fingerprints.

Policies (one per data type):

``concat_split_docmask``  (web prose)
    Documents are concatenated and cut at exact ``seq_len`` boundaries; a
    document may continue in the next sequence. 100% slot utilization.
``bestfit_nosplit_docmask``  (code)
    A document is never split across sequences (functions stay intact).
    Best-fit bin packing over a bounded set of open bins; documents longer than
    ``seq_len`` are truncated (recorded in the span). Remaining slots are padded.
``bestfit_nosplit_response_only``  (instruct / chat)
    Same bin packing, but loss is only taken on the assistant response
    (``loss_start`` from the manifest), never on the prompt.

Every policy uses *document masking*: segment ids label each packed document,
attention is causal **within** a segment only, position ids restart at 0 at
each segment, and the last token of every segment carries no loss (its next
token belongs to a different document or to the next sequence).

A packed sequence is fully described by its list of spans
``(shard, doc_idx, start, end)``; materialization from spans is a pure
function, which is what makes ledger-driven replay possible.
"""
from __future__ import annotations

import numpy as np

from .tokenizer import PAD
from .util import canonical_json, sha256_bytes

IGNORE = PAD  # label value at non-loss positions (masked out anyway)


# ------------------------------------------------------------------ packers
class ConcatSplitPacker:
    policy = "concat_split_docmask"

    def __init__(self, seq_len: int, **_):
        self.L = seq_len
        self.buffer: list = []   # [shard, doc_idx, next_start, end]

    def add(self, shard: str, doc_idx: int, length: int) -> None:
        self.buffer.append([shard, doc_idx, 0, length])

    def buffered_tokens(self) -> int:
        return sum(b - a for _, _, a, b in self.buffer)

    def ready_count(self) -> int:
        return self.buffered_tokens() // self.L

    def emit(self) -> dict:
        assert self.ready_count() >= 1
        spans, room = [], self.L
        while room > 0:
            s, i, a, b = self.buffer[0]
            take = min(room, b - a)
            spans.append([s, i, a, a + take])
            room -= take
            if a + take == b:
                self.buffer.pop(0)
            else:
                self.buffer[0][2] = a + take
        return {"spans": spans, "pad": 0}

    def get_state(self) -> dict:
        return {"buffer": [list(x) for x in self.buffer]}

    def set_state(self, st: dict) -> None:
        self.buffer = [list(x) for x in st["buffer"]]


class BestFitPacker:
    policy = "bestfit_nosplit_docmask"

    def __init__(self, seq_len: int, max_open_bins: int = 4, min_residual: int = 4):
        self.L, self.max_open, self.min_residual = seq_len, max_open_bins, min_residual
        self.open: list = []     # [{"spans": [[s,i,a,b]], "used": n}]
        self.closed: list = []

    def _close(self, k: int) -> None:
        self.closed.append(self.open.pop(k))

    def add(self, shard: str, doc_idx: int, length: int) -> None:
        n = min(length, self.L)                    # truncate over-long documents
        span = [shard, doc_idx, 0, n]
        fits = [k for k, b in enumerate(self.open) if self.L - b["used"] >= n]
        if fits:
            k = min(fits, key=lambda k: (self.L - self.open[k]["used"] - n, k))
        else:
            if len(self.open) >= self.max_open:     # evict the fullest open bin
                full = max(range(len(self.open)), key=lambda k: (self.open[k]["used"], -k))
                self._close(full)
            self.open.append({"spans": [], "used": 0})
            k = len(self.open) - 1
        self.open[k]["spans"].append(span)
        self.open[k]["used"] += n
        if self.L - self.open[k]["used"] < self.min_residual:
            self._close(k)

    def buffered_tokens(self) -> int:
        return sum(b["used"] for b in self.open + self.closed)

    def ready_count(self) -> int:
        return len(self.closed)

    def emit(self) -> dict:
        b = self.closed.pop(0)
        return {"spans": b["spans"], "pad": self.L - b["used"]}

    def get_state(self) -> dict:
        cp = lambda bins: [{"spans": [list(s) for s in b["spans"]], "used": b["used"]} for b in bins]
        return {"open": cp(self.open), "closed": cp(self.closed)}

    def set_state(self, st: dict) -> None:
        self.open = [{"spans": [list(s) for s in b["spans"]], "used": b["used"]} for b in st["open"]]
        self.closed = [{"spans": [list(s) for s in b["spans"]], "used": b["used"]}
                       for b in st["closed"]]


class ResponseOnlyPacker(BestFitPacker):
    policy = "bestfit_nosplit_response_only"


POLICIES = {c.policy: c for c in (ConcatSplitPacker, BestFitPacker, ResponseOnlyPacker)}


def make_packer(policy: str, seq_len: int, packing_cfg: dict):
    return POLICIES[policy](seq_len, **packing_cfg)


# ----------------------------------------------------------- materialize
def materialize(sequences: list, catalog, seq_len: int) -> dict:
    """Turn span lists into model inputs. Pure function of (spans, shards)."""
    B, L = len(sequences), seq_len
    tokens = np.full((B, L), PAD, dtype=np.int64)
    labels = np.full((B, L), IGNORE, dtype=np.int64)
    loss_mask = np.zeros((B, L), dtype=np.uint8)
    pos = np.zeros((B, L), dtype=np.int64)
    seg = np.zeros((B, L), dtype=np.int32)
    for r, sq in enumerate(sequences):
        p = 0
        for k, sp in enumerate(sq["segments"], 1):
            doc = catalog.doc_tokens(sp["shard"], sp["doc_idx"])
            a, b = sp["start"], sp["end"]
            t = doc[a:b].astype(np.int64)
            n = t.size
            tokens[r, p:p + n] = t
            pos[r, p:p + n] = np.arange(n)
            seg[r, p:p + n] = k
            if n > 1:
                labels[r, p:p + n - 1] = t[1:]
                lm = np.ones(n - 1, dtype=np.uint8)
                ls = catalog.doc_meta(sp["shard"], sp["doc_idx"])["loss_start"]
                if ls is not None and sq["policy"] == "bestfit_nosplit_response_only":
                    # position j predicts doc token a+j+1; loss only on the response
                    lm = ((a + 1 + np.arange(n - 1)) >= ls).astype(np.uint8)
                loss_mask[r, p:p + n - 1] = lm
            p += n
        assert p <= L, "sequence overflow"
    return {"tokens": tokens, "labels": labels, "loss_mask": loss_mask,
            "position_ids": pos, "segment_ids": seg}


def attention_mask(segment_ids: np.ndarray) -> np.ndarray:
    """Block-diagonal causal mask [B, L, L]: i may attend j iff same segment and j <= i.
    Pad positions (segment 0) attend only to themselves (keeps softmax finite)."""
    s = segment_ids
    L = s.shape[1]
    same = (s[:, :, None] == s[:, None, :]) & (s[:, :, None] > 0)
    causal = np.tril(np.ones((L, L), dtype=bool))[None]
    eye = np.eye(L, dtype=bool)[None]
    return (same & causal) | (eye & (s[:, :, None] == 0))


def batch_fingerprint(sequences: list, arrays: dict) -> str:
    layout = [{"lane": sq["lane"], "policy": sq["policy"],
               "spans": [[sp["shard"], sp["doc_idx"], sp["start"], sp["end"]]
                         for sp in sq["segments"]]} for sq in sequences]
    parts = [canonical_json(layout).encode()]
    for k in ("tokens", "labels", "loss_mask", "position_ids", "segment_ids"):
        parts.append(k.encode() + np.ascontiguousarray(arrays[k], dtype="<i4").tobytes())
    return sha256_bytes(b"|".join(parts))


def batch_stats(arrays: dict) -> dict:
    seg = arrays["segment_ids"]
    return {"slots": int(seg.size), "real_tokens": int((seg > 0).sum()),
            "pad_tokens": int((seg == 0).sum()), "loss_tokens": int(arrays["loss_mask"].sum())}


# --------------------------------------------------------------- invariants
def check_invariants(sequences: list, arrays: dict, catalog) -> list:
    """Independent re-check of every mask property. Returns list of violations."""
    errs = []
    tok, lab, lm = arrays["tokens"], arrays["labels"], arrays["loss_mask"]
    pos, seg = arrays["position_ids"], arrays["segment_ids"]
    L = tok.shape[1]
    for r, sq in enumerate(sequences):
        p = 0
        for k, sp in enumerate(sq["segments"], 1):
            meta = catalog.doc_meta(sp["shard"], sp["doc_idx"])
            if catalog.split_of(sp["shard"]) != "train":
                errs.append(f"row {r}: non-train shard {sp['shard']}")
            if not (0 <= sp["start"] < sp["end"] <= meta["length"]):
                errs.append(f"row {r} seg {k}: bad span")
                continue
            n = sp["end"] - sp["start"]
            if sq["policy"] != "concat_split_docmask" and sp["start"] != 0:
                errs.append(f"row {r} seg {k}: no-split policy produced a split document")
            want = catalog.doc_tokens(sp["shard"], sp["doc_idx"])[sp["start"]:sp["end"]]
            if not np.array_equal(tok[r, p:p + n], want):
                errs.append(f"row {r} seg {k}: tokens differ from shard span")
            if not np.array_equal(pos[r, p:p + n], np.arange(n)):
                errs.append(f"row {r} seg {k}: position ids do not restart at segment start")
            if not np.all(seg[r, p:p + n] == k):
                errs.append(f"row {r} seg {k}: segment ids wrong")
            if lm[r, p + n - 1] != 0:
                errs.append(f"row {r} seg {k}: last token of segment carries loss")
            for j in range(n - 1):
                if lm[r, p + j] and lab[r, p + j] != tok[r, p + j + 1]:
                    errs.append(f"row {r} seg {k}: label is not next token at {j}")
                    break
            if sq["policy"] == "bestfit_nosplit_response_only" and meta["loss_start"] is not None:
                # no loss on any label that is a prompt token
                tgt = sp["start"] + 1 + np.arange(n - 1)
                if np.any(lm[r, p:p + n - 1][tgt < meta["loss_start"]]):
                    errs.append(f"row {r} seg {k}: loss on prompt tokens")
                if not np.all(lm[r, p:p + n - 1][tgt >= meta["loss_start"]]):
                    errs.append(f"row {r} seg {k}: response token without loss")
            p += n
        if np.any(seg[r, p:] != 0) or np.any(lm[r, p:] != 0) or np.any(tok[r, p:] != PAD):
            errs.append(f"row {r}: padding region not clean")
        if p != L - sq["pad"]:
            errs.append(f"row {r}: pad count mismatch")
    am = attention_mask(seg)
    for r in range(am.shape[0]):
        a = am[r]
        si, sj = seg[r][:, None], seg[r][None, :]
        cross = a & (si != sj)
        future = a & ~np.tril(np.ones((L, L), dtype=bool))
        if cross.any():
            errs.append(f"row {r}: attention crosses documents")
        if future.any():
            errs.append(f"row {r}: attention sees the future")
    return errs
