"""Packing efficiency and throughput, computed only from ledgers + shards.

Every number in ``performance.json`` is derived from the committed ledgers
(token counts from the recorded spans / loss masks, time from per-step timing
recorded in the learning ledger), so the auditor can recompute it.
"""
from __future__ import annotations

import math

from . import config as C
from .packing import materialize
from .trainer import sequences_from_record


def packing_metrics(env, cons_records: list) -> dict:
    L = C.TRAIN["seq_len"]
    per_policy: dict = {}
    tot = {"sequences": 0, "slots": 0, "real_tokens": 0, "pad_tokens": 0, "loss_tokens": 0,
           "truncated_tokens_dropped": 0, "split_documents": 0, "segments": 0}
    doc_tokens: dict = {}
    for rec in cons_records:
        seqs = sequences_from_record(rec)
        arrays = materialize(seqs, env.catalog, L)
        for r, sq in enumerate(seqs):
            p = per_policy.setdefault(sq["policy"], {"lane": sq["lane"], "sequences": 0,
                                                     "slots": 0, "real_tokens": 0,
                                                     "loss_tokens": 0, "segments": 0})
            real = int((arrays["segment_ids"][r] > 0).sum())
            loss = int(arrays["loss_mask"][r].sum())
            for k, v in (("sequences", 1), ("slots", L), ("real_tokens", real),
                         ("loss_tokens", loss), ("segments", len(sq["segments"]))):
                p[k] += v
                if k in tot:
                    tot[k] += v
            tot["pad_tokens"] += L - real
            for g in sq["segments"]:
                n = env.catalog.doc_meta(g["shard"], g["doc_idx"])["length"]
                if sq["policy"] != "concat_split_docmask" and g["end"] < n:
                    tot["truncated_tokens_dropped"] += n - g["end"]
                if sq["policy"] == "concat_split_docmask" and g["start"] > 0:
                    tot["split_documents"] += 1
                key = (g["shard"], g["doc_idx"])
                doc_tokens[key] = doc_tokens.get(key, 0) + g["end"] - g["start"]
    for p in per_policy.values():
        p["utilization"] = p["real_tokens"] / p["slots"]
        p["loss_bearing_fraction"] = p["loss_tokens"] / p["slots"]
        p["docs_per_sequence"] = p["segments"] / p["sequences"]
    # naive baseline: each consumed document alone in ceil(len / L) padded sequences
    naive_seqs = sum(math.ceil(n / L) for n in doc_tokens.values())
    tot["utilization"] = tot["real_tokens"] / tot["slots"]
    tot["loss_bearing_fraction"] = tot["loss_tokens"] / tot["slots"]
    naive = {"sequences_needed": naive_seqs, "slots": naive_seqs * L,
             "utilization": tot["real_tokens"] / (naive_seqs * L),
             "loss_bearing_fraction": tot["loss_tokens"] / (naive_seqs * L)}
    return {"totals": tot, "per_policy": per_policy, "naive_one_doc_per_sequence": naive,
            "sequence_savings_vs_naive": 1 - tot["sequences"] / naive_seqs}


def throughput_metrics(learn_records: list) -> dict:
    data = sum(r["timing_ms"]["data"] for r in learn_records) / 1e3
    ledger = sum(r["timing_ms"]["ledger"] for r in learn_records) / 1e3
    compute = sum(r["timing_ms"]["compute"] for r in learn_records) / 1e3
    wall = data + ledger + compute
    slots = len(learn_records) * C.TRAIN["batch_size"] * C.TRAIN["seq_len"]
    loss_tokens = sum(r["loss_tokens"] for r in learn_records)
    per_step = sorted(r["timing_ms"]["data"] + r["timing_ms"]["ledger"] +
                      r["timing_ms"]["compute"] for r in learn_records)
    return {"steps": len(learn_records), "seconds": {"data": data, "ledger": ledger,
                                                     "compute": compute, "total": wall},
            "slot_tokens_per_sec": slots / wall,
            "useful_loss_bearing_tokens_per_sec": loss_tokens / wall,
            "loss_tokens": loss_tokens, "slot_tokens": slots,
            "data_pipeline_share_of_step_time": data / wall,
            "median_step_ms": per_step[len(per_step) // 2]}
