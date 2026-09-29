"""Compile curriculum stages + lane weights + protected floors into a per-step plan.

The plan is fully materialized before training ("mixture compiled"): for every
step it states the stage, the OPUS thresholds, and exactly how many sequences
each lane must contribute. It is written to ``manifests/mixture_plan.json`` and
its hash is pinned in every checkpoint, so a resume or replay under a different
mixture is refused.

Apportionment uses integer smooth weighted round-robin inside each stage, so
cumulative lane shares track the target weights with error < 1 sequence at
every step, and the result is exactly reproducible.
"""
from __future__ import annotations

import math

from .util import sha256_json


class PlanError(RuntimeError):
    pass


def _swrr_counts(weights: dict, lane_order: list, batch_size: int, cur: dict) -> dict:
    w = {l: int(round(weights.get(l, 0.0) * 1000)) for l in lane_order}
    total = sum(w.values())
    counts = {l: 0 for l in lane_order}
    for _ in range(batch_size):
        for l in lane_order:
            cur[l] += w[l]
        pick = max(lane_order, key=lambda l: (cur[l], -lane_order.index(l)))
        cur[pick] -= total
        counts[pick] += 1
    return counts


def compile_plan(stages: list, lanes: dict, lane_order: list, batch_size: int,
                 total_steps: int, data_config_sha256: str) -> dict:
    # --- validate stage coverage
    expect = 1
    for st in stages:
        if st["start"] != expect or st["end"] < st["start"]:
            raise PlanError(f"stage {st['name']} does not start at step {expect}")
        if abs(sum(st["weights"].values()) - 1.0) > 1e-9:
            raise PlanError(f"stage {st['name']} weights do not sum to 1")
        expect = st["end"] + 1
    if expect <= total_steps:
        raise PlanError("stages do not cover every step")

    floors = {l: math.ceil(lanes[l]["floor"] * batch_size - 1e-9) if lanes[l]["protected"] else 0
              for l in lane_order}
    if sum(floors.values()) > batch_size:
        raise PlanError("protected floors exceed batch size")
    backfill = [l for l in lane_order if lanes[l].get("backfill")]
    steps = []
    for st in stages:
        cur = {l: 0 for l in lane_order}
        for step in range(st["start"], min(st["end"], total_steps) + 1):
            counts = _swrr_counts(st["weights"], lane_order, batch_size, cur)
            adjustments = []
            for l in lane_order:
                while counts[l] < floors[l]:
                    donors = sorted((l2 for l2 in lane_order if counts[l2] > floors[l2]),
                                    key=lambda l2: (l2 not in backfill, -counts[l2]))
                    donor = donors[0]
                    counts[donor] -= 1
                    counts[l] += 1
                    adjustments.append({"lane": l, "from": donor, "reason": "protected_floor"})
            steps.append({"step": step, "stage": st["name"], "weights": st["weights"],
                          "opus": st["opus"], "seqs": counts, "floor_adjustments": adjustments})

    summary = {}
    for st in stages:
        rows = [s for s in steps if s["stage"] == st["name"]]
        if not rows:
            continue
        tot = sum(sum(r["seqs"].values()) for r in rows)
        summary[st["name"]] = {
            "steps": [rows[0]["step"], rows[-1]["step"]],
            "target_weights": st["weights"],
            "planned_seq_share": {l: round(sum(r["seqs"][l] for r in rows) / tot, 4)
                                  for l in lane_order},
            "floor_adjusted_steps": sum(1 for r in rows if r["floor_adjustments"]),
        }
    plan = {"schema": "v5.mixture_plan/1", "batch_size": batch_size,
            "total_steps": total_steps, "lane_order": lane_order, "floors_seqs": floors,
            "backfill_lanes": backfill, "stages": stages, "data_config_sha256": data_config_sha256,
            "summary": summary, "steps": steps}
    plan["plan_sha256"] = sha256_json({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


def verify_plan(plan: dict) -> list:
    """Return a list of violations (empty == compliant)."""
    errs = []
    if sha256_json({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        errs.append("plan hash mismatch")
    for s in plan["steps"]:
        if sum(s["seqs"].values()) != plan["batch_size"]:
            errs.append(f"step {s['step']}: seqs do not sum to batch size")
        for l, f in plan["floors_seqs"].items():
            if s["seqs"][l] < f:
                errs.append(f"step {s['step']}: {l} below floor")
    return errs
