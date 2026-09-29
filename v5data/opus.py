"""OPUS - the online admission gate between lane streams and packers.

For every candidate document OPUS records exactly one decision:

``ACCEPT``                 score >= stage accept threshold
``DEFER``                  reject <= score < accept: re-queued for the next step
                           with an aging bonus (anti-starvation); after
                           ``max_defers`` deferrals it expires (REJECT defer_expired)
``REJECT``                 hard: evaluation contamination, duplicate content
                           soft: score below the stage reject threshold
``ACCEPT_FLOOR_OVERRIDE``  a deferred candidate of a *protected* lane promoted
                           because the lane would otherwise fall below its
                           protected floor within its scan budget. Hard rejects
                           are never overridable.

The score is a pure function of the manifest quality, the deferral count and
the stage thresholds, so the auditor can re-derive every decision.
"""
from __future__ import annotations

ACCEPT, DEFER, REJECT, OVERRIDE = "ACCEPT", "DEFER", "REJECT", "ACCEPT_FLOOR_OVERRIDE"
HARD_REASONS = ("eval_contamination", "duplicate_content")


class OpusGate:
    def __init__(self, cfg: dict, flagged: dict):
        self.max_defers = cfg["max_defers"]
        self.aging = cfg["aging_bonus"]
        self.flagged = flagged                 # doc_key -> contamination info
        self.accepted_hashes: set = set()

    def score(self, meta: dict, defer_count: int) -> float:
        return round(meta["quality"] + self.aging * defer_count, 4)

    def evaluate(self, key: str, meta: dict, defer_count: int, th: dict) -> tuple:
        """-> (decision, reason, score). Does not mutate state."""
        s = self.score(meta, defer_count)
        if key in self.flagged:
            return REJECT, "eval_contamination:" + self.flagged[key]["reason"], s
        if meta["content_sha256"] in self.accepted_hashes:
            return REJECT, "duplicate_content", s
        if s < th["reject"]:
            return REJECT, "low_quality", s
        if s < th["accept"]:
            if defer_count >= self.max_defers:
                return REJECT, "defer_expired", s
            return DEFER, "below_accept_threshold", s
        return ACCEPT, "above_accept_threshold", s

    def admit(self, meta: dict) -> None:
        self.accepted_hashes.add(meta["content_sha256"])

    def get_state(self) -> dict:
        return {"accepted_hashes": sorted(self.accepted_hashes)}

    def set_state(self, st: dict) -> None:
        self.accepted_hashes = set(st["accepted_hashes"])


def replay_decision(rec: dict, cfg: dict, flagged: dict, accepted_before: set) -> tuple:
    """Auditor helper: recompute a recorded decision from its inputs."""
    g = OpusGate(cfg, flagged)
    g.accepted_hashes = accepted_before
    return g.evaluate(rec["doc_key"], {"quality": rec["quality"],
                                       "content_sha256": rec["content_sha256"]},
                      rec["defer_count"], rec["thresholds"])
