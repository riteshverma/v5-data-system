"""Deterministic document quality heuristic (the OPUS utility signal).

Computed once at shard-build time from the raw text and stored in the shard
manifest, so OPUS decisions are a pure function of (manifest, stage, state)
and can be re-derived by the auditor.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z]+")


def quality_score(text: str) -> float:
    words = _WORD.findall(text)
    if not words:
        return 0.0
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    nonspace = [c for c in text if not c.isspace()]
    alpha_frac = sum(c.isalpha() for c in nonspace) / max(1, len(nonspace))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    line_dup = 1.0 - len(set(lines)) / len(lines) if len(lines) > 1 else 0.0
    length_term = min(1.0, len(words) / 30.0)
    q = 0.80 * unique_ratio + 0.10 * alpha_frac + 0.10 * length_term - 0.40 * line_dup
    return round(max(0.0, min(1.0, q)), 4)
