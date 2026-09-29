"""Append-only, hash-chained JSONL ledgers (a tiny transaction log).

Each line is ``{"seq": n, "prev": <hash of line n-1>, "data": {...}, "hash": h}``
with ``h = sha256(canonical_json({seq, prev, data}))``. Any edit, deletion or
reordering breaks the chain. A checkpoint stores each ledger's offset
``{bytes, records, head}``; crash recovery truncates the ledger back to that
offset (moving the uncommitted tail to an orphan file, never deleting it) and
verifies that the surviving chain ends exactly at ``head``.
"""
from __future__ import annotations

import json
import os

from .util import canonical_json, sha256_json

GENESIS = "0" * 64


class LedgerError(RuntimeError):
    pass


def scan(path: str) -> dict:
    """Parse and verify a ledger file. Tolerates (and reports) a torn final line."""
    out = {"path": path, "records": [], "ok": True, "errors": [], "head": GENESIS,
           "count": 0, "bytes_valid": 0, "torn_tail_bytes": 0, "genesis": GENESIS}
    if not os.path.exists(path):
        return out
    with open(path, "rb") as f:
        raw = f.read()
    pos, prev = 0, None
    while pos < len(raw):
        nl = raw.find(b"\n", pos)
        if nl < 0:
            out["torn_tail_bytes"] = len(raw) - pos
            break
        line = raw[pos:nl]
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            out["ok"] = False
            out["errors"].append(f"unparseable line at byte {pos}")
            break
        body = {"seq": rec["seq"], "prev": rec["prev"], "data": rec["data"]}
        if prev is None:
            out["genesis"] = rec["prev"]
            prev = rec["prev"]
            expect_seq = rec["seq"]
        if rec["prev"] != prev or rec["seq"] != expect_seq or sha256_json(body) != rec["hash"]:
            out["ok"] = False
            out["errors"].append(f"chain broken at seq {rec.get('seq')}")
            break
        out["records"].append(rec)
        prev, expect_seq = rec["hash"], expect_seq + 1
        pos = nl + 1
        out["bytes_valid"] = pos
    out["count"] = len(out["records"])
    out["head"] = prev if prev is not None else GENESIS
    return out


class Ledger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s = scan(path)
        if not s["ok"]:
            raise LedgerError(f"{path}: {s['errors']}")
        if s["torn_tail_bytes"]:
            raise LedgerError(f"{path}: torn tail of {s['torn_tail_bytes']} bytes; "
                              "recover to a checkpoint offset first")
        self.count, self.head, self.bytes = s["count"], s["head"], s["bytes_valid"]
        self.first_seq = s["records"][0]["seq"] if s["records"] else 0
        self.f = open(path, "ab")

    def _line(self, data: dict) -> tuple:
        body = {"seq": self.first_seq + self.count, "prev": self.head, "data": data}
        h = sha256_json(body)
        return (canonical_json({**body, "hash": h}) + "\n").encode(), h

    def append(self, data: dict) -> str:
        line, h = self._line(data)
        self.f.write(line)
        self.f.flush()
        self.count += 1
        self.head = h
        self.bytes += len(line)
        return h

    def commit(self) -> None:
        self.f.flush()
        os.fsync(self.f.fileno())

    def offset(self) -> dict:
        return {"bytes": self.bytes, "records": self.count, "head": self.head}

    def write_torn(self, data: dict) -> None:
        """Fault injection: write only half of a record, as a crash mid-write would."""
        line, _ = self._line(data)
        self.f.write(line[: len(line) // 2])
        self.commit()

    def close(self) -> None:
        self.f.close()


def recover_to(path: str, offset: dict, orphan_path: str) -> dict:
    """Roll a ledger back to a checkpoint offset; preserve the tail as an orphan file."""
    size = os.path.getsize(path) if os.path.exists(path) else 0
    if size < offset["bytes"]:
        raise LedgerError(f"{path}: shorter ({size}) than checkpoint offset {offset['bytes']}")
    with open(path, "rb") as f:
        f.seek(offset["bytes"])
        tail = f.read()
    orphan_records = 0
    if tail:
        os.makedirs(os.path.dirname(orphan_path), exist_ok=True)
        with open(orphan_path, "wb") as f:
            f.write(tail)
        orphan_records = tail.count(b"\n")
    with open(path, "r+b") as f:
        f.truncate(offset["bytes"])
        f.flush()
        os.fsync(f.fileno())
    s = scan(path)
    ok = s["ok"] and s["head"] == offset["head"] and s["count"] == offset["records"] \
        and s["torn_tail_bytes"] == 0
    if not ok:
        raise LedgerError(f"{path}: post-recovery chain does not end at checkpoint head")
    return {"ledger": os.path.basename(path), "truncated_bytes": len(tail),
            "orphaned_complete_records": orphan_records,
            "torn_bytes": len(tail) - (tail.rfind(b"\n") + 1) if tail else 0,
            "orphan_file": orphan_path if tail else None, "head_verified": offset["head"]}


def parse_orphans(orphan_path: str) -> list:
    """Complete records from an orphan tail (torn last line ignored)."""
    out = []
    if not orphan_path or not os.path.exists(orphan_path):
        return out
    with open(orphan_path, "rb") as f:
        for line in f.read().split(b"\n"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out
