"""Hashing, canonical JSON, atomic writes and the run/event log."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import stat
import sys


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_json(obj) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str, start: int = 0, end: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        remaining = None if end is None else end - start
        while True:
            n = 1 << 20 if remaining is None else min(1 << 20, remaining)
            if n == 0:
                break
            chunk = f.read(n)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str, obj, indent: int | None = 2) -> None:
    txt = json.dumps(obj, indent=indent, sort_keys=True) + "\n"
    atomic_write_bytes(path, txt.encode("utf-8"))


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> list:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def make_readonly(path: str) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def is_readonly(path: str) -> bool:
    return not (os.stat(path).st_mode & stat.S_IWRITE)


def rmtree_force(path: str) -> None:
    """shutil.rmtree that also removes read-only (immutable) shard files."""
    import shutil

    def onerr(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if os.path.exists(path):
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=onerr)
        else:  # pragma: no cover
            shutil.rmtree(path, onerror=onerr)


def rel(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


class RunLog:
    """Human log (run.log) + machine log (events.jsonl), shared by all processes.

    Both files are opened in append mode per write, so the orchestrator and the
    training subprocesses (including one that dies mid-step) interleave safely.
    """

    def __init__(self, art_dir: str, component: str, echo: bool = True):
        self.art_dir = art_dir
        self.component = component
        self.echo = echo
        self.log_path = os.path.join(art_dir, "run.log")
        self.events_path = os.path.join(art_dir, "ledgers", "events.jsonl")
        os.makedirs(os.path.dirname(self.events_path), exist_ok=True)

    def _write(self, level: str, event: str, msg: str, fields: dict) -> None:
        ts = _dt.datetime.now().isoformat(timespec="milliseconds")
        line = f"{ts} {self.component:<8} [{level}] {event}"
        if msg:
            line += f" | {msg}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        rec = {"ts": ts, "level": level, "component": self.component,
               "event": event, "msg": msg, **fields}
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(canonical_json(rec) + "\n")
        if self.echo:
            print(line, flush=True)

    def info(self, event: str, msg: str = "", **fields):
        self._write("INFO", event, msg, fields)

    def section(self, title: str):
        self._write("STAGE", title, "", {})

    def check(self, ok: bool, event: str, msg: str = "", **fields) -> bool:
        self._write("PASS" if ok else "FAIL", event, msg, fields)
        return ok

    def warn(self, event: str, msg: str = "", **fields):
        self._write("WARN", event, msg, fields)
