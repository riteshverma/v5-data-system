"""One command, full demonstration:

    python run_demo.py            (regenerates ./submission_artifacts from scratch)

documents -> tokenized shards -> manifests -> mixture schedule -> packing ->
batches -> training -> consumption ledger -> learning ledger -> checkpoint ->
crash -> resume -> replay -> fork -> audit -> evidence

Training, crash, resume and fork run as separate OS processes; the crash is a
real ``os._exit(137)`` in the middle of a step.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from v5data import config as C  # noqa: E402
from v5data import demo_phases as D  # noqa: E402
from v5data.util import RunLog, rmtree_force  # noqa: E402


def spawn(art: str, *args) -> int:
    cmd = [sys.executable, "-m", "v5data.trainer", "--art", art, *map(str, args)]
    return subprocess.run(cmd, cwd=HERE).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=os.path.join(HERE, "submission_artifacts"))
    art = os.path.abspath(ap.parse_args().art)
    t0 = time.perf_counter()
    rmtree_force(art)
    os.makedirs(art)
    log = RunLog(art, "demo")
    T = C.TRAIN["total_steps"]

    log.section("PHASE 1  documents -> frozen tokenizer")
    corpus, tok, lock = D.documents_and_tokenizer(art, log)
    log.section("PHASE 2  immutable shards + manifests")
    D.shards_and_manifests(art, log, corpus, tok, lock)
    log.section("PHASE 3  evaluation firewall + mixture compilation")
    D.firewall_and_mixture(art, log, lock)
    log.section("PHASE 4  reference (uninterrupted, data-only) stream")
    ref_secs = D.reference_stream(art, log)

    log.section("PHASE 5  training attempt 1 (deliberate crash)")
    crash = C.TRAIN["crash_at_step"]
    rc = spawn(art, "--mode", "train", "--end-step", T, "--crash-at", crash)
    D.after_crash(art, log, rc, crash)

    log.section("PHASE 6  resume from last checkpoint")
    rc = spawn(art, "--mode", "resume", "--end-step", T)
    D.after_resume(art, log, rc)

    log.section("PHASE 7  replay of a historical interval")
    D.replay(art, log)

    log.section("PHASE 8  fork from an earlier checkpoint")
    F = C.FORK
    before = D.parent_ledger_hashes(art)
    rc = spawn(art, "--mode", "fork", "--branch", F["name"], "--fork-from", F["from_checkpoint_step"],
               "--diverge-at", F["diverge_at_step"], "--end-step", F["end_step"])
    D.after_fork(art, log, rc, before)

    log.section("PHASE 9  performance")
    D.performance(art, log, ref_secs)

    log.section("PHASE 10 independent audit")
    ok = D.audit(art, log, t0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
