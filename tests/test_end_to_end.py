"""Run the whole demonstration once, then prove the auditor is not a rubber stamp:
corrupt copies of the artifacts in targeted ways and check the matching
requirement flips to FAIL."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from v5data.ledger import scan
from v5data.util import canonical_json, rmtree_force, sha256_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    art = str(tmp_path_factory.mktemp("demo") / "submission_artifacts")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run_demo.py"), "--art", art],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return art


def _copy(art, tmp_path):
    dst = str(tmp_path / "copy")
    shutil.copytree(art, dst)
    return dst


def _results(art):
    """Audit in a separate process (keeps this test process free of torch)."""
    r = subprocess.run([sys.executable, "-m", "v5data.audit", "--art", art], cwd=ROOT,
                       capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def _rewrite_ledger(path, mutate):
    """Mutate records and re-chain every hash: a 'clever' forgery that keeps the chain valid."""
    recs = [r["data"] for r in scan(path)["records"]]
    recs = mutate(recs)
    prev = "0" * 64
    with open(path, "wb") as f:
        for i, d in enumerate(recs):
            body = {"seq": i, "prev": prev, "data": d}
            h = sha256_json(body)
            f.write((canonical_json({**body, "hash": h}) + "\n").encode())
            prev = h


def test_demo_produces_required_structure_and_passes(demo):
    for p in ("run.log", "evidence.json", "evidence.md", "manifests", "ledgers", "checkpoints",
              "performance.json"):
        assert os.path.exists(os.path.join(demo, p)), p
    ev = json.load(open(os.path.join(demo, "evidence.json")))
    assert ev["overall"] == "PASS"
    log = open(os.path.join(demo, "run.log")).read()
    for tag in ("[PASS] tokenizer_hash_verified", "[PASS] eval_shard_blocked",
                "[PASS] checkpoint_saved", "[PASS] resume_next_batch_matched",
                "[PASS] replay_hash_matched", "[PASS] audit_completed"):
        assert tag in log
    assert "[FAIL]" not in log


def test_demo_is_reproducible(demo, tmp_path):
    """A second full run yields byte-identical shards/manifests/plan and the same batch stream."""
    art2 = str(tmp_path / "again")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run_demo.py"), "--art", art2],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0
    for rel in ("manifests/catalog.json", "manifests/mixture_plan.json",
                "tokenizer/tokenizer.lock.json", "reports/reference_stream.json"):
        a = json.load(open(os.path.join(demo, rel)))
        b = json.load(open(os.path.join(art2, rel)))
        if rel.endswith("reference_stream.json"):
            a = [(x["batch_id"], x["batch_hash"]) for x in a["batches"]]
            b = [(x["batch_id"], x["batch_hash"]) for x in b["batches"]]
        assert a == b, rel
    ids = lambda art: [r["data"]["batch_hash"] for r in
                       scan(os.path.join(art, "ledgers", "consumption.jsonl"))["records"]]
    assert ids(demo) == ids(art2)
    rmtree_force(art2)


def test_audit_catches_skipped_batch(demo, tmp_path):
    art = _copy(demo, tmp_path)
    _rewrite_ledger(os.path.join(art, "ledgers", "consumption.jsonl"),
                    lambda rs: [r for r in rs if r["step"] != 50])
    res = _results(art)
    assert not res["ledgers"] and not res["crash_recovery"]


def test_audit_catches_repeated_batch(demo, tmp_path):
    art = _copy(demo, tmp_path)

    def dup(rs):
        rs[50] = dict(rs[49], step=51)
        return rs
    _rewrite_ledger(os.path.join(art, "ledgers", "consumption.jsonl"), dup)
    assert not _results(art)["crash_recovery"]


def test_audit_catches_eval_tokens_in_training_batch(demo, tmp_path):
    art = _copy(demo, tmp_path)

    def leak(rs):
        rs[10]["sequences"][0]["spans"][0][0] = "web-eval-000"
        return rs
    _rewrite_ledger(os.path.join(art, "ledgers", "consumption.jsonl"), leak)
    res = _results(art)
    assert not res["eval_firewall"] and not res["packing"]


def test_audit_catches_forged_opus_decision(demo, tmp_path):
    art = _copy(demo, tmp_path)

    def forge(rs):
        i = next(i for i, r in enumerate(rs) if r["decision"] == "REJECT")
        rs[i]["decision"] = "ACCEPT"
        return rs
    _rewrite_ledger(os.path.join(art, "ledgers", "opus_decisions.jsonl"), forge)
    assert not _results(art)["opus"]


def test_audit_catches_unlinked_loss(demo, tmp_path):
    art = _copy(demo, tmp_path)

    def bump(rs):
        rs[5]["segments"][0][8] += 1.0
        return rs
    _rewrite_ledger(os.path.join(art, "ledgers", "learning.jsonl"), bump)
    assert not _results(art)["learning_trace"]


def test_audit_catches_inflated_performance(demo, tmp_path):
    art = _copy(demo, tmp_path)
    p = os.path.join(art, "performance.json")
    perf = json.load(open(p))
    perf["packing"]["totals"]["utilization"] = 0.999
    json.dump(perf, open(p, "w"))
    assert not _results(art)["throughput"]


def test_audit_catches_simple_ledger_edit(demo, tmp_path):
    art = _copy(demo, tmp_path)
    p = os.path.join(art, "ledgers", "learning.jsonl")
    raw = open(p, "rb").read()
    open(p, "wb").write(raw.replace(b'"step":12,', b'"step":13,', 1))
    assert not _results(art)["ledgers"]
