"""Fast unit tests of the data-system invariants (no training run needed)."""
import json
import os
import shutil

import numpy as np
import pytest

from v5data import build
from v5data import config as C
from v5data.firewall import FirewallViolation, assert_batch_trainable
from v5data.ledger import Ledger, LedgerError, recover_to, scan
from v5data.mixture import PlanError, compile_plan, verify_plan
from v5data.opus import ACCEPT, DEFER, OVERRIDE, REJECT, OpusGate
from v5data.packing import (BestFitPacker, ConcatSplitPacker, attention_mask, batch_fingerprint,
                            check_invariants, materialize)
from v5data.pipeline import DataPipeline, clone_pipeline, held_out_sequences
from v5data.shards import ShardExistsError, ShardIntegrityError, ShardReader, write_shard
from v5data.tokenizer import ASSISTANT, EOS, USER, FrozenTokenizer, TokenizerIntegrityError


# ------------------------------------------------------------- tokenizer
def test_tokenizer_frozen_hash_and_roundtrip(built):
    P = build.paths(built["art"])
    tok = FrozenTokenizer.load_frozen(P["tokenizer"], P["tokenizer_lock"])
    assert tok.sha256 == built["lock"]["tokenizer_sha256"]
    for d in built["corpus"][("web", "train")][:20]:
        assert tok.decode(tok.encode(d["text"])) == d["text"]


def test_tokenizer_tamper_rejected(built, tmp_path):
    P = build.paths(built["art"])
    d = json.load(open(P["tokenizer"]))
    d["merges"] = d["merges"][:-1]
    bad = tmp_path / "tok.json"
    bad.write_text(json.dumps(d))
    with pytest.raises(TokenizerIntegrityError):
        FrozenTokenizer.load_frozen(str(bad), P["tokenizer_lock"])


def test_every_manifest_pins_tokenizer_and_tokens_reproduce(built):
    env, tok = built["env"], built["tok"]
    raw = build.raw_doc_index(built["art"])
    for sid, m in env.catalog.manifests.items():
        assert m["tokenizer_sha256"] == built["lock"]["tokenizer_sha256"]
        for row in m["docs"][:10]:
            ids, ls = tok.encode_document(raw[row["doc_id"]])
            assert list(env.catalog.doc_tokens(sid, row["doc_idx"])) == ids
            assert ls == row["loss_start"]


# ---------------------------------------------------------------- shards
def test_shards_are_write_once(built):
    env = built["env"]
    m = next(iter(env.catalog.manifests.values()))
    P = build.paths(built["art"])
    with pytest.raises(ShardExistsError):
        write_shard(P["shards"], P["shard_manifests"], m["shard_id"], m["lane"], m["split"], [],
                    built["tok"], built["lock"]["tokenizer_sha256"])


def test_shard_bit_flip_detected(built, tmp_path):
    m = next(iter(built["env"].catalog.manifests.values()))
    os.makedirs(tmp_path / "shards")
    for k in ("tokens", "index"):
        shutil.copy(os.path.join(built["art"], m["files"][k]["path"]), tmp_path / m["files"][k]["path"])
    p = tmp_path / m["files"]["tokens"]["path"]
    os.chmod(p, 0o666)
    b = bytearray(p.read_bytes()); b[5] ^= 1; p.write_bytes(bytes(b))
    with pytest.raises(ShardIntegrityError):
        ShardReader(str(tmp_path), m, verify=True)


def test_manifest_edit_detected(built):
    m = dict(next(iter(built["env"].catalog.manifests.values())))
    m["split"] = "train" if m["split"] != "train" else "eval"
    with pytest.raises(ShardIntegrityError):
        ShardReader(built["art"], m, verify=True)


# --------------------------------------------------------------- packing
class _FakeCatalog:
    """Tiny in-memory catalog with hand-written documents for golden mask tests."""

    def __init__(self, docs):
        self.docs = docs  # key -> (tokens, loss_start)

    def doc_tokens(self, s, i):
        return np.asarray(self.docs[(s, i)][0], dtype=np.uint16)

    def doc_meta(self, s, i):
        t, ls = self.docs[(s, i)]
        return {"length": len(t), "loss_start": ls, "doc_id": f"{s}-{i}"}

    def split_of(self, s):
        return "train"


def _seq(policy, spans, L):
    used = sum(b - a for _, _, a, b in spans)
    return {"lane": "x", "policy": policy, "pad": L - used,
            "segments": [{"shard": s, "doc_idx": i, "start": a, "end": b, "doc_id": ""}
                         for s, i, a, b in spans]}


def test_golden_masks_concat_split():
    cat = _FakeCatalog({("s", 0): ([10, 11, 12, EOS], None), ("s", 1): ([20, 21, 22, 23, 24, EOS], None)})
    L = 8
    pk = ConcatSplitPacker(L)
    pk.add("s", 0, 4); pk.add("s", 1, 6)
    assert pk.ready_count() == 1
    e = pk.emit()
    assert e["spans"] == [["s", 0, 0, 4], ["s", 1, 0, 4]] and e["pad"] == 0
    assert pk.buffer == [["s", 1, 4, 6]]           # rest of doc 1 carried over
    seqs = [_seq("concat_split_docmask", e["spans"], L)]
    a = materialize(seqs, cat, L)
    assert a["tokens"][0].tolist() == [10, 11, 12, EOS, 20, 21, 22, 23]
    assert a["position_ids"][0].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert a["segment_ids"][0].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert a["loss_mask"][0].tolist() == [1, 1, 1, 0, 1, 1, 1, 0]
    assert a["labels"][0][:3].tolist() == [11, 12, EOS]
    m = attention_mask(a["segment_ids"])[0]
    assert m[4, :4].sum() == 0 and m[4, 4] and not m[4, 5]  # doc 2 cannot see doc 1 or the future
    assert check_invariants(seqs, a, cat) == []


def test_golden_masks_response_only_and_padding():
    # <user> 5 6 <assistant> 7 8 <eos> ; loss_start = 4 (first response token)
    cat = _FakeCatalog({("s", 0): ([USER, 5, 6, ASSISTANT, 7, 8, EOS], 4)})
    L = 10
    seqs = [_seq("bestfit_nosplit_response_only", [["s", 0, 0, 7]], L)]
    a = materialize(seqs, cat, L)
    # position j predicts token j+1; loss only where the target is a response token (index >= 4)
    assert a["loss_mask"][0].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert a["segment_ids"][0].tolist()[7:] == [0, 0, 0]
    assert check_invariants(seqs, a, cat) == []
    bad = {k: v.copy() for k, v in a.items()}
    bad["loss_mask"][0, 1] = 1                         # loss on a prompt token
    assert any("prompt" in e for e in check_invariants(seqs, bad, cat))


def test_bestfit_never_splits_and_truncates_long_docs():
    pk = BestFitPacker(16, max_open_bins=2, min_residual=2)
    for i, n in enumerate([10, 9, 7, 30, 5, 3]):
        pk.add("s", i, n)
    spans = [sp for b in pk.closed + pk.open for sp in b["spans"]]
    assert all(sp[2] == 0 for sp in spans)
    assert ["s", 3, 0, 16] in spans                   # truncated to seq_len
    assert all(b["used"] <= 16 for b in pk.closed + pk.open)


def test_attention_isolation_in_model():
    import torch

    from v5data.model import TinyGPT
    torch.manual_seed(0)
    m = TinyGPT(300, 16, 32, 2, 4).eval()
    seg = np.array([[1] * 6 + [2] * 6 + [0] * 4])
    tok = torch.randint(0, 256, (1, 16))
    pos = torch.tensor([list(range(6)) + list(range(6)) + [0] * 4])
    mask = torch.from_numpy(attention_mask(seg))
    with torch.no_grad():
        base = m(tok, pos, mask)
        t2 = tok.clone(); t2[0, 2] = (t2[0, 2] + 1) % 256
        out = m(t2, pos, mask)
    d = (out - base).abs().amax(-1)[0]
    assert d[6:12].max() == 0          # document 2 unaffected by an edit in document 1
    assert d[:2].max() == 0            # causal: earlier tokens unaffected
    assert d[2:6].max() > 0


def test_real_batches_pass_invariants(built):
    env = built["env"]
    p = env.pipeline()
    for s in range(1, 25):
        b = p.next_batch(s)
        assert check_invariants(b["sequences"], b["arrays"], env.catalog) == []


# --------------------------------------------------------------- mixture
def test_plan_floors_and_shares(built):
    plan = built["env"].plan
    assert verify_plan(plan) == []
    for s in plan["steps"]:
        assert sum(s["seqs"].values()) == C.TRAIN["batch_size"]
        assert s["seqs"]["code"] >= 2 and s["seqs"]["instruct"] >= 1
    for name, st in plan["summary"].items():
        for lane, share in st["planned_seq_share"].items():
            assert abs(share - st["target_weights"][lane]) <= 0.05


def test_compiler_raises_floor_when_weight_too_low():
    stages = [{"name": "a", "start": 1, "end": 4, "opus": {"accept": 0.5, "reject": 0.4},
               "weights": {"web": 0.95, "code": 0.05, "instruct": 0.0}}]
    plan = compile_plan(stages, C.LANES, C.LANE_ORDER, 8, 4, "x")
    assert all(s["seqs"]["code"] == 2 and s["seqs"]["instruct"] == 1 for s in plan["steps"])
    assert all(s["floor_adjustments"] for s in plan["steps"])


def test_compiler_rejects_gaps():
    stages = [{"name": "a", "start": 1, "end": 3, "opus": {}, "weights": {"web": 1.0}}]
    with pytest.raises(PlanError):
        compile_plan(stages, C.LANES, C.LANE_ORDER, 8, 5, "x")


def test_plan_tamper_detected(built):
    plan = json.loads(json.dumps(built["env"].plan))
    plan["steps"][3]["seqs"]["web"] += 1
    assert verify_plan(plan)


# ------------------------------------------------------------------ OPUS
def test_opus_decision_types():
    g = OpusGate(C.OPUS, {"s/9": {"reason": "eval_exact_match"}})
    th = {"accept": 0.6, "reject": 0.4}
    meta = lambda q, h="h": {"quality": q, "content_sha256": h}
    assert g.evaluate("s/1", meta(0.7), 0, th)[0] == ACCEPT
    assert g.evaluate("s/1", meta(0.3), 0, th)[:2] == (REJECT, "low_quality")
    assert g.evaluate("s/1", meta(0.5), 0, th)[0] == DEFER
    assert g.evaluate("s/1", meta(0.5), 2, th)[0] == DEFER          # 0.5 + 2*0.04 < 0.6
    assert g.evaluate("s/1", meta(0.5), 3, th)[0] == ACCEPT         # aging lets it through
    assert g.evaluate("s/1", meta(0.41), 3, th)[:2] == (REJECT, "defer_expired")
    assert g.evaluate("s/9", meta(0.99), 0, th)[1].startswith("eval_contamination")
    g.admit(meta(0.7, "dup"))
    assert g.evaluate("s/2", meta(0.9, "dup"), 0, th)[:2] == (REJECT, "duplicate_content")


def test_floor_override_only_promotes_deferrals_in_protected_lanes(built):
    env = built["env"]
    p = env.pipeline()
    decs = []
    for s in range(1, 81):
        decs += p.next_plan(s)["decisions"]
    ov = [d for d in decs if d["decision"] == OVERRIDE]
    assert ov, "demo configuration should exercise the protected-floor override"
    assert all(C.LANES[d["lane"]]["protected"] for d in ov)
    assert all(d["overrides"] == DEFER for d in ov)
    kinds = {d["decision"] for d in decs}
    assert kinds == {ACCEPT, DEFER, REJECT, OVERRIDE}


# -------------------------------------------------------------- firewall
def test_firewall_blocks_held_out_shards(built):
    env = built["env"]
    p = DataPipeline(env.catalog, env.plan, C.LANES, C.OPUS, C.PACKING, C.TRAIN["seq_len"], C.SEED,
                     env.contamination, auto_register=False)
    for split in ("val", "eval"):
        with pytest.raises(FirewallViolation):
            p.register_source("web", env.catalog.shards_where(lane="web", split=split)[0])
    with pytest.raises(FirewallViolation):
        assert_batch_trainable(env.catalog, held_out_sequences(
            env.catalog, "eval", C.LANES, C.LANE_ORDER, C.TRAIN["seq_len"], C.PACKING))


def test_contamination_flags_planted_leaks_and_nothing_is_consumed(built):
    env = built["env"]
    flagged = env.contamination["flagged"]
    reasons = {v["reason"] for v in flagged.values()}
    assert {"eval_exact_match", "eval_ngram_overlap"} <= reasons
    p = env.pipeline()
    for s in range(1, 81):
        b = p.next_plan(s)
        for sq in b["sequences"]:
            for g in sq["segments"]:
                assert f"{g['shard']}/{g['doc_idx']}" not in flagged
                assert env.catalog.split_of(g["shard"]) == "train"


# ------------------------------------------------------ determinism/state
def test_pipeline_state_roundtrip_resumes_identical_stream(built):
    env = built["env"]
    a = env.pipeline()
    for s in range(1, 31):
        a.next_batch(s)
    snap = a.get_state()
    tail_a = [a.next_batch(s)["batch_hash"] for s in range(31, 51)]
    b = env.pipeline()
    b.set_state(snap)
    tail_b = [b.next_batch(s)["batch_hash"] for s in range(31, 51)]
    assert tail_a == tail_b


def test_pipeline_refuses_skipping_or_repeating_steps(built):
    p = built["env"].pipeline()
    p.next_plan(1)
    with pytest.raises(RuntimeError):
        p.next_plan(1)
    with pytest.raises(RuntimeError):
        p.next_plan(3)


def test_clone_lookahead_does_not_advance_original(built):
    p = built["env"].pipeline()
    p.next_batch(1)
    h = p.state_sha256()
    nxt = clone_pipeline(p).next_batch(2)["batch_hash"]
    assert p.state_sha256() == h
    assert p.next_batch(2)["batch_hash"] == nxt


def test_batch_fingerprint_sensitive_to_masks(built):
    b = built["env"].pipeline().next_batch(1)
    a2 = {k: v.copy() for k, v in b["arrays"].items()}
    a2["loss_mask"][0, 0] ^= 1
    assert batch_fingerprint(b["sequences"], a2) != b["batch_hash"]


# ---------------------------------------------------------------- ledger
def test_ledger_chain_detects_edit_and_recovers_to_offset(tmp_path):
    p = str(tmp_path / "l.jsonl")
    L = Ledger(p)
    for i in range(5):
        L.append({"step": i})
    L.commit()
    off = L.offset()
    L.append({"step": 5}); L.write_torn({"step": 6}); L.close()
    assert scan(p)["torn_tail_bytes"] > 0
    with pytest.raises(LedgerError):
        Ledger(p)                                      # refuses to append after a torn tail
    r = recover_to(p, off, str(tmp_path / "orphan.jsonl"))
    assert r["orphaned_complete_records"] == 1 and r["torn_bytes"] > 0
    s = scan(p)
    assert s["ok"] and s["count"] == 5 and s["head"] == off["head"]
    raw = open(p, "rb").read().replace(b'"step":2', b'"step":9')
    open(p, "wb").write(raw)
    assert not scan(p)["ok"]
