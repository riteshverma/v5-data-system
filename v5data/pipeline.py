"""The deterministic data pipeline: lane streams -> OPUS -> packers -> batches.

The whole pipeline is a state machine whose state is a small JSON document
(lane cursors, deferral queues, packer buffers, OPUS dedupe set). Given
(catalog, plan, state) the next batch is fully determined, and it does not
depend on model weights. That property is what gives:

* **resume**  - restore the state saved in a checkpoint, continue at step k+1
* **replay**  - restore an older checkpoint's state, regenerate an interval
* **fork**    - restore an older state, continue under a different plan
* **lookahead** - a checkpoint can record the exact next batch it expects
"""
from __future__ import annotations

import copy

import numpy as np

from . import firewall
from .opus import ACCEPT, DEFER, OVERRIDE, OpusGate
from .packing import batch_fingerprint, batch_stats, make_packer, materialize
from .util import sha256_json


class DataExhausted(RuntimeError):
    pass


def doc_key(shard: str, idx: int) -> str:
    return f"{shard}/{idx}"


class DataPipeline:
    def __init__(self, catalog, plan: dict, lanes: dict, opus_cfg: dict, packing_cfg: dict,
                 seq_len: int, seed: int, contamination: dict, auto_register: bool = True):
        self._args = dict(catalog=catalog, plan=plan, lanes=lanes, opus_cfg=opus_cfg,
                          packing_cfg=packing_cfg, seq_len=seq_len, seed=seed,
                          contamination=contamination)
        self.catalog, self.plan, self.lanes = catalog, plan, lanes
        self.seq_len, self.seed = seq_len, seed
        self.lane_order = plan["lane_order"]
        self.sources = {l: [] for l in self.lane_order}
        self.opus = OpusGate(opus_cfg, contamination["flagged"])
        self.packers = {l: make_packer(lanes[l]["policy"], seq_len, packing_cfg)
                        for l in self.lane_order}
        self.cursor = {l: 0 for l in self.lane_order}
        self.deferred = {l: [] for l in self.lane_order}
        self.last_step = 0
        self.orders = None
        if auto_register:
            for l in self.lane_order:
                for sid in catalog.shards_where(lane=l, split="train"):
                    self.register_source(l, sid)
            self.seal()

    # ------------------------------------------------------------ sources
    def register_source(self, lane: str, shard_id: str) -> None:
        """Firewall choke point: only train-split shards can become sources."""
        if self.orders is not None:
            raise RuntimeError("pipeline sealed; sources are immutable after sealing")
        m = self.catalog.manifests[shard_id]
        firewall.assert_trainable(m)
        if m["lane"] != lane:
            raise ValueError(f"shard {shard_id} belongs to lane {m['lane']}, not {lane}")
        self.sources[lane].append(shard_id)

    def seal(self) -> None:
        """Freeze sources and derive each lane's deterministic document order."""
        self.orders = {}
        for li, l in enumerate(self.lane_order):
            keys = [doc_key(sid, i) for sid in sorted(self.sources[l])
                    for i in range(self.catalog.manifests[sid]["num_docs"])]
            perm = np.random.default_rng([self.seed, li]).permutation(len(keys))
            self.orders[l] = [keys[p] for p in perm]
        self.order_sha256 = sha256_json(self.orders)

    # --------------------------------------------------------------- state
    def get_state(self) -> dict:
        return copy.deepcopy({
            "last_step": self.last_step, "cursor": self.cursor, "deferred": self.deferred,
            "packers": {l: p.get_state() for l, p in self.packers.items()},
            "opus": self.opus.get_state(), "order_sha256": self.order_sha256,
            "plan_sha256": self.plan["plan_sha256"]})

    def set_state(self, st: dict) -> None:
        if st["order_sha256"] != self.order_sha256:
            raise RuntimeError("lane order mismatch: different sources or seed")
        st = copy.deepcopy(st)
        self.last_step = st["last_step"]
        self.cursor = st["cursor"]
        self.deferred = st["deferred"]
        for l, p in self.packers.items():
            p.set_state(st["packers"][l])
        self.opus.set_state(st["opus"])

    def state_sha256(self) -> str:
        return sha256_json(self.get_state())

    # ---------------------------------------------------------- candidates
    def _next_candidate(self, lane: str, step: int):
        dq = self.deferred[lane]
        for i, item in enumerate(dq):
            if item["ready"] <= step:
                return dq.pop(i), "deferred"
        if self.cursor[lane] >= len(self.orders[lane]):
            return None, None
        key = self.orders[lane][self.cursor[lane]]
        self.cursor[lane] += 1
        return {"k": key, "n": 0, "ready": step}, "fresh"

    def _meta(self, key: str) -> dict:
        sid, idx = key.rsplit("/", 1)
        return self.catalog.doc_meta(sid, int(idx))

    def _record(self, out: list, step, lane, stage, item, source, decision, reason, score, th,
                **extra) -> None:
        meta = self._meta(item["k"])
        out.append({"step": step, "lane": lane, "stage": stage, "doc_key": item["k"],
                    "doc_id": meta["doc_id"], "content_sha256": meta["content_sha256"],
                    "quality": meta["quality"], "tokens": meta["length"],
                    "defer_count": item["n"], "score": score, "thresholds": th,
                    "decision": decision, "reason": reason, "source": source, **extra})

    def _admit(self, lane: str, key: str) -> None:
        meta = self._meta(key)
        self.opus.admit(meta)
        sid, idx = key.rsplit("/", 1)
        self.packers[lane].add(sid, int(idx), meta["length"])

    # ---------------------------------------------------------------- fill
    def _fill_lane(self, lane, need, floor_n, th, step, stage, budget_tokens, decisions):
        packer, scanned, window = self.packers[lane], 0, []
        while packer.ready_count() < need:
            if budget_tokens is not None and scanned >= budget_tokens:
                break
            item, src = self._next_candidate(lane, step)
            if item is None:
                raise DataExhausted(f"lane {lane} exhausted at step {step}")
            meta = self._meta(item["k"])
            scanned += meta["length"]
            dec, reason, score = self.opus.evaluate(item["k"], meta, item["n"], th)
            self._record(decisions, step, lane, stage, item, src, dec, reason, score, th)
            if dec == ACCEPT:
                self._admit(lane, item["k"])
            elif dec == DEFER:
                q = {"k": item["k"], "n": item["n"] + 1, "ready": step + 1}
                self.deferred[lane].append(q)
                window.append((score, len(window), q))

        overrides = 0
        if self.lanes[lane]["protected"] and packer.ready_count() < floor_n:
            # 1) promote this step's deferred candidates, best score first
            for score, _, q in sorted(window, key=lambda t: (-t[0], t[1])):
                if packer.ready_count() >= floor_n:
                    break
                meta = self._meta(q["k"])
                if meta["content_sha256"] in self.opus.accepted_hashes:
                    continue
                self.deferred[lane].remove(q)
                self._record(decisions, step, lane, stage, {**q, "n": q["n"] - 1}, "override",
                             OVERRIDE, f"protected_floor:{lane} ready={packer.ready_count()}"
                             f"<floor={floor_n}", score, th, overrides=DEFER)
                self._admit(lane, q["k"])
                overrides += 1
            # 2) keep scanning; DEFER-class candidates are admitted as overrides
            while packer.ready_count() < floor_n:
                item, src = self._next_candidate(lane, step)
                if item is None:
                    raise DataExhausted(f"protected lane {lane} cannot meet floor at step {step}")
                meta = self._meta(item["k"])
                dec, reason, score = self.opus.evaluate(item["k"], meta, item["n"], th)
                if dec == DEFER:
                    self._record(decisions, step, lane, stage, item, src, OVERRIDE,
                                 f"protected_floor:{lane} ready={packer.ready_count()}"
                                 f"<floor={floor_n}", score, th, overrides=DEFER)
                    self._admit(lane, item["k"])
                    overrides += 1
                else:
                    self._record(decisions, step, lane, stage, item, src, dec, reason, score, th)
                    if dec == ACCEPT:
                        self._admit(lane, item["k"])
        k = min(need, packer.ready_count())
        seqs = []
        for _ in range(k):
            e = packer.emit()
            seqs.append({"lane": lane, "policy": packer.policy, "pad": e["pad"],
                         "segments": [{"shard": s, "doc_idx": i, "start": a, "end": b,
                                       "doc_id": self.catalog.doc_meta(s, i)["doc_id"]}
                                      for s, i, a, b in e["spans"]]})
        return seqs, need - k, scanned, overrides

    def next_plan(self, step: int) -> dict:
        """Decide the next batch (spans + OPUS decisions). Advances state."""
        if step != self.last_step + 1:
            raise RuntimeError(f"pipeline expects step {self.last_step + 1}, asked for {step}")
        ps = self.plan["steps"][step - 1]
        assert ps["step"] == step
        th, stage = ps["opus"], ps["stage"]
        decisions, sequences, lane_report = [], [], {}
        short_total = 0
        backfill = [l for l in self.lane_order if self.lanes[l].get("backfill")]
        for lane in self.lane_order:
            need = ps["seqs"][lane] + (short_total if lane in backfill else 0)
            f = self.lanes[lane]
            budget = None if f.get("budget_factor") is None else \
                int(f["budget_factor"] * ps["seqs"][lane] * self.seq_len)
            floor_n = self.plan["floors_seqs"][lane]
            seqs, short, scanned, ov = self._fill_lane(lane, need, floor_n, th, step, stage,
                                                       budget, decisions)
            if lane in backfill and short:
                raise DataExhausted(f"backfill lane {lane} short at step {step}")
            if lane not in backfill:
                short_total += short
            sequences += seqs
            lane_report[lane] = {"planned": ps["seqs"][lane], "emitted": len(seqs),
                                 "shortfall": short, "scanned_tokens": scanned,
                                 "floor": floor_n, "overrides": ov,
                                 "backfill_received": (need - ps["seqs"][lane])
                                 if lane in backfill else 0}
        self.last_step = step
        return {"step": step, "stage": stage, "sequences": sequences,
                "decisions": decisions, "lanes": lane_report}

    def next_batch(self, step: int) -> dict:
        bp = self.next_plan(step)
        firewall.assert_batch_trainable(self.catalog, bp["sequences"])
        arrays = materialize(bp["sequences"], self.catalog, self.seq_len)
        h = batch_fingerprint(bp["sequences"], arrays)
        bp.update({"arrays": arrays, "batch_hash": h, "batch_id": f"B{step:05d}-{h[:12]}",
                   "stats": batch_stats(arrays)})
        return bp


def clone_pipeline(p: DataPipeline) -> DataPipeline:
    """Independent copy at the same state (used for checkpoint lookahead)."""
    q = DataPipeline(**p._args)
    q.set_state(p.get_state())
    return q


def held_out_sequences(catalog, split: str, lanes: dict, lane_order: list, seq_len: int,
                       packing_cfg: dict) -> list:
    """Deterministic packing of a held-out split (val/eval). No OPUS, never trained on."""
    if split not in firewall.HELD_OUT_SPLITS:
        raise firewall.FirewallViolation(f"{split} is not a held-out split")
    seqs = []
    for lane in lane_order:
        pk = make_packer(lanes[lane]["policy"], seq_len, packing_cfg)
        for sid in catalog.shards_where(lane=lane, split=split):
            for i in range(catalog.manifests[sid]["num_docs"]):
                pk.add(sid, i, catalog.doc_meta(sid, i)["length"])
        # flush everything, including partially filled sequences
        if hasattr(pk, "open"):
            while pk.open:
                pk._close(0)
        out = []
        while pk.ready_count():
            out.append(pk.emit())
        if not hasattr(pk, "open") and pk.buffer:
            out.append({"spans": [list(x) for x in pk.buffer],
                        "pad": seq_len - pk.buffered_tokens()})
        for e in out:
            seqs.append({"lane": lane, "policy": pk.policy, "pad": e["pad"], "split": split,
                         "segments": [{"shard": s, "doc_idx": i, "start": a, "end": b,
                                       "doc_id": catalog.doc_meta(s, i)["doc_id"]}
                                      for s, i, a, b in e["spans"]]})
    return seqs
