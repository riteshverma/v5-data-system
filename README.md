# V5 Training Data Execution System

A small but complete training-data execution system that can prove **what a run consumed, why it
consumed it, what the model learned from it, and how the run can be reconstructed**. The corpus,
tokenizer and model are deliberately tiny (≈200k tokens, 580-token BPE, 0.5M-param GPT on CPU);
the data system around them is the point.

```
python run_demo.py        # ~35 s on a laptop CPU; regenerates ./submission_artifacts
python -m pytest tests    # 33 tests, ~90 s (includes a full demo run + fault injection)
```

Requirements: Python ≥ 3.10, `numpy`, `torch` (CPU is enough), `pytest` for the tests.

## What the demo does

`run_demo.py` runs the full path with no manual steps. Training, crash, resume and fork are
**separate OS processes**; the crash is a real `os._exit(137)` in the middle of a step.

| # | Phase | What happens |
|---|---|---|
| 1 | documents → tokenizer | Generate a deterministic synthetic corpus (web prose, code, chat) with *planted* problems: duplicates, eval documents leaked verbatim and with a one-word edit, spam, low-quality code. Train a byte-level BPE on the **train split only**, freeze it behind a SHA-256 lock, prove round-trip and that a one-merge edit is rejected. |
| 2 | shards → manifests | 33 write-once, read-only shards + self-hashing manifests + catalog. Validation re-tokenizes every raw document and must reproduce the stored tokens. Overwrite is refused and a flipped bit is detected. |
| 3 | firewall + mixture | Scan train documents against eval (exact hash, 16-gram overlap). Compile curriculum stages and floors into an 80-step plan. Attempt to register eval and val shards as training sources and to feed a packed eval batch to the optimizer; all three are blocked. |
| 4 | reference stream | Run the data pipeline alone for 80 steps: the stream any training attempt must reproduce. |
| 5 | train + crash | Train with checkpoints every 20 steps; die mid-step 47 after the consumption and OPUS records are written and while the learning record is half written. |
| 6 | resume | Roll every ledger back to the step-40 checkpoint offsets (the uncommitted tail is kept as orphans), restore model, optimizer, RNG and pipeline state, and prove step 41 is exactly the batch the checkpoint said would come next. |
| 7 | replay | Rebuild steps 21–40 three independent ways (ledger spans only / pipeline from checkpoint 20 / full retraining) and compare ids, spans, hashes, OPUS decisions and token-level losses. |
| 8 | fork | Branch from checkpoint 20 with a new mixture from step 26. Steps 21–25 are identical to the parent, 26–35 diverge, and the parent ledgers stay byte-identical. |
| 9 | performance | Packing utilization, loss-bearing fraction and useful tokens/s, all computed from ledgers. |
| 10 | audit | An independent auditor re-derives every claim from the files on disk → `evidence.json` / `evidence.md`. |

## Results of the committed run

| | |
|---|---|
| Audit | **13/13 requirements, 83 checks PASS**, 0 `[FAIL]` lines in `run.log` |
| Crash / resume | checkpoint expected `B00041-f9f434d7c3e4`, resumed run consumed `B00041-f9f434d7c3e4`; all 80 batches equal the uninterrupted reference stream; resumed losses for steps 41–46 are **bitwise equal** to the crashed attempt's |
| Replay 21–40 | ids, spans, hashes, OPUS decision stream and pipeline state all match; retrained token losses bitwise equal (max loss diff 0.0) |
| OPUS | 1519 ACCEPT, 334 DEFER, 254 REJECT, 91 ACCEPT_FLOOR_OVERRIDE; all 11 contaminated candidates hard-rejected |
| Mixture | protected floors (code ≥ 2, instruct ≥ 1 of 8 sequences) held on every step; worst stage deviation 0.05, fully explained by recorded OPUS shortfalls |
| Packing | 94.5% slot utilization vs 37.9% for one-document-per-sequence (−60% sequences); 82.1% of slots carry loss |
| Throughput | 23.5k useful loss-bearing tokens/s end-to-end on CPU; data pipeline alone ≈1.9M slot tokens/s (1.7% of step time) |
| Learning | train loss 5.53 → 1.35 (first vs last 10 steps); val loss 3.14 → 1.37 |

## Architecture

```
corpus.py ─► tokenizer.py ─► shards.py ──► firewall.py ──► mixture.py
 (raw docs)   (frozen BPE)    (immutable     (contamination   (compiled per-step
                              shards +       scan, split      plan: stages,
                              manifests)     gate)            weights, floors)
                                   │
                                   ▼
                     pipeline.py  lane streams ─► opus.py ─► packing.py ─► batch
                     (JSON state)  (seeded order)  (admission)  (3 policies)  (masks + hash)
                                   │
                                   ▼
trainer.py ──► ledger.py (consumption, opus_decisions, learning, validation, lifecycle)
         └──► checkpoint.py (model + optimizer + RNG + pipeline state + ledger offsets)
replay.py / audit.py / perf.py read only files on disk.
```

### Key design decisions

**The data stream is a pure function of (catalog, plan, pipeline state) and never depends on
model weights.** OPUS scores come from manifest quality plus deferral aging, not from live model
loss. This is what makes resume, replay, fork and checkpoint lookahead exact rather than
approximate: the batch for step *k+1* can be computed, recorded in the checkpoint, and checked
later. (A model-scored OPUS would need its scores written to the ledger and replayed from it;
this system has the ledger hooks for that, but a learned score was not needed to prove the
mechanics.)

**Pipeline state is a small JSON document**: lane cursors, deferral queues, packer carry-over
buffers and the OPUS dedupe set. It is checkpointed with the model, and its hash is written into
every consumption record, so drift is detected at the exact step it happens. This is the
Megatron "sample index" idea extended to an online, stateful admission policy. Mosaic-style
mid-epoch resume falls out of it.

**Write-ahead ledgers, hash-chained, with offsets in the checkpoint.** Per step: consumption
record, then OPUS records, then fsync, then the optimizer step, then the token-loss file and
learning record, then fsync. Every line carries `prev`/`hash`, so any edit, deletion or reorder
breaks the chain. A checkpoint stores `{bytes, records, head}` per ledger. Recovery truncates
to those offsets and verifies the surviving chain ends at `head`. The discarded tail is moved to
`ledgers/orphans/attemptN/`, never deleted, which is how the demo proves the resumed attempt
recomputed the same batches and losses the crashed one had. Lifecycle events (checkpoint saved,
crash recovered, process exited, forked) go to an append-only `lifecycle.jsonl`, in the spirit
of a Delta/Iceberg transaction log.

**Packing policies per data type**, all with document masking (segment ids → block-diagonal
causal attention, position ids restart per document, and no loss on a document's last token):

| lane | policy | why |
|---|---|---|
| web | `concat_split_docmask` | prose tolerates splitting; zero padding |
| code | `bestfit_nosplit_docmask` | never cut a function in half; best-fit over 4 open bins; over-long docs truncated (recorded in the span) |
| instruct | `bestfit_nosplit_response_only` | never split a conversation; loss only on the assistant response |

The auditor re-materializes all 80 batches from the ledger spans and re-checks every mask
property. It also perturbs one packed document on the *trained* model and verifies that the
logits of the other documents in that row do not move.

**Mixture = compiled plan + protected floors + recorded shortfall.** Integer smooth weighted
round-robin turns stage weights into exact per-step sequence counts, and floors are enforced
at compile time. At run time each lane has an OPUS scan budget. A non-protected lane that
cannot fill in budget cedes sequences to the web backfill lane, and a *protected* lane instead
promotes its best deferred candidates (`ACCEPT_FLOOR_OVERRIDE`). Hard rejects (eval
contamination, duplicates) are never overridable. The auditor reconciles every step as
`actual = planned − shortfall + backfill` and re-derives every one of the 2198 OPUS decisions
from the manifest and the plan.

**Evaluation firewall in two layers.** (1) Split gate: only `train` shards can be registered as
sources, and every batch is re-checked right before the optimizer. Val and eval data are only
scored under `no_grad` and logged as `loss_bearing: false`. The tokenizer is trained on train
text only. (2) Content gate: exact content-hash matches with eval on every lane, plus 16-gram
overlap ≥ 0.6 on prose. Calibration on this corpus: clean web documents max out at 0.47,
leaked near-copies start at 0.81. Templated code and chat share boilerplate n-grams with eval
(p99 overlap 0.76–0.92), so for those lanes only exact hashes are used. A production system
would use per-lane calibrated MinHash, e.g. NeMo Curator.

**Bit-exact CPU training** (`torch.use_deterministic_algorithms`, fixed thread count, no
dropout) turns "the learning trace is reproducible" from a claim into a checked equality.

**Evidence is computed, not asserted.** `run_demo.py` only calls library code. Every `[PASS]` is
the result of a comparison, and `audit.py` recomputes everything from disk: hashes, ledger
chains, re-materialized batches, OPUS decisions, mixture reconciliation and performance
numbers. `tests/test_end_to_end.py` forges artifacts (dropped batch, duplicated batch, eval span
injected with a re-chained ledger, flipped OPUS decision, altered segment loss, inflated
utilization, plain ledger edit) and asserts that the matching requirement flips to FAIL.

## Artifacts (`submission_artifacts/`)

| path | contents |
|---|---|
| `run.log` | every event from all four processes, in order (`[PASS] resume_next_batch_matched`, …) |
| `evidence.json`, `evidence.md` | requirement → PASS/FAIL, each named check with its detail, file pointers |
| `performance.json` | packing and throughput metrics with definitions, plus the naive baseline |
| `manifests/` | shard manifests, catalog, validation and contamination reports, compiled mixture plan, branch lineage |
| `ledgers/` | consumption, opus_decisions, learning, validation, evaluation, lifecycle, events; `token_losses/` (per-token loss arrays); `orphans/attempt1/` (the crashed tail); `branches/<fork>/` |
| `checkpoints/` | `step_000020..80/` (model, optimizer, pipeline state, `meta.json` with ledger offsets + expected next batch), `LATEST.json`, branch checkpoint |
| `reports/` | reference stream, replay, resume, fork, packing, mixture, OPUS and learning-trace reports |
| `shards/`, `tokenizer/`, `corpus/` | immutable token shards, frozen tokenizer + lock, raw documents |

Batch ids are `B<step>-<first 12 hex of batch hash>`. The batch hash covers the span layout and
the token, label, loss-mask, position-id and segment-id arrays.

## Code map

| module | responsibility |
|---|---|
| `v5data/config.py` | the single configuration (corpus, lanes, stages, OPUS, crash/replay/fork points) |
| `corpus.py`, `quality.py` | synthetic corpus with planted defects; deterministic quality score |
| `tokenizer.py` | byte-level BPE, freeze/lock/verify |
| `shards.py` | write-once shards, manifests, catalog, full validation |
| `firewall.py` | split gate, contamination scan |
| `mixture.py` | stage/weight/floor compiler and verifier |
| `opus.py` | admission decisions |
| `packing.py` | packers, materialization, attention mask, invariants, fingerprints |
| `pipeline.py` | lane streams → OPUS → packers → batches; get/set state; held-out packing |
| `ledger.py`, `checkpoint.py` | hash-chained ledgers with recovery; atomic checkpoints |
| `model.py`, `trainer.py` | tiny GPT; train / crash / resume / fork process |
| `replay.py`, `perf.py`, `audit.py` | reconstruction, metrics, independent audit + evidence |
| `demo_phases*.py`, `run_demo.py` | the one-command demonstration |

## Limitations

Single process, single device. Scaling out would shard the pipeline state per data-parallel
rank, and the ledgers would become per-rank streams merged by step. Quality scores are
heuristic. Throughput is measured on a toy model, so the headline number is the utilization
and loss-bearing fraction; tokens/s mainly shows that the data path and ledger writes (≈20% of
step time at this model size, dominated by fsync) are cheap relative to compute.
