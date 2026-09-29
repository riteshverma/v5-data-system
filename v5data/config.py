"""Single source of truth for the demonstration run.

Everything that influences *which tokens are trained on* lives here and is
hashed into the compiled mixture plan, so a checkpoint can refuse to resume
under a different data configuration.
"""
from __future__ import annotations

import copy

SEED = 5_2026

# ---------------------------------------------------------------- corpus
CORPUS = {
    "seed": SEED,
    "docs_per_shard": 128,
    # lane -> split -> number of generated documents
    "counts": {
        "web": {"train": 900, "val": 24, "eval": 40},
        "code": {"train": 800, "val": 16, "eval": 16},
        "instruct": {"train": 1500, "val": 32, "eval": 40},
    },
    "planted": {
        "web_exact_duplicates": 18,      # train docs duplicated inside train
        "web_eval_exact_copies": 6,      # eval docs leaked verbatim into train
        "web_eval_near_copies": 5,       # eval docs leaked with a one-word edit
        "instruct_eval_exact_copies": 4,
        "web_junk": 30,                  # spam / boilerplate
    },
}

# ------------------------------------------------------------- tokenizer
TOKENIZER = {"n_merges": 320, "train_splits": ["train"]}

# -------------------------------------------------------------- training
TRAIN = {
    "seq_len": 128,
    "batch_size": 8,              # sequences per step
    "total_steps": 80,
    "ckpt_every": 20,
    "crash_at_step": 47,          # deliberate crash in attempt 1
    "lr": 3e-3,
    "warmup_steps": 8,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "torch_threads": 2,           # fixed so CPU math is bit-reproducible
    "model": {"d_model": 128, "n_layers": 2, "n_heads": 4},
    "val_every": 20,
}

# ----------------------------------------------------------------- lanes
# policy: how documents of this data type become fixed-length sequences.
LANES = {
    "code": {
        "policy": "bestfit_nosplit_docmask",
        "protected": True,
        "floor": 0.25,            # >= 25% of every batch's sequences
        "budget_factor": 2.0,     # OPUS may scan <= 2x the tokens it needs
    },
    "instruct": {
        "policy": "bestfit_nosplit_response_only",
        "protected": True,
        "floor": 0.125,
        "budget_factor": 2.0,
    },
    "web": {
        "policy": "concat_split_docmask",
        "protected": False,
        "floor": 0.0,
        "backfill": True,         # absorbs sequence shortfalls of other lanes
        "budget_factor": None,    # unbounded (backfill must always fill)
    },
}
LANE_ORDER = ["code", "instruct", "web"]   # processing order; backfill last

PACKING = {"max_open_bins": 4, "min_residual": 4}

# ---------------------------------------------------- curriculum / OPUS
STAGES = [
    {"name": "warmup", "start": 1, "end": 20,
     "weights": {"web": 0.60, "code": 0.25, "instruct": 0.15},
     "opus": {"accept": 0.50, "reject": 0.45}},
    {"name": "main", "start": 21, "end": 60,
     "weights": {"web": 0.45, "code": 0.30, "instruct": 0.25},
     "opus": {"accept": 0.56, "reject": 0.45}},
    {"name": "anneal", "start": 61, "end": 80,
     "weights": {"web": 0.30, "code": 0.30, "instruct": 0.40},
     "opus": {"accept": 0.66, "reject": 0.45}},
]

OPUS = {
    "max_defers": 3,
    "aging_bonus": 0.04,          # +score per deferral (anti-starvation)
    "contamination_ngram": 16,
    "contamination_threshold": 0.6,
    "ngram_lanes": ["web"],       # prose lanes; templated lanes use exact hashes only
}

# ----------------------------------------------------------- demo phases
REPLAY = {"from_checkpoint_step": 20, "start": 21, "end": 40}
FORK = {
    "name": "fork_instruct_heavy",
    "from_checkpoint_step": 20,
    "end_step": 35,
    "diverge_at_step": 26,
    # from diverge_at_step onwards the fork trains with this mixture
    "weights": {"web": 0.25, "code": 0.25, "instruct": 0.50},
}


def data_config() -> dict:
    """The subset of config that determines the data stream (hashed into the plan)."""
    return copy.deepcopy({
        "seed": SEED, "corpus": CORPUS, "tokenizer": TOKENIZER,
        "seq_len": TRAIN["seq_len"], "batch_size": TRAIN["batch_size"],
        "total_steps": TRAIN["total_steps"], "lanes": LANES, "lane_order": LANE_ORDER,
        "packing": PACKING, "stages": STAGES, "opus": OPUS,
    })


def fork_stages(stages: list, diverge_at: int, weights: dict) -> list:
    """Stages for a fork: identical history until diverge_at, new weights after."""
    out = []
    for st in stages:
        if st["end"] < diverge_at:
            out.append(copy.deepcopy(st))
        elif st["start"] >= diverge_at:
            s = copy.deepcopy(st)
            s["name"] = st["name"] + "_fork"
            s["weights"] = dict(weights)
            out.append(s)
        else:  # stage straddles the divergence point: split it
            a = copy.deepcopy(st); a["end"] = diverge_at - 1
            b = copy.deepcopy(st); b["start"] = diverge_at
            b["name"] = st["name"] + "_fork"; b["weights"] = dict(weights)
            out += [a, b]
    return out
