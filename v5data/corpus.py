"""Deterministic synthetic corpus with three data types and three splits.

The corpus is small but deliberately *dirty*, so that every defence in the
data system has something real to catch:

* exact duplicates inside the web training split (-> OPUS duplicate reject)
* evaluation documents leaked verbatim / with a one-word edit into training
  (-> evaluation firewall contamination reject)
* spam / boilerplate documents (-> OPUS low-quality reject)
* code of mixed quality (-> OPUS deferrals and protected-floor overrides)

Documents have learnable structure (templated facts, arithmetic, small Python
functions) so the tiny model's loss visibly falls.
"""
from __future__ import annotations

import json
import os
import random

from .util import canonical_json, sha256_text

NOUNS = ["river", "garden", "engine", "library", "market", "forest", "harbor", "bridge",
         "school", "tower", "valley", "island", "castle", "museum", "station", "village"]
ADJS = ["quiet", "ancient", "busy", "narrow", "bright", "cold", "hidden", "wide",
        "green", "old", "small", "famous"]
VERBS = ["crosses", "faces", "surrounds", "follows", "protects", "overlooks", "joins", "shelters"]
PEOPLE = ["farmers", "students", "sailors", "traders", "workers", "children", "visitors", "painters"]
ACTS = ["gather", "meet", "rest", "trade", "study", "walk", "work", "sing"]
TIMES = ["at dawn", "in the evening", "every morning", "during winter", "on market days",
         "after the rain", "in summer", "at night"]
COUNTRIES = ["Aldoria", "Brevia", "Castellan", "Dunmore", "Elvania", "Farrowland",
             "Galdor", "Hestia", "Istrana", "Jorvik", "Kelmar", "Lunaris"]
CAPITALS = ["Aldport", "Brevik", "Castra", "Dunhaven", "Elvara", "Farrow",
            "Galden", "Hestholm", "Istra", "Jorby", "Kelstad", "Lunara"]
FUNCS = [("add", "+", "sum"), ("sub", "-", "difference"), ("mul", "*", "product"),
         ("maximum", "max", "larger value"), ("minimum", "min", "smaller value")]
WORDS = ["apple", "stone", "river", "cloud", "tiger", "lemon", "piano", "rocket",
         "garden", "silver", "window", "planet"]
SPAM = ["click", "here", "buy", "now", "free", "offer", "win", "cash", "deal", "!!!", "$$$", "best"]


def _web_doc(rng: random.Random) -> str:
    sents = []
    for _ in range(rng.randint(3, 8)):
        k = rng.random()
        if k < 0.35:
            sents.append(f"The {rng.choice(ADJS)} {rng.choice(NOUNS)} {rng.choice(VERBS)} "
                         f"the {rng.choice(NOUNS)}.")
        elif k < 0.65:
            sents.append(f"The {rng.choice(PEOPLE)} {rng.choice(ACTS)} near the "
                         f"{rng.choice(NOUNS)} {rng.choice(TIMES)}.")
        else:
            i = rng.randrange(len(COUNTRIES))
            sents.append(f"The capital of {COUNTRIES[i]} is {CAPITALS[i]}.")
    return " ".join(sents)


def _junk_doc(rng: random.Random) -> str:
    return " ".join(rng.choice(SPAM[:5]) for _ in range(rng.randint(20, 40))) + " " + \
        " ".join(rng.choice(SPAM) for _ in range(10))


def _code_doc(rng: random.Random) -> str:
    """Code of varying quality: clean functions vs. copy-pasted filler lines."""
    parts = []
    messy = rng.random()
    for _ in range(rng.randint(1, 3)):
        name, op, desc = rng.choice(FUNCS)
        a, b = rng.choice([("a", "b"), ("x", "y"), ("left", "right")])
        scale = rng.randint(2, 9)
        if op in ("max", "min"):
            body = f"    return {op}({a}, {b}) * {scale}"
        else:
            body = f"    return ({a} {op} {b}) * {scale}"
        fn = [f"def {name}_{rng.choice(WORDS)}_{rng.choice(WORDS)}({a}, {b}):",
              f'    """Return the {desc} of {a} and {b}."""']
        if messy > 0.6:  # low-quality: repeated filler lines and TODO noise
            fn += ["    # TODO: fix", "    # TODO: fix"] * rng.randint(1, 1 + int(messy * 6))
            fn += ["    tmp = 0", "    tmp = 0"] * rng.randint(1, 1 + int(messy * 4))
        fn.append(body)
        parts.append("\n".join(fn))
    return "\n\n".join(parts) + "\n"


def _instruct_doc(rng: random.Random) -> dict:
    k = rng.random()
    if k < 0.4:
        a, b = rng.randint(1, 199), rng.randint(1, 199)
        if rng.random() < 0.5:
            return {"prompt": f"What is {a} plus {b}?", "response": f"{a} plus {b} is {a + b}."}
        return {"prompt": f"What is {a} minus {b}?", "response": f"{a} minus {b} is {a - b}."}
    if k < 0.7:
        i = rng.randrange(len(COUNTRIES))
        adj, noun = rng.choice(ADJS), rng.choice(NOUNS)
        return {"prompt": f"Name the capital of {COUNTRIES[i]} and describe its {noun}.",
                "response": f"The capital of {COUNTRIES[i]} is {CAPITALS[i]}. "
                            f"Its {noun} is {adj}."}
    w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(4, 7)))
    return {"prompt": f"Spell the word {w} backwards.",
            "response": f"The word {w} backwards is {w[::-1]}."}


def doc_text(doc: dict) -> str:
    """Canonical text used for content hashing, quality and contamination."""
    if doc["lane"] == "instruct":
        return canonical_json({"prompt": doc["prompt"], "response": doc["response"]})
    return doc["text"]


def generate(cfg: dict) -> dict:
    """Return {(lane, split): [doc, ...]} fully determined by cfg['seed']."""
    rng = random.Random(cfg["seed"])
    out: dict = {}
    for lane in ("web", "code", "instruct"):
        for split in ("train", "val", "eval"):
            n = cfg["counts"][lane][split]
            docs = []
            for i in range(n):
                d = {"doc_id": f"{lane}-{split}-{i:05d}", "lane": lane, "split": split}
                if lane == "web":
                    d["text"] = _web_doc(rng)
                elif lane == "code":
                    d["text"] = _code_doc(rng)
                else:
                    d.update(_instruct_doc(rng))
                d["origin"] = "generated"
                docs.append(d)
            out[(lane, split)] = docs

    p = cfg["planted"]
    web, n = out[("web", "train")], len(out[("web", "train")])

    def add(lane, src, origin, **over):
        lst = out[(lane, "train")]
        d = {k: v for k, v in src.items() if k not in ("doc_id", "split", "origin")}
        d.update(over)
        d.update({"doc_id": f"{lane}-train-{len(lst):05d}", "split": "train", "origin": origin})
        lst.append(d)

    for j in range(p["web_exact_duplicates"]):
        add("web", web[rng.randrange(n)], "planted_duplicate")
    for j in range(p["web_eval_exact_copies"]):
        add("web", out[("web", "eval")][j], "planted_eval_leak_exact")
    for j in range(p["web_eval_near_copies"]):
        src = out[("web", "eval")][10 + j]
        words = src["text"].split(" ")
        words[1] = "nameless"
        add("web", src, "planted_eval_leak_near", text=" ".join(words))
    for j in range(p["instruct_eval_exact_copies"]):
        add("instruct", out[("instruct", "eval")][j], "planted_eval_leak_exact")
    for j in range(p["web_junk"]):
        add("web", {"lane": "web", "text": _junk_doc(rng)}, "planted_junk")

    # Shuffle planted docs into the training stream so they are not all at the end,
    # then re-number doc ids so the id reveals nothing about provenance.
    for lane in ("web", "instruct"):
        lst = out[(lane, "train")]
        rng.shuffle(lst)
        for i, d in enumerate(lst):
            d["doc_id"] = f"{lane}-train-{i:05d}"
    for docs in out.values():
        for d in docs:
            d["content_sha256"] = sha256_text(doc_text(d))
    return out


def write_raw(corpus: dict, raw_dir: str) -> dict:
    """Persist raw documents (the 'documents' stage) and return their file hashes."""
    os.makedirs(raw_dir, exist_ok=True)
    hashes = {}
    for (lane, split), docs in sorted(corpus.items()):
        path = os.path.join(raw_dir, f"{lane}.{split}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for d in docs:
                f.write(json.dumps(d, sort_keys=True) + "\n")
        from .util import sha256_file
        hashes[f"{lane}.{split}.jsonl"] = sha256_file(path)
    return hashes
