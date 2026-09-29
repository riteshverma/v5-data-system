"""Small byte-level BPE tokenizer that is trained once and then frozen.

Freezing means: the serialized tokenizer (merges + specials + pre-tokenizer
regex) is written once, its SHA-256 is recorded in ``tokenizer.lock.json``, and
every later load must present the same hash. Every shard manifest records the
tokenizer hash it was built with, so a shard can never be silently mixed with
tokens produced by a different vocabulary.
"""
from __future__ import annotations

import collections
import json
import re

from .util import atomic_write_json, canonical_json, read_json, sha256_text

SPECIALS = ["<|pad|>", "<|eos|>", "<|user|>", "<|assistant|>"]
PAD, EOS, USER, ASSISTANT = 256, 257, 258, 259
FIRST_MERGE_ID = 260
PRETOKENIZER = r" ?[A-Za-z]+| ?[0-9]| ?[^A-Za-z0-9\s]+|\s+"


class TokenizerIntegrityError(RuntimeError):
    pass


class FrozenTokenizer:
    def __init__(self, merges: list, pretokenizer: str = PRETOKENIZER):
        self.merges = tuple(tuple(m) for m in merges)
        self.pretokenizer = pretokenizer
        self._re = re.compile(pretokenizer)
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, s in enumerate(SPECIALS):
            self.vocab[256 + i] = s.encode()
        for i, (a, b) in enumerate(self.merges):
            self.vocab[FIRST_MERGE_ID + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict = {}

    # ---------------------------------------------------------- identity
    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE_ID + len(self.merges)

    def to_dict(self) -> dict:
        return {"schema": "v5.tokenizer/1", "type": "byte_level_bpe",
                "specials": {s: 256 + i for i, s in enumerate(SPECIALS)},
                "pretokenizer": self.pretokenizer,
                "merges": [list(m) for m in self.merges]}

    @property
    def sha256(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))

    # ------------------------------------------------------------ codec
    def _encode_word(self, word: bytes) -> list:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        ids = list(word)
        while len(ids) > 1:
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            new_id, out, i = FIRST_MERGE_ID + best_rank, [], 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best:
                    out.append(new_id); i += 2
                else:
                    out.append(ids[i]); i += 1
            ids = out
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list:
        out = []
        for m in self._re.finditer(text):
            out.extend(self._encode_word(m.group(0).encode("utf-8")))
        return out

    def decode(self, ids) -> str:
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def encode_document(self, doc: dict) -> tuple[list, int | None]:
        """Token ids for one document and, for chat data, where the loss region starts.

        plain text / code : text <eos>                       loss_start = None (all tokens)
        instruct          : <user> prompt <assistant> response <eos>
                            loss_start = index of the first response token
        """
        if doc["lane"] == "instruct":
            prompt = [USER] + self.encode(doc["prompt"]) + [ASSISTANT]
            return prompt + self.encode(" " + doc["response"]) + [EOS], len(prompt)
        return self.encode(doc["text"]) + [EOS], None

    # --------------------------------------------------- train / persist
    @classmethod
    def train(cls, texts, n_merges: int) -> "FrozenTokenizer":
        rx = re.compile(PRETOKENIZER)
        words = collections.Counter()
        for t in texts:
            for m in rx.finditer(t):
                words[m.group(0).encode("utf-8")] += 1
        seqs = {w: list(w) for w in words}
        merges = []
        for k in range(n_merges):
            pairs = collections.Counter()
            for w, c in words.items():
                s = seqs[w]
                for p in zip(s, s[1:]):
                    pairs[p] += c
            if not pairs:
                break
            # deterministic tie-break: highest count, then smallest pair
            best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            new_id = FIRST_MERGE_ID + k
            merges.append(best)
            for w in seqs:
                s, out, i = seqs[w], [], 0
                while i < len(s):
                    if i + 1 < len(s) and (s[i], s[i + 1]) == best:
                        out.append(new_id); i += 2
                    else:
                        out.append(s[i]); i += 1
                seqs[w] = out
        return cls(merges)

    def save_frozen(self, path: str, lock_path: str, provenance: dict) -> dict:
        atomic_write_json(path, self.to_dict(), indent=None)
        lock = {"schema": "v5.tokenizer_lock/1", "tokenizer_sha256": self.sha256,
                "vocab_size": self.vocab_size, "frozen": True, **provenance}
        atomic_write_json(lock_path, lock)
        return lock

    @classmethod
    def load_frozen(cls, path: str, lock_path: str) -> "FrozenTokenizer":
        """Load and verify against the lock; raises on any mismatch."""
        lock = read_json(lock_path)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        tok = cls(d["merges"], d["pretokenizer"])
        if tok.to_dict() != d:
            raise TokenizerIntegrityError("tokenizer file has unexpected fields")
        if tok.sha256 != lock["tokenizer_sha256"]:
            raise TokenizerIntegrityError(
                f"tokenizer hash {tok.sha256[:12]} != locked {lock['tokenizer_sha256'][:12]}")
        return tok
