import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from v5data import build  # noqa: E402
from v5data.shards import Catalog  # noqa: E402


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Documents -> frozen tokenizer -> shards -> contamination scan -> plan (no training)."""
    art = str(tmp_path_factory.mktemp("art"))
    corpus, _ = build.build_corpus(art)
    tok, lock = build.build_tokenizer(art, corpus)
    build.build_shards(art, corpus, tok, lock["tokenizer_sha256"])
    build.build_contamination(art, Catalog(art, lock["tokenizer_sha256"]))
    build.compile_mixture(art)
    return {"art": art, "corpus": corpus, "tok": tok, "lock": lock, "env": build.Env(art)}
