from __future__ import annotations

import numpy as np
import pytest

from polyglot_rag.bm25 import BM25Index
from polyglot_rag.fusion import rrf, top_k
from polyglot_rag.tokenize import is_unspaced, tokenize


def test_latin_casefold_and_punctuation() -> None:
    assert tokenize("Hello, WORLD! straße") == ["hello", "world", "strasse"]


def test_indic_marks_stay_inside_words() -> None:
    assert tokenize("हिन्दी भाषा") == ["हिन्दी", "भाषा"]
    assert tokenize("தமிழ்") == ["தமிழ்"]


def test_cjk_and_thai_become_bigrams() -> None:
    assert tokenize("东京是首都") == ["东京", "京是", "是首", "首都"]
    assert tokenize("ภาษา") == ["ภา", "าษ", "ษา"]
    assert tokenize("字") == ["字"]


def test_mixed_script_run_splits() -> None:
    assert tokenize("iPhone15手机") == ["iphone15", "手机"]
    assert is_unspaced("日") and not is_unspaced("a")


def test_bm25_ranks_matching_doc_first() -> None:
    idx = BM25Index(["the cat sat", "dogs bark loudly", "a cat and a dog"])
    s = idx.scores("cat")
    assert s[1] == 0.0
    assert top_k(s, 1) == [0]  # shorter doc wins
    assert idx.vocab_size > 0
    assert np.all(idx.scores("unknownterm") == 0)


def test_bm25_requires_docs() -> None:
    with pytest.raises(ValueError):
        BM25Index([])


def test_top_k_is_stable_on_ties() -> None:
    assert top_k(np.array([1.0, 3.0, 3.0, 0.0]), 3) == [1, 2, 0]
    assert top_k(np.array([1.0]), 0) == []
    assert top_k(np.array([1.0, 2.0]), 5) == [1, 0]


def test_rrf_fuses_and_breaks_ties() -> None:
    # docs 1 and 3: 1/61 + 1/63 > doc 2: 2/62; 1 vs 3 tie -> best rank, then lower id
    assert rrf([[1, 2, 3], [3, 2, 1]]) == [1, 3, 2]
    assert rrf([[5], [6]], depth=1) == [5]
