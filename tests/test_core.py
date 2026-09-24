from __future__ import annotations

import numpy as np
import pytest

from polyglot_rag.embed import HashingEmbedder, load_embedder
from polyglot_rag.metrics import bootstrap_ci, first_hit_rank, paired_delta_ci, per_query
from polyglot_rag.retriever import Retriever

DOCS = [
    "Reset your password from the sign-in page.",
    "Refunds are available within 14 days.",
    "Shipping takes three to seven days.",
]


def test_hashing_embedder_is_normalised_and_deterministic() -> None:
    e = HashingEmbedder(dim=64)
    a = e.encode_passages(["hello world", "hello world", ""])
    assert a.shape == (3, 64)
    assert np.allclose(np.linalg.norm(a[:2], axis=1), 1.0)
    assert np.allclose(a[0], a[1])
    assert e.encode_queries([]).shape == (0, 64)
    assert load_embedder("hashing").name.startswith("hashing")


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_retriever_methods_find_the_right_doc(method: str) -> None:
    r = Retriever(["a", "b", "c"], DOCS, HashingEmbedder())
    hits = r.search("how do I reset my password", method, k=2)
    assert hits[0].doc_id == "a"
    assert [h.rank for h in hits] == [1, 2]


def test_retriever_errors() -> None:
    with pytest.raises(ValueError):
        Retriever(["a"], DOCS)
    r = Retriever(["a", "b", "c"], DOCS)
    with pytest.raises(ValueError, match="embedder"):
        r.rank_batch(["x"], "dense")
    with pytest.raises(ValueError, match="unknown method"):
        r.rank_batch(["x"], "splade")


def test_per_query_metrics() -> None:
    pq = per_query([[3, 1, 2], [9] * 10 + [4], [0]], [1, 4, 7])
    assert pq["recall@5"].tolist() == [1.0, 0.0, 0.0]
    assert pq["mrr@10"].tolist() == [0.5, 0.0, 0.0]
    assert first_hit_rank([1, 2], 3) is None
    with pytest.raises(ValueError):
        per_query([[1]], [1, 2])


def test_bootstrap_ci_brackets_mean() -> None:
    v = np.array([1.0] * 80 + [0.0] * 20)
    e = bootstrap_ci(v, n_boot=500, seed=1)
    assert e.mean == pytest.approx(0.8)
    assert e.lo <= e.mean <= e.hi
    assert e.hi - e.lo < 0.25
    assert set(e.to_dict()) == {"mean", "lo", "hi"}
    same = bootstrap_ci(np.ones(10))
    assert same.lo == same.hi == 1.0
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]))


def test_paired_delta() -> None:
    a = np.array([1.0, 1.0, 1.0, 0.0])
    b = np.array([0.0, 1.0, 1.0, 0.0])
    assert paired_delta_ci(a, b).mean == pytest.approx(0.25)
    with pytest.raises(ValueError):
        paired_delta_ci(a, b[:2])
