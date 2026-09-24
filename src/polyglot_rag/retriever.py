"""Retrieval over a small corpus: BM25, dense, and RRF hybrid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from polyglot_rag.bm25 import BM25Index
from polyglot_rag.embed import Embedder, Matrix
from polyglot_rag.fusion import rrf, top_k

METHODS = ("bm25", "dense", "hybrid")
# Each system contributes its top CANDIDATE_DEPTH docs to fusion.
CANDIDATE_DEPTH = 100


@dataclass(frozen=True)
class Hit:
    doc_index: int
    doc_id: str
    rank: int
    text: str


class Retriever:
    """Index a list of passages once; retrieve with any of :data:`METHODS`."""

    def __init__(
        self,
        doc_ids: Sequence[str],
        docs: Sequence[str],
        embedder: Embedder | None = None,
    ) -> None:
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have equal length")
        self.doc_ids = list(doc_ids)
        self.docs = list(docs)
        self.bm25 = BM25Index(self.docs)
        self.embedder = embedder
        self._doc_emb: Matrix | None = None
        if embedder is not None:
            self._doc_emb = embedder.encode_passages(self.docs)

    def _require_dense(self) -> Matrix:
        if self._doc_emb is None or self.embedder is None:
            raise ValueError("dense/hybrid retrieval needs an embedder")
        return self._doc_emb

    def rank_batch(self, queries: Sequence[str], method: str, k: int = 10) -> list[list[int]]:
        """Ranked doc indices (best first, length ``k``) for each query."""
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
        depth = max(k, CANDIDATE_DEPTH)
        sparse: list[list[int]] = []
        dense: list[list[int]] = []
        if method in ("bm25", "hybrid"):
            sparse = [top_k(self.bm25.scores(q), depth) for q in queries]
        if method in ("dense", "hybrid"):
            doc_emb = self._require_dense()
            assert self.embedder is not None
            q_emb = self.embedder.encode_queries(queries)
            sims = q_emb @ doc_emb.T
            dense = [top_k(np.asarray(row, dtype=np.float64), depth) for row in sims]
        if method == "bm25":
            return [r[:k] for r in sparse]
        if method == "dense":
            return [r[:k] for r in dense]
        return [rrf([s, d], depth=k) for s, d in zip(sparse, dense, strict=True)]

    def search(self, query: str, method: str = "hybrid", k: int = 5) -> list[Hit]:
        ranked = self.rank_batch([query], method, k)[0]
        return [
            Hit(doc_index=i, doc_id=self.doc_ids[i], rank=r, text=self.docs[i])
            for r, i in enumerate(ranked, start=1)
        ]
