"""A small, dependency-free Okapi BM25 index (numpy scoring)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from polyglot_rag.tokenize import tokenize

# Robertson/Sparck-Jones defaults; fixed, not tuned on the evaluation data.
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


class BM25Index:
    """Okapi BM25 with the Lucene-style non-negative IDF ``log(1 + (N-df+.5)/(df+.5))``."""

    def __init__(
        self,
        docs: Sequence[str],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        if not docs:
            raise ValueError("BM25Index needs at least one document")
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.n_docs = len(docs)
        self._postings: dict[str, list[tuple[int, int]]] = {}
        lengths = np.zeros(self.n_docs, dtype=np.float64)
        for i, doc in enumerate(docs):
            toks = tokenizer(doc)
            lengths[i] = len(toks)
            for term, tf in Counter(toks).items():
                self._postings.setdefault(term, []).append((i, tf))
        self._doc_len = lengths
        self._avgdl = float(lengths.mean()) if lengths.mean() > 0 else 1.0
        self._idf = {
            t: math.log(1.0 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self._postings.items()
        }

    @property
    def vocab_size(self) -> int:
        return len(self._postings)

    def scores(self, query: str) -> NDArray[np.float64]:
        """BM25 score of every document for ``query`` (shape ``[n_docs]``)."""
        out = np.zeros(self.n_docs, dtype=np.float64)
        norm = self.k1 * (1.0 - self.b + self.b * self._doc_len / self._avgdl)
        for term, qtf in Counter(self.tokenizer(query)).items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idx = np.fromiter((d for d, _ in postings), dtype=np.int64, count=len(postings))
            tf = np.fromiter((f for _, f in postings), dtype=np.float64, count=len(postings))
            out[idx] += qtf * self._idf[term] * tf * (self.k1 + 1.0) / (tf + norm[idx])
        return out
