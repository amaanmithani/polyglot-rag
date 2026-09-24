"""Rank helpers and Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

# Cormack et al. (2009) constant; fixed, not tuned.
RRF_K = 60


def top_k(scores: NDArray[np.floating], k: int) -> list[int]:
    """Indices of the ``k`` highest scores, best first; ties broken by lower index."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return []
    # Stable sort on negated scores gives deterministic tie-breaking.
    order = np.argsort(-scores, kind="stable")
    return [int(i) for i in order[:k]]


def rrf(
    rankings: Sequence[Sequence[int]], *, k: int = RRF_K, depth: int | None = None
) -> list[int]:
    """Fuse ranked lists of doc ids with Reciprocal Rank Fusion.

    ``score(d) = sum_r 1 / (k + rank_r(d))`` with 1-based ranks. Ties are broken by
    the best rank in any list, then by doc id, so output is deterministic.
    """
    fused: dict[int, float] = {}
    best: dict[int, int] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
            best[doc] = min(best.get(doc, rank), rank)
    order = sorted(fused, key=lambda d: (-fused[d], best[d], d))
    return order if depth is None else order[:depth]
