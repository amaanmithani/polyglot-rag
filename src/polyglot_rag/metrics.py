"""Retrieval metrics with percentile-bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

KS = (5, 10)
MRR_K = 10


def first_hit_rank(ranking: Sequence[int], relevant: int) -> int | None:
    """1-based rank of the relevant doc in ``ranking``, or None if absent."""
    for r, d in enumerate(ranking, start=1):
        if d == relevant:
            return r
    return None


def per_query(
    rankings: Sequence[Sequence[int]], relevant: Sequence[int]
) -> dict[str, NDArray[np.float64]]:
    """Per-query recall@k and RR@10 (single relevant passage per query)."""
    if len(rankings) != len(relevant):
        raise ValueError("rankings and relevant must align")
    ranks = [first_hit_rank(r, g) for r, g in zip(rankings, relevant, strict=True)]
    out: dict[str, NDArray[np.float64]] = {}
    for k in KS:
        out[f"recall@{k}"] = np.array([1.0 if r is not None and r <= k else 0.0 for r in ranks])
    out[f"mrr@{MRR_K}"] = np.array(
        [1.0 / r if r is not None and r <= MRR_K else 0.0 for r in ranks]
    )
    return out


@dataclass(frozen=True)
class Estimate:
    mean: float
    lo: float
    hi: float

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def bootstrap_ci(
    values: NDArray[np.float64], *, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> Estimate:
    """Mean with a percentile bootstrap (1 - alpha) CI, resampling queries."""
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(values.mean()), float(lo), float(hi))


def paired_delta_ci(
    a: NDArray[np.float64], b: NDArray[np.float64], *, n_boot: int = 1000, seed: int = 0
) -> Estimate:
    """Bootstrap CI of mean(a - b) with queries resampled jointly (paired)."""
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal shape")
    return bootstrap_ci(a - b, n_boot=n_boot, seed=seed)
