"""Per-language retrieval evaluation on Belebele with bootstrap CIs."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from polyglot_rag import bm25, fusion
from polyglot_rag.data import (
    DATASET,
    PIVOT,
    REVISION,
    RetrievalSet,
    Row,
    build_set,
    load_rows,
    sample_indices,
)
from polyglot_rag.embed import Embedder
from polyglot_rag.metrics import KS, MRR_K, bootstrap_ci, paired_delta_ci, per_query
from polyglot_rag.retriever import Retriever

MODES = ("mono", "cross")
HEADLINE_METHOD = "hybrid"
HEADLINE_METRIC = "recall@10"

PerQuery = dict[str, NDArray[np.float64]]
Log = Callable[[str], None]


def _score_set(
    retriever: Retriever,
    rset: RetrievalSet,
    idx: Sequence[int],
    methods: Sequence[str],
) -> dict[str, PerQuery]:
    queries = [rset.questions[i] for i in idx]
    gold = [rset.relevant[i] for i in idx]
    return {m: per_query(retriever.rank_batch(queries, m, k=max(KS)), gold) for m in methods}


def _summarise(pq: dict[str, PerQuery], *, n_boot: int, seed: int, n_docs: int) -> dict[str, Any]:
    methods = list(pq)
    out: dict[str, Any] = {
        "n_queries": int(next(iter(pq.values()))[HEADLINE_METRIC].size),
        "n_docs": n_docs,
        "methods": {
            m: {
                metric: bootstrap_ci(v, n_boot=n_boot, seed=seed).to_dict()
                for metric, v in d.items()
            }
            for m, d in pq.items()
        },
        "deltas": {},
    }
    if HEADLINE_METHOD in pq:
        for other in methods:
            if other == HEADLINE_METHOD:
                continue
            out["deltas"][f"{HEADLINE_METHOD}-{other}"] = {
                metric: paired_delta_ci(
                    pq[HEADLINE_METHOD][metric], pq[other][metric], n_boot=n_boot, seed=seed
                ).to_dict()
                for metric in (HEADLINE_METRIC, f"mrr@{MRR_K}")
            }
    return out


def _macro(per_lang: dict[str, dict[str, PerQuery]], *, n_boot: int, seed: int) -> dict[str, Any]:
    """Macro average over languages; CI from a bootstrap stratified by language."""
    langs = list(per_lang)
    if not langs:
        return {}
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for m in per_lang[langs[0]]:
        out[m] = {}
        for metric in per_lang[langs[0]][m]:
            arrays = [per_lang[lang][m][metric] for lang in langs]
            point = float(np.mean([a.mean() for a in arrays]))
            boots = np.zeros(n_boot)
            for a in arrays:
                boots += a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
            boots /= len(arrays)
            lo, hi = np.quantile(boots, [0.025, 0.975])
            out[m][metric] = {
                "mean": round(point, 4),
                "lo": round(float(lo), 4),
                "hi": round(float(hi), 4),
            }
    return out


def _vs_reference(
    per_lang: dict[str, dict[str, PerQuery]], ref: str, *, n_boot: int, seed: int
) -> dict[str, Any]:
    """Paired delta (lang - ref) for the headline method; same questions in every language."""
    if ref not in per_lang or HEADLINE_METHOD not in per_lang[ref]:
        return {}
    base = per_lang[ref][HEADLINE_METHOD][HEADLINE_METRIC]
    return {
        lang: paired_delta_ci(
            d[HEADLINE_METHOD][HEADLINE_METRIC], base, n_boot=n_boot, seed=seed
        ).to_dict()
        for lang, d in per_lang.items()
        if lang != ref and HEADLINE_METHOD in d
    }


def run_eval(
    langs: Sequence[str],
    *,
    n: int,
    seed: int,
    methods: Sequence[str],
    modes: Sequence[str],
    embedder: Embedder | None,
    n_boot: int = 1000,
    root: Path | None = None,
    fetch: bool = True,
    log: Log = lambda _msg: None,
) -> dict[str, Any]:
    """Evaluate ``methods`` for every language in ``langs`` and return a JSON-able dict."""
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    t0 = time.perf_counter()
    pivot_rows: list[Row] = load_rows(PIVOT, root, fetch=fetch)
    idx = sample_indices(len(pivot_rows), n, seed)
    pivot_set = build_set(PIVOT, pivot_rows, pivot_rows)
    pivot_retriever: Retriever | None = None

    mono: dict[str, dict[str, PerQuery]] = {}
    cross: dict[str, dict[str, PerQuery]] = {}
    results: dict[str, Any] = {"monolingual": {}, "crosslingual": {}}
    for lang in langs:
        rows = pivot_rows if lang == PIVOT else load_rows(lang, root, fetch=fetch)
        rset = build_set(lang, rows, pivot_rows)
        if "mono" in modes:
            log(f"[mono ] {lang}: indexing {len(rset.docs)} passages")
            retriever = Retriever(rset.doc_ids, rset.docs, embedder)
            if lang == PIVOT:
                pivot_retriever = retriever
            mono[lang] = _score_set(retriever, rset, idx, methods)
            results["monolingual"][lang] = _summarise(
                mono[lang], n_boot=n_boot, seed=seed, n_docs=len(rset.docs)
            )
        if "cross" in modes and lang != PIVOT:
            if pivot_retriever is None:
                pivot_retriever = Retriever(pivot_set.doc_ids, pivot_set.docs, embedder)
            log(f"[cross] {lang} -> {PIVOT}")
            # Queries in `lang`, corpus + gold passage ids from the English pivot.
            xset = RetrievalSet(
                lang=lang,
                doc_ids=pivot_set.doc_ids,
                docs=pivot_set.docs,
                questions=rset.questions,
                relevant=pivot_set.relevant,
            )
            cross[lang] = _score_set(pivot_retriever, xset, idx, methods)
            results["crosslingual"][lang] = _summarise(
                cross[lang], n_boot=n_boot, seed=seed, n_docs=len(xset.docs)
            )

    results["macro"] = {
        "monolingual": _macro(mono, n_boot=n_boot, seed=seed),
        "crosslingual": _macro(cross, n_boot=n_boot, seed=seed),
    }
    results["vs_english"] = {"monolingual": _vs_reference(mono, PIVOT, n_boot=n_boot, seed=seed)}
    results["meta"] = {
        "dataset": DATASET,
        "revision": REVISION,
        "task": "question -> FLORES passage retrieval (1 relevant passage per query)",
        "languages": list(langs),
        "pivot": PIVOT,
        "n_per_language": len(idx),
        "n_total_questions": len(pivot_rows),
        "sample_seed": seed,
        "bootstrap": {"n_boot": n_boot, "seed": seed, "ci": "95% percentile, resampling queries"},
        "methods": list(methods),
        "modes": list(modes),
        "embedder": embedder.name if embedder is not None else None,
        "bm25": {"k1": bm25.DEFAULT_K1, "b": bm25.DEFAULT_B},
        "rrf_k": fusion.RRF_K,
        "headline": {"method": HEADLINE_METHOD, "metric": HEADLINE_METRIC},
        "sample_indices_sha": _digest(idx),
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    return results


def _digest(idx: Sequence[int]) -> str:
    import hashlib

    return hashlib.sha256(",".join(map(str, idx)).encode()).hexdigest()[:16]
