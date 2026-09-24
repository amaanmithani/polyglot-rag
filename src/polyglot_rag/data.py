"""Belebele (facebook/belebele) turned into a passage-retrieval benchmark.

Belebele is a parallel multiple-choice reading-comprehension set: 900 questions
over 488 FLORES-200 passages, professionally translated into 122 language
variants. Row ``i`` is the same item in every language, which gives us a
*parallel* retrieval benchmark: the query is the question, the corpus is the
language's 488 passages, the single relevant document is the question's passage.
The same framing is used by MTEB's ``BelebeleRetrieval`` task.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

DATASET = "facebook/belebele"
# Pinned dataset commit so results are reproducible.
REVISION = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
URL = "https://huggingface.co/datasets/{ds}/resolve/{rev}/data/{lang}.jsonl"
PIVOT = "eng_Latn"

# 24 languages: 10 scripts, high- to low-resource. Chosen before any results were seen.
LANGUAGES: tuple[str, ...] = (
    "eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "por_Latn", "ita_Latn",
    "nld_Latn", "pol_Latn", "rus_Cyrl", "ukr_Cyrl", "tur_Latn", "arb_Arab",
    "heb_Hebr", "hin_Deva", "ben_Beng", "tam_Taml", "zho_Hans", "jpn_Jpan",
    "kor_Hang", "vie_Latn", "tha_Thai", "ind_Latn", "swh_Latn", "yor_Latn",
)  # fmt: skip

LANGUAGE_NAMES: dict[str, str] = {
    "eng_Latn": "English", "deu_Latn": "German", "fra_Latn": "French",
    "spa_Latn": "Spanish", "por_Latn": "Portuguese", "ita_Latn": "Italian",
    "nld_Latn": "Dutch", "pol_Latn": "Polish", "rus_Cyrl": "Russian",
    "ukr_Cyrl": "Ukrainian", "tur_Latn": "Turkish", "arb_Arab": "Arabic",
    "heb_Hebr": "Hebrew", "hin_Deva": "Hindi", "ben_Beng": "Bengali",
    "tam_Taml": "Tamil", "zho_Hans": "Chinese (Simpl.)", "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean", "vie_Latn": "Vietnamese", "tha_Thai": "Thai",
    "ind_Latn": "Indonesian", "swh_Latn": "Swahili", "yor_Latn": "Yoruba",
}  # fmt: skip


@dataclass(frozen=True)
class Row:
    link: str
    number: int
    passage: str
    question: str


@dataclass(frozen=True)
class RetrievalSet:
    """One language's corpus plus aligned queries."""

    lang: str
    doc_ids: list[str]
    docs: list[str]
    questions: list[str]  # all 900, row-aligned with the pivot language
    relevant: list[int]  # doc index of each question's passage


def data_dir() -> Path:
    return Path(os.environ.get("POLYGLOT_RAG_DATA", "data/cache/belebele"))


def download(lang: str, dest: Path, *, timeout: float = 60.0) -> Path:  # pragma: no cover - network
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{lang}.jsonl"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = URL.format(ds=DATASET, rev=REVISION, lang=lang)
    tmp = path.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes():
                fh.write(chunk)
    tmp.rename(path)
    return path


def read_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append(
                Row(
                    link=d["link"],
                    number=int(d["question_number"]),
                    passage=d["flores_passage"],
                    question=d["question"],
                )
            )
    return rows


def load_rows(lang: str, root: Path | None = None, *, fetch: bool = True) -> list[Row]:
    root = root or data_dir()
    path = root / f"{lang}.jsonl"
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(path)
        download(lang, root)  # pragma: no cover - network
    return read_rows(path)


def passage_keys(pivot: Sequence[Row]) -> list[int]:
    """Map each row to a passage index, defined on the pivot language.

    Two rows share a passage iff they have the same source link and the same
    pivot-language passage text.
    """
    seen: dict[tuple[str, str], int] = {}
    keys: list[int] = []
    for row in pivot:
        keys.append(seen.setdefault((row.link, row.passage), len(seen)))
    return keys


def build_set(lang: str, rows: Sequence[Row], pivot: Sequence[Row]) -> RetrievalSet:
    """Build a language's corpus, with passage identities taken from ``pivot``.

    Rows are re-ordered to the pivot's order by their ``(link, question_number)``
    key: the per-language files are *not* stored in the same row order.
    """
    if len(rows) != len(pivot):
        raise ValueError(f"{lang}: {len(rows)} rows vs {len(pivot)} pivot rows")
    by_key = {(r.link, r.number): r for r in rows}
    missing = [(p.link, p.number) for p in pivot if (p.link, p.number) not in by_key]
    if missing or len(by_key) != len(rows):
        raise ValueError(f"{lang}: rows are not aligned with the pivot language ({missing[:1]})")
    rows = [by_key[(p.link, p.number)] for p in pivot]
    keys = passage_keys(pivot)
    n_passages = max(keys) + 1
    docs: list[str | None] = [None] * n_passages
    for key, row in zip(keys, rows, strict=True):
        if docs[key] is None:
            docs[key] = row.passage
    return RetrievalSet(
        lang=lang,
        doc_ids=[f"p{k:03d}" for k in range(n_passages)],
        docs=[d or "" for d in docs],
        questions=[r.question for r in rows],
        relevant=keys,
    )


def sample_indices(n_total: int, n: int, seed: int) -> list[int]:
    """Fixed-seed sample of ``n`` question indices, shared by every language."""
    if n >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_total, size=n, replace=False))
