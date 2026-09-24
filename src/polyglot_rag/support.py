"""Support-corpus loading and a tiny extractive answer layer (no LLM)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from polyglot_rag.bm25 import BM25Index
from polyglot_rag.embed import Embedder
from polyglot_rag.retriever import Hit


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    text: str

    @property
    def full(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


def load_corpus(path: Path | None = None) -> list[Doc]:
    """Load a JSONL corpus with ``id``, ``text`` and optional ``title``.

    Defaults to the bundled illustrative support FAQ (fictional product).
    """
    if path is None:
        raw = resources.files("polyglot_rag").joinpath("support_faq.jsonl").read_text("utf-8")
    else:
        raw = path.read_text(encoding="utf-8")
    docs = []
    for line in raw.splitlines():
        if line.strip():
            d = json.loads(line)
            docs.append(Doc(id=str(d["id"]), title=d.get("title", ""), text=d["text"]))
    if not docs:
        raise ValueError("corpus is empty")
    return docs


_SENT = re.compile("(?<=[.!?\u3002\uff01\uff1f\u0964])\\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def extract_answer(query: str, hit: Hit, embedder: Embedder | None) -> str:
    """Return the sentence of the top hit most similar to the query.

    Uses the dense embedder when available (works cross-lingually), else BM25.
    """
    body = hit.text.split("\n", 1)[-1]  # skip the title line of "title\ntext" docs
    sents = split_sentences(body) or [body]
    if len(sents) == 1:
        return sents[0]
    if embedder is not None:
        q = embedder.encode_queries([query])
        s = embedder.encode_passages(sents)
        return sents[int(np.argmax(s @ q[0]))]
    scores = BM25Index(sents).scores(query)
    return sents[int(np.argmax(scores))]
