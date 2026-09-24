"""Embedders: a real multilingual sentence-transformers model and an offline hashing fallback."""

from __future__ import annotations

import hashlib
import os
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]


class Embedder(Protocol):
    """Anything that maps queries and passages into a shared, L2-normalised space."""

    name: str

    def encode_queries(self, texts: Sequence[str]) -> Matrix: ...

    def encode_passages(self, texts: Sequence[str]) -> Matrix: ...


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    query_prefix: str
    passage_prefix: str


# Known small multilingual models (both ~118M params, <500 MB on disk).
MODELS: dict[str, ModelSpec] = {
    "e5-small": ModelSpec("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "minilm": ModelSpec("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
}
DEFAULT_MODEL = "e5-small"


def _l2(x: NDArray[Any]) -> Matrix:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Matrix = (x / norms).astype(np.float32)
    return out


class HashingEmbedder:
    """Character n-gram feature hashing. Offline, deterministic, no model download.

    It is *not* a semantic model: it has no cross-lingual ability. It exists so the
    pipeline and tests run without network, and as a sanity floor.
    """

    def __init__(self, dim: int = 512, ngram: int = 3) -> None:
        self.dim = dim
        self.ngram = ngram
        self.name = f"hashing-{dim}-{ngram}"

    def _one(self, text: str) -> NDArray[np.float32]:
        v = np.zeros(self.dim, dtype=np.float32)
        s = f" {unicodedata.normalize('NFKC', text).casefold()} "
        for i in range(max(1, len(s) - self.ngram + 1)):
            g = s[i : i + self.ngram].encode("utf-8")
            h = int.from_bytes(hashlib.blake2b(g, digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def _encode(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(np.stack([self._one(t) for t in texts]))

    def encode_queries(self, texts: Sequence[str]) -> Matrix:
        return self._encode(texts)

    def encode_passages(self, texts: Sequence[str]) -> Matrix:
        return self._encode(texts)


def default_cache_dir() -> Path:
    return Path(os.environ.get("POLYGLOT_RAG_CACHE", ".cache")) / "embeddings"


class SentenceTransformerEmbedder:  # pragma: no cover - needs the `dense` extra + a model download
    """CPU sentence-transformers embedder with an on-disk embedding cache."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        batch_size: int = 32,
        cache_dir: Path | None = None,
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval needs the `dense` extra: uv sync --extra dense"
            ) from exc
        self.spec = MODELS.get(model) or ModelSpec(model, "", "")
        self.name = self.spec.hf_id
        self.batch_size = batch_size
        self.cache_dir = (cache_dir or default_cache_dir()) / self.name.replace("/", "__")
        self._model = SentenceTransformer(self.spec.hf_id, device=device)

    def _encode(self, texts: Sequence[str], prefix: str) -> Matrix:
        payload = "\x1f".join(prefix + t for t in texts).encode("utf-8")
        key = hashlib.sha256(payload).hexdigest()[:24]
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            cached: Matrix = np.load(path).astype(np.float32)
            return cached
        emb = self._model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out = _l2(emb)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, out)
        return out

    def encode_queries(self, texts: Sequence[str]) -> Matrix:
        return self._encode(texts, self.spec.query_prefix)

    def encode_passages(self, texts: Sequence[str]) -> Matrix:
        return self._encode(texts, self.spec.passage_prefix)


def load_embedder(name: str, **kwargs: Any) -> Embedder:
    """Return the embedder called ``name`` (``hashing`` or a sentence-transformers model)."""
    if name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(name, **kwargs)  # pragma: no cover
