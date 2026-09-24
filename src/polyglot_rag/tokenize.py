"""Script-aware tokenizer for BM25.

Design (fixed a priori, not tuned on the evaluation data):

* NFKC-normalise and casefold.
* A token is a maximal run of Unicode letters (L*), combining marks (M*) and
  numbers (N*). Keeping marks inside tokens matters for Indic scripts, where
  vowel signs and viramas are category ``Mc``/``Mn`` and a naive ``\\w+`` regex
  would shred every word.
* Scripts written without spaces between words (Han, Kana, Thai, Lao, Khmer,
  Myanmar) are split into overlapping character bigrams, the standard
  dictionary-free fallback used by e.g. Lucene's CJKBigramFilter.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

# Unicode ranges for scripts that do not separate words with spaces.
_UNSPACED_RANGES: tuple[tuple[int, int], ...] = (
    (0x0E00, 0x0E7F),  # Thai
    (0x0E80, 0x0EFF),  # Lao
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK Ext A
    (0x4E00, 0x9FFF),  # CJK Unified
    (0xF900, 0xFAFF),  # CJK Compatibility
    (0x20000, 0x2FFFF),  # CJK Ext B+
)


@lru_cache(maxsize=65536)
def _is_word_char(ch: str) -> bool:
    return unicodedata.category(ch)[0] in ("L", "M", "N")


@lru_cache(maxsize=65536)
def is_unspaced(ch: str) -> bool:
    """True if ``ch`` belongs to a script written without inter-word spaces."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _UNSPACED_RANGES)


def _runs(text: str) -> list[str]:
    runs: list[str] = []
    buf: list[str] = []
    for ch in text:
        if _is_word_char(ch):
            buf.append(ch)
        elif buf:
            runs.append("".join(buf))
            buf = []
    if buf:
        runs.append("".join(buf))
    return runs


def _split_unspaced(run: str) -> list[str]:
    """Split a run into spaced segments and unspaced segments, bigram the latter."""
    out: list[str] = []
    seg: list[str] = []
    seg_unspaced: bool | None = None

    def flush() -> None:
        if not seg:
            return
        s = "".join(seg)
        if seg_unspaced:
            if len(s) == 1:
                out.append(s)
            else:
                out.extend(s[i : i + 2] for i in range(len(s) - 1))
        else:
            out.append(s)

    for ch in run:
        u = is_unspaced(ch)
        # Combining marks attach to whatever segment they follow.
        if unicodedata.category(ch)[0] == "M" and seg:
            seg.append(ch)
            continue
        if seg_unspaced is not None and u != seg_unspaced:
            flush()
            seg = []
        seg.append(ch)
        seg_unspaced = u
    flush()
    return out


def tokenize(text: str) -> list[str]:
    """Tokenise ``text`` into BM25 terms."""
    norm = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    for run in _runs(norm):
        if any(is_unspaced(c) for c in run):
            tokens.extend(_split_unspaced(run))
        else:
            tokens.append(run)
    return tokens
