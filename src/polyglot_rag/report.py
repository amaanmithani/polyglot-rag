"""Render the README results section from a committed results JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from polyglot_rag.data import LANGUAGE_NAMES

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
METHOD_LABEL = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid (RRF)"}


def _ci(e: dict[str, float]) -> str:
    return f"{e['mean']:.3f} [{e['lo']:.2f}, {e['hi']:.2f}]"


def _pt(e: dict[str, float]) -> str:
    return f"{e['mean']:.3f}"


def _headline_table(
    block: dict[str, Any], macro: dict[str, Any], vs_eng: dict[str, Any] | None
) -> list[str]:
    if not block:
        return ["_(not run)_"]
    methods = list(next(iter(block.values()))["methods"])
    head = "hybrid" if "hybrid" in methods else methods[-1]
    others = [m for m in methods if m != head]
    cols = ["Language", "Code", "n"]
    cols += [f"{METHOD_LABEL.get(m, m)} R@10" for m in others]
    cols += [
        f"{METHOD_LABEL.get(head, head)} {x}" for x in ("R@5", "R@10 [95% CI]", "MRR@10 [95% CI]")
    ]
    if vs_eng is not None:
        cols.append("vs English (R@10)")
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for lang, r in block.items():
        m = r["methods"]
        row = [LANGUAGE_NAMES.get(lang, lang), f"`{lang}`", str(r["n_queries"])]
        row += [_pt(m[o]["recall@10"]) for o in others]
        row += [_pt(m[head]["recall@5"]), _ci(m[head]["recall@10"]), _ci(m[head]["mrr@10"])]
        if vs_eng is not None:
            d = vs_eng.get(lang)
            if d is None:
                row.append("reference")
            else:
                flag = " (below EN)" if d["hi"] < 0 else ""
                row.append(f"{d['mean']:+.3f} [{d['lo']:+.2f}, {d['hi']:+.2f}]{flag}")
        lines.append("| " + " | ".join(row) + " |")
    if macro:
        row = ["**Macro avg**", f"{len(block)} langs", "-"]
        row += [_pt(macro[o]["recall@10"]) for o in others]
        row += [
            _pt(macro[head]["recall@5"]),
            _ci(macro[head]["recall@10"]),
            _ci(macro[head]["mrr@10"]),
        ]
        if vs_eng is not None:
            row.append("")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _findings(res: dict[str, Any]) -> list[str]:
    out: list[str] = []
    mono = res.get("monolingual", {})
    vs = res.get("vs_english", {}).get("monolingual", {})
    if mono and "hybrid" in next(iter(mono.values()))["methods"]:
        ranked = sorted(
            mono.items(), key=lambda kv: kv[1]["methods"]["hybrid"]["recall@10"]["mean"]
        )
        worst = ", ".join(
            f"{LANGUAGE_NAMES.get(k, k)} ({v['methods']['hybrid']['recall@10']['mean']:.3f})"
            for k, v in ranked[:3]
        )
        out.append(f"- Lowest monolingual hybrid R@10: {worst}.")
        weak = [LANGUAGE_NAMES.get(k, k) for k, d in vs.items() if d["hi"] < 0]
        if weak:
            out.append(
                "- Significantly below English (paired bootstrap 95% CI of the R@10 gap "
                f"excludes 0): {', '.join(weak)}."
            )
        else:
            out.append("- No language is significantly below English on hybrid R@10.")
        worse = [
            LANGUAGE_NAMES.get(k, k)
            for k, v in mono.items()
            if "hybrid-dense" in v["deltas"] and v["deltas"]["hybrid-dense"]["recall@10"]["hi"] < 0
        ]
        better = [
            LANGUAGE_NAMES.get(k, k)
            for k, v in mono.items()
            if "hybrid-dense" in v["deltas"] and v["deltas"]["hybrid-dense"]["recall@10"]["lo"] > 0
        ]
        out.append(
            f"- Hybrid significantly beats dense-only on R@10 in {len(better)}/{len(mono)} "
            f"languages and is significantly worse in {len(worse)}"
            + (f" ({', '.join(worse)})" if worse else "")
            + "."
        )
    cross = res.get("crosslingual", {})
    macro = res.get("macro", {})
    if cross and macro.get("crosslingual") and macro.get("monolingual"):
        mm, cm = macro["monolingual"], macro["crosslingual"]
        parts = [
            f"{METHOD_LABEL.get(m, m)} {mm[m]['recall@10']['mean']:.3f} -> "
            f"{cm[m]['recall@10']['mean']:.3f}"
            for m in cm
            if m in mm
        ]
        out.append(
            "- Macro R@10, monolingual -> cross-lingual (query in X, English corpus; "
            "note the cross-lingual macro excludes English): " + "; ".join(parts) + "."
        )
    return out


def _detail_table(block: dict[str, Any]) -> list[str]:
    if not block:
        return []
    methods = list(next(iter(block.values()))["methods"])
    metrics = ["recall@5", "recall@10", "mrr@10"]
    cols = ["Language"] + [f"{METHOD_LABEL.get(m, m)} {x}" for m in methods for x in metrics]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for lang, r in block.items():
        cells = [f"`{lang}`"] + [_ci(r["methods"][m][x]) for m in methods for x in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render(res: dict[str, Any], source: str) -> str:
    meta = res["meta"]
    lines = [
        START,
        f"_Generated by `polyglot-rag report` from [`{source}`]({source}). Do not edit by hand._",
        "",
        f"Dataset `{meta['dataset']}` @ `{meta['revision'][:10]}`; "
        f"{meta['n_per_language']} of {meta['n_total_questions']} questions per language "
        f"(seed {meta['sample_seed']}, identical question ids in every language); "
        f"embedder `{meta['embedder']}`; BM25 k1={meta['bm25']['k1']} b={meta['bm25']['b']}; "
        f"RRF k={meta['rrf_k']}; {meta['bootstrap']['n_boot']} bootstrap resamples.",
        "",
        "### Monolingual (query and passages in the same language)",
        "",
        *_headline_table(
            res.get("monolingual", {}),
            res.get("macro", {}).get("monolingual", {}),
            res.get("vs_english", {}).get("monolingual", {}),
        ),
        "",
        "### Cross-lingual (query in language X, passages in English)",
        "",
        *_headline_table(
            res.get("crosslingual", {}), res.get("macro", {}).get("crosslingual", {}), None
        ),
        "",
        "### Findings (auto-generated from the JSON)",
        "",
        *_findings(res),
        "",
        "<details><summary>All methods x all metrics, monolingual (95% CIs)</summary>",
        "",
        *_detail_table(res.get("monolingual", {})),
        "",
        "</details>",
        "",
        "<details><summary>All methods x all metrics, cross-lingual (95% CIs)</summary>",
        "",
        *_detail_table(res.get("crosslingual", {})),
        "",
        "</details>",
        END,
    ]
    return "\n".join(lines)


def update_readme(readme: Path, results: Path) -> str:
    """Replace the block between the RESULTS markers in ``readme``; return the block."""
    res = json.loads(results.read_text(encoding="utf-8"))
    block = render(res, results.as_posix())
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(f"{readme} is missing the {START} / {END} markers")
    pre, rest = text.split(START, 1)
    _, post = rest.split(END, 1)
    readme.write_text(pre + block + post, encoding="utf-8")
    return block
