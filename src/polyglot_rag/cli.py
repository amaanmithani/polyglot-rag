"""Command-line interface: ``polyglot-rag {eval,query,report,download}``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from polyglot_rag.data import LANGUAGES, data_dir, download
from polyglot_rag.embed import DEFAULT_MODEL, Embedder, load_embedder
from polyglot_rag.evaluate import MODES, run_eval
from polyglot_rag.report import update_readme
from polyglot_rag.retriever import METHODS, Retriever
from polyglot_rag.support import extract_answer, load_corpus


def _csv(value: str, allowed: Sequence[str], all_value: Sequence[str] | None = None) -> list[str]:
    if value == "all":
        return list(all_value if all_value is not None else allowed)
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown value(s) {bad}; allowed: {list(allowed)}")
    return items


def _embedder(name: str, batch_size: int, methods: Sequence[str]) -> Embedder | None:
    if not any(m in ("dense", "hybrid") for m in methods):
        return None
    if name == "hashing":
        return load_embedder("hashing")
    return load_embedder(name, batch_size=batch_size)  # pragma: no cover - model download


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="polyglot-rag", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="build per-language indexes and run the Belebele evaluation")
    e.add_argument("--langs", default="all", help=f"'all' or comma list of {', '.join(LANGUAGES)}")
    e.add_argument("--n", type=int, default=100, help="questions sampled per language")
    e.add_argument("--seed", type=int, default=13)
    e.add_argument("--methods", default="all", help=f"'all' or comma list of {METHODS}")
    e.add_argument("--modes", default="all", help=f"'all' or comma list of {MODES}")
    e.add_argument("--model", default=DEFAULT_MODEL, help="e5-small | minilm | hashing | HF id")
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    e.add_argument("--data", type=Path, default=None, help="Belebele jsonl cache dir")
    e.add_argument("--no-fetch", action="store_true", help="fail instead of downloading")
    e.add_argument("--out", type=Path, default=Path("results/belebele.json"))

    q = sub.add_parser("query", help="retrieve from a support corpus")
    q.add_argument("text")
    q.add_argument("--corpus", type=Path, default=None, help="JSONL with id,title,text")
    q.add_argument("--method", choices=METHODS, default="dense")
    q.add_argument("--model", default=DEFAULT_MODEL)
    q.add_argument("-k", type=int, default=3)
    q.add_argument("--answer", action="store_true", help="also print an extractive answer")
    q.add_argument("--json", action="store_true", help="machine-readable output")

    r = sub.add_parser("report", help="render the README results table from a results JSON")
    r.add_argument("--results", type=Path, default=Path("results/belebele.json"))
    r.add_argument("--readme", type=Path, default=Path("README.md"))

    d = sub.add_parser("download", help="download Belebele files for the chosen languages")
    d.add_argument("--langs", default="all")
    d.add_argument("--data", type=Path, default=None)
    return p


def cmd_eval(a: argparse.Namespace) -> int:
    langs = _csv(a.langs, LANGUAGES)
    methods = _csv(a.methods, METHODS)
    modes = _csv(a.modes, MODES)
    emb = _embedder(a.model, a.batch_size, methods)
    res = run_eval(
        langs,
        n=a.n,
        seed=a.seed,
        methods=methods,
        modes=modes,
        embedder=emb,
        n_boot=a.boot,
        root=a.data,
        fetch=not a.no_fetch,
        log=lambda m: print(m, file=sys.stderr, flush=True),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    macro = res["macro"]
    for mode, block in macro.items():
        for m, metrics in block.items():
            r10 = metrics["recall@10"]
            print(
                f"{mode:13s} {m:7s} macro R@10 {r10['mean']:.3f} [{r10['lo']:.3f}, {r10['hi']:.3f}]"
            )
    print(f"wrote {a.out}")
    return 0


def cmd_query(a: argparse.Namespace) -> int:
    docs = load_corpus(a.corpus)
    emb = _embedder(a.model, 32, [a.method])
    retriever = Retriever([d.id for d in docs], [d.full for d in docs], emb)
    hits = retriever.search(a.text, a.method, k=a.k)
    answer = extract_answer(a.text, hits[0], emb) if a.answer and hits else None
    if a.json:
        payload = {
            "query": a.text,
            "method": a.method,
            "hits": [{"rank": h.rank, "id": h.doc_id, "text": h.text} for h in hits],
            "answer": answer,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    for h in hits:
        print(f"{h.rank}. [{h.doc_id}] {h.text.splitlines()[0]}")
    if answer is not None:
        print(f"\nanswer: {answer}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "eval":
            return cmd_eval(a)
        if a.cmd == "query":
            return cmd_query(a)
        if a.cmd == "report":
            update_readme(a.readme, a.results)
            print(f"updated {a.readme} from {a.results}")
            return 0
        for lang in _csv(a.langs, LANGUAGES):  # download
            print(download(lang, a.data or data_dir()))  # pragma: no cover - network
        return 0  # pragma: no cover
    except (argparse.ArgumentTypeError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
