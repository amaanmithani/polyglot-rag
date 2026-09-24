from __future__ import annotations

import json
from pathlib import Path

import pytest

from polyglot_rag import cli
from polyglot_rag.data import Row, build_set, load_rows, passage_keys, read_rows, sample_indices
from polyglot_rag.embed import HashingEmbedder
from polyglot_rag.evaluate import run_eval
from polyglot_rag.report import END, START, render, update_readme
from polyglot_rag.retriever import Hit
from polyglot_rag.support import extract_answer, load_corpus, split_sentences


def test_rows_and_sets(belebele_dir: Path) -> None:
    eng = read_rows(belebele_dir / "eng_Latn.jsonl")
    fra = load_rows("fra_Latn", belebele_dir, fetch=False)
    assert len(eng) == 6
    assert passage_keys(eng) == [0, 0, 1, 2, 2, 3]
    s = build_set("fra_Latn", fra, eng)
    assert len(s.docs) == 4 and s.doc_ids[0] == "p000"
    assert s.docs[1].startswith("Tokyo est")
    with pytest.raises(ValueError, match="rows"):
        build_set("fra_Latn", fra[:3], eng)
    shuffled = build_set("fra_Latn", list(reversed(fra)), eng)
    assert shuffled.questions == s.questions and shuffled.relevant == s.relevant
    bad = [*fra[:5], Row(link="https://other", number=99, passage="x", question="y")]
    with pytest.raises(ValueError, match="aligned"):
        build_set("fra_Latn", bad, eng)
    with pytest.raises(FileNotFoundError):
        load_rows("deu_Latn", belebele_dir, fetch=False)


def test_sample_indices_fixed_seed() -> None:
    a = sample_indices(900, 50, seed=13)
    assert a == sample_indices(900, 50, seed=13)
    assert a != sample_indices(900, 50, seed=14)
    assert len(set(a)) == 50 and a == sorted(a)
    assert sample_indices(5, 10, seed=0) == [0, 1, 2, 3, 4]


def test_run_eval_end_to_end(belebele_dir: Path) -> None:
    logs: list[str] = []
    res = run_eval(
        ["eng_Latn", "fra_Latn"],
        n=6,
        seed=0,
        methods=["bm25", "dense", "hybrid"],
        modes=["mono", "cross"],
        embedder=HashingEmbedder(),
        n_boot=200,
        root=belebele_dir,
        fetch=False,
        log=logs.append,
    )
    mono = res["monolingual"]
    assert set(mono) == {"eng_Latn", "fra_Latn"}
    assert set(res["crosslingual"]) == {"fra_Latn"}
    r = mono["eng_Latn"]
    assert r["n_queries"] == 6 and r["n_docs"] == 4
    assert r["methods"]["bm25"]["recall@10"]["mean"] == 1.0  # only 4 docs
    assert "hybrid-dense" in r["deltas"]
    assert set(res["vs_english"]["monolingual"]) == {"fra_Latn"}
    assert res["meta"]["n_per_language"] == 6
    assert res["macro"]["monolingual"]["bm25"]["mrr@10"]["mean"] > 0
    assert logs

    block = render(res, "results/x.json")
    assert block.startswith(START) and block.endswith(END)
    assert "| French | `fra_Latn` | 6 |" in block
    assert "Macro avg" in block and "Findings" in block


def test_run_eval_rejects_bad_mode(belebele_dir: Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        run_eval(
            ["eng_Latn"],
            n=2,
            seed=0,
            methods=["bm25"],
            modes=["zz"],
            embedder=None,
            root=belebele_dir,
            fetch=False,
        )


def test_run_eval_bm25_only_and_cross_only(belebele_dir: Path) -> None:
    res = run_eval(
        ["fra_Latn"],
        n=4,
        seed=0,
        methods=["bm25"],
        modes=["cross"],
        embedder=None,
        n_boot=50,
        root=belebele_dir,
        fetch=False,
    )
    assert res["monolingual"] == {} and "fra_Latn" in res["crosslingual"]
    assert res["vs_english"]["monolingual"] == {}
    block = render(res, "r.json")
    assert "_(not run)_" in block


def test_update_readme(tmp_path: Path, belebele_dir: Path) -> None:
    res = run_eval(
        ["eng_Latn", "fra_Latn"],
        n=6,
        seed=0,
        methods=["bm25", "hybrid"],
        modes=["mono"],
        embedder=HashingEmbedder(),
        n_boot=50,
        root=belebele_dir,
        fetch=False,
    )
    results = tmp_path / "r.json"
    results.write_text(json.dumps(res))
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nOLDCONTENT\n{END}\noutro\n")
    update_readme(readme, results)
    text = readme.read_text()
    assert "OLDCONTENT" not in text and text.startswith("intro") and text.endswith("outro\n")
    readme.write_text("no markers")
    with pytest.raises(ValueError):
        update_readme(readme, results)


def test_support_corpus_and_answers(tmp_path: Path) -> None:
    docs = load_corpus()
    assert len(docs) >= 20 and docs[0].full.startswith(docs[0].title)
    p = tmp_path / "c.jsonl"
    p.write_text('{"id": 1, "text": "Only text."}\n\n')
    (d,) = load_corpus(p)
    assert d.id == "1" and d.full == "Only text."
    p.write_text("\n")
    with pytest.raises(ValueError):
        load_corpus(p)
    assert split_sentences("A b. C d? E!") == ["A b.", "C d?", "E!"]
    hit = Hit(0, "x", 1, "Refunds take 14 days. Shipping is free. Call us anytime.")
    assert extract_answer("is shipping free", hit, None) == "Shipping is free."
    assert extract_answer("shipping free", hit, HashingEmbedder()) == "Shipping is free."
    assert extract_answer("q", Hit(0, "x", 1, "one"), None) == "one"
    titled = Hit(0, "x", 1, "Refund policy?\nRefunds take 14 days. Shipping is free.")
    assert extract_answer("refund policy", titled, None) == "Refunds take 14 days."


def test_cli_eval_report_query(
    belebele_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "res.json"
    rc = cli.main(
        [
            "eval",
            "--langs",
            "eng_Latn,fra_Latn",
            "--n",
            "6",
            "--model",
            "hashing",
            "--boot",
            "50",
            "--data",
            str(belebele_dir),
            "--no-fetch",
            "--out",
            str(out),
        ]
    )
    assert rc == 0 and json.loads(out.read_text())["meta"]["embedder"].startswith("hashing")
    assert "macro R@10" in capsys.readouterr().out

    readme = tmp_path / "README.md"
    readme.write_text(f"{START}\n{END}\n")
    assert cli.main(["report", "--results", str(out), "--readme", str(readme)]) == 0
    assert "English" in readme.read_text()
    capsys.readouterr()

    assert cli.main(["query", "How can I get my money back?", "--method", "bm25", "--answer"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("1. [") and "answer:" in text

    assert cli.main(["query", "reset password", "--model", "hashing", "--json", "--answer"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hits"][0]["id"] == "reset-password" and payload["answer"]


def test_cli_errors(capsys: pytest.CaptureFixture[str], belebele_dir: Path) -> None:
    assert cli.main(["eval", "--langs", "xx_Yyyy"]) == 2
    assert "unknown value" in capsys.readouterr().err
    assert (
        cli.main(
            [
                "eval",
                "--langs",
                "deu_Latn",
                "--methods",
                "bm25",
                "--modes",
                "mono",
                "--data",
                str(belebele_dir),
                "--no-fetch",
            ]
        )
        == 2
    )
