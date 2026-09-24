from __future__ import annotations

import json
from pathlib import Path

import pytest

# Tiny parallel Belebele-shaped fixture: 4 passages, 6 questions, 2 languages.
PASSAGES = {
    "eng_Latn": [
        "The accordion gets extra volume from the bellows, not from hitting keys harder.",
        "Tokyo is the capital of Japan and has a very large metro system.",
        "Penguins live in the southern hemisphere and cannot fly.",
        "Coffee beans are roasted seeds of the coffea plant.",
    ],
    "fra_Latn": [
        "L'accordéon obtient plus de volume grâce au soufflet, pas en frappant les touches.",
        "Tokyo est la capitale du Japon et possède un très grand métro.",
        "Les manchots vivent dans l'hémisphère sud et ne peuvent pas voler.",
        "Les grains de café sont les graines torréfiées du caféier.",
    ],
}
QUESTIONS = {
    "eng_Latn": [
        (0, "How does the accordion get extra volume?"),
        (0, "What should you not do with accordion keys?"),
        (1, "What is the capital of Japan?"),
        (2, "Where do penguins live?"),
        (2, "Can penguins fly?"),
        (3, "What are coffee beans?"),
    ],
    "fra_Latn": [
        (0, "Comment l'accordéon obtient-il plus de volume ?"),
        (0, "Que ne faut-il pas faire avec les touches de l'accordéon ?"),
        (1, "Quelle est la capitale du Japon ?"),
        (2, "Où vivent les manchots ?"),
        (2, "Les manchots peuvent-ils voler ?"),
        (3, "Que sont les grains de café ?"),
    ],
}


def write_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for lang, qs in QUESTIONS.items():
        with (root / f"{lang}.jsonl").open("w", encoding="utf-8") as fh:
            for n, (pid, q) in enumerate(qs):
                row = {
                    "link": f"https://example.org/{pid}",
                    "question_number": n,
                    "flores_passage": PASSAGES[lang][pid],
                    "question": q,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n\n")
    return root


@pytest.fixture
def belebele_dir(tmp_path: Path) -> Path:
    return write_fixture(tmp_path / "belebele")
