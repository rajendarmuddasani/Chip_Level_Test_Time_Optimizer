"""Public documentation stays aligned with canonical evidence."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs" / "DATA_CARD.md",
    ROOT / "docs" / "MODEL_CARD.md",
    ROOT / "docs" / "DEPLOYMENT.md",
    ROOT / "evidence" / "PDF_REPOSITORY_AUDIT.md",
    ROOT / "evidence" / "METRIC_IMPROVEMENT_PLAN.md",
)


def test_documentation_contains_accepted_numbers_and_bundle():
    combined = "\n".join(path.read_text(encoding="utf-8") for path in DOCUMENTS)

    for value in (
        "13.524%",
        "6/600",
        "4.149%",
        "98.571%",
        "10.22 requests/s",
        "470.50/565.37/721.73 ms",
        "53ce0e9ccbd63b3c84c581a0dedc325782e8c09b72847977626f41aa6ad3d1fe",
    ):
        assert value in combined


def test_documentation_does_not_claim_zero_escape_guarantee():
    combined = "\n".join(path.read_text(encoding="utf-8") for path in DOCUMENTS)

    assert "guarantees zero escapees" not in combined.lower()
    assert "zero observed escapes is an unmet objective" in combined.lower()


def test_local_markdown_links_exist():
    missing = []
    for document in DOCUMENTS:
        content = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)", content):
            if target.startswith(("http", "#")):
                continue
            local_target = (document.parent / target).resolve()
            root_target = (ROOT / target).resolve()
            if not local_target.exists() and not root_target.exists():
                missing.append((document.relative_to(ROOT).as_posix(), target))

    assert not missing