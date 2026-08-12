"""Executed public evidence notebook stays truthful and complete."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_public_evidence.ipynb"


def test_public_evidence_notebook_is_executed_without_errors():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]

    assert len(code_cells) == 5
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert all(cell.get("outputs") for cell in code_cells)
    assert not [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]


def test_public_evidence_notebook_contains_canonical_result_and_boundary():
    content = NOTEBOOK.read_text(encoding="utf-8")

    for value in (
        "13.524%",
        "6 / 600",
        "4.149%",
        "98.571%",
        "d5d50071d10de5bf0dfb531b843c49567a42cb170b595cc1b990f4c2a30661db",
        "zero-observed-escape objectives remain unmet",
    ):
        assert value in content

    for stale_claim in (
        "3.2M annual savings",
        "0% escapee rate (no bad chips marked as skip)",
        "simulate predictions",
        "202 features",
    ):
        assert stale_claim not in content