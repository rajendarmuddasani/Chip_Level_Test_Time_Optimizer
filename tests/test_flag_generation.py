"""Working CLI boundary for the frozen public policy runtime."""

import json
from pathlib import Path

import pandas as pd
import pytest

from deployment.generate_flags import FlagGenerator
from deployment.runtime import HybridPolicyRuntime


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "public_v1" / "runtime_manifest.json"
INPUT = ROOT / "examples" / "public_synthetic_input.json"


@pytest.fixture(scope="module")
def generator():
    return FlagGenerator(runtime=HybridPolicyRuntime(MANIFEST))


@pytest.fixture(scope="module")
def public_frame():
    return pd.DataFrame.from_records(
        json.loads(INPUT.read_text(encoding="utf-8"))["records"]
    )


def test_flag_generator_replays_public_fixture(generator, public_frame):
    predictions = generator.generate_flags(public_frame)
    summary = generator.summary(predictions)

    assert summary["total_chips"] == 500
    assert summary["skip_count"] == 450
    assert summary["run_count"] == 50
    assert summary["simulated_time_reduction_percent"] == pytest.approx(13.5)
    assert summary["bundle_id"] == generator.runtime.manifest["bundle_id"]


def test_flag_generator_accepts_explicit_identifier_columns(generator, public_frame):
    renamed = public_frame.rename(
        columns={"chip_id": "CHIP_ID", "lot_id": "LOT_ID"}
    )

    predictions = generator.generate_flags(
        renamed,
        chip_id_col="CHIP_ID",
        lot_id_col="LOT_ID",
    )

    assert predictions["chip_id"].tolist() == public_frame["chip_id"].tolist()
    assert predictions["lot_id"].tolist() == public_frame["lot_id"].tolist()


def test_missing_lot_context_runs_every_chip_and_writes_json(
    generator,
    public_frame,
    tmp_path,
):
    predictions = generator.generate_flags(public_frame.head(5).drop(columns="lot_id"))
    output = tmp_path / "predictions.json"

    generator.save_predictions(predictions, output, "json")
    saved = json.loads(output.read_text(encoding="utf-8"))["records"]

    assert set(predictions["decision"]) == {"RUN"}
    assert set(predictions["lot_drift_reason"]) == {"insufficient_lot_context"}
    assert {record["lot_drift_score"] for record in saved} == {None}


def test_sortfile_contains_binary_decisions(generator, public_frame, tmp_path):
    predictions = generator.generate_flags(public_frame.head(5).drop(columns="lot_id"))
    output = tmp_path / "flags.txt"

    generator.save_predictions(predictions, output, "sortfile")

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines == [f"{chip_id} 1" for chip_id in public_frame["chip_id"].head(5)]