"""Runtime integrity, fail-closed behavior, and exact evidence replay."""

import json
from pathlib import Path

import numpy as np
import pytest

from benchmark.policy_metrics import evaluate_policy
from benchmark.synthetic_data import BASE_SEED, FEATURE_NAMES, SplitSpec, generate_split
from deployment.runtime import HybridPolicyRuntime, InputValidationError


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "public_v1" / "runtime_manifest.json"


@pytest.fixture(scope="module")
def runtime():
    return HybridPolicyRuntime(MANIFEST)


def test_runtime_loads_hash_verified_bundle(runtime):
    assert (
        runtime.manifest["bundle_id"]
        == "53ce0e9ccbd63b3c84c581a0dedc325782e8c09b72847977626f41aa6ad3d1fe"
    )
    assert len(runtime.feature_names) == 32


def test_runtime_replays_operational_confirmation(runtime):
    spec = SplitSpec(
        "operational_envelope_confirmation",
        220,
        20,
        500,
        0.06,
        0.28,
    )
    frame = generate_split(spec, base_seed=BASE_SEED)
    result = runtime.predict_dataframe(frame[["chip_id", "lot_id", *FEATURE_NAMES]])
    metrics = evaluate_policy(
        frame["label"].to_numpy(dtype=np.int64),
        result["flag"].to_numpy(dtype=np.int64),
    )
    canonical = json.loads(
        (ROOT / "evidence" / "operational_envelope_confirmation.json").read_text(
            encoding="utf-8"
        )
    )

    for key, expected in canonical["metrics"].items():
        if isinstance(expected, float):
            assert metrics[key] == pytest.approx(expected, rel=1e-12, abs=1e-15)
        elif isinstance(expected, list):
            assert metrics[key] == pytest.approx(expected, rel=1e-12, abs=1e-15)
        else:
            assert metrics[key] == expected
    assert result["lot_drift_blocked"].sum() == 0


def test_runtime_without_lot_context_forces_full_testing(runtime):
    payload = json.loads(
        (ROOT / "examples" / "public_synthetic_input.json").read_text(
            encoding="utf-8"
        )
    )
    records = [
        {key: value for key, value in row.items() if key != "lot_id"}
        for row in payload["records"][:5]
    ]

    result = runtime.predict_records(records)

    assert {record["decision"] for record in result} == {"RUN"}
    assert {record["lot_drift_reason"] for record in result} == {
        "insufficient_lot_context"
    }


def test_runtime_rejects_missing_feature(runtime):
    payload = json.loads(
        (ROOT / "examples" / "public_synthetic_input.json").read_text(
            encoding="utf-8"
        )
    )
    record = dict(payload["records"][0])
    record.pop(FEATURE_NAMES[0])

    with pytest.raises(InputValidationError, match="Missing features"):
        runtime.predict_records([record])