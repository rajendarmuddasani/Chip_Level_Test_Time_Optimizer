"""Confirm the frozen bundle on new lots inside its development envelope."""

from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.artifacts import file_sha256, write_json
from benchmark.policy_metrics import (
    SafetyGates,
    evaluate_by_failure_mode,
    evaluate_policy,
    gate_candidate,
)
from benchmark.synthetic_data import (
    BASE_SEED,
    FEATURE_NAMES,
    SplitSpec,
    build_dataset_manifest,
    generate_split,
)
from deployment.runtime import HybridPolicyRuntime


CONFIRMATION_SPEC = SplitSpec(
    "operational_envelope_confirmation",
    first_time_index=220,
    lot_count=20,
    chips_per_lot=500,
    fail_rate=0.06,
    phase_drift=0.28,
)


def main() -> None:
    manifest_path = (
        REPOSITORY_ROOT / "artifacts" / "public_v1" / "runtime_manifest.json"
    )
    evaluation_path = (
        REPOSITORY_ROOT / "evidence" / "public_synthetic_evaluation.json"
    )
    output_path = (
        REPOSITORY_ROOT / "evidence" / "operational_envelope_confirmation.json"
    )
    runtime = HybridPolicyRuntime(manifest_path)
    base_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    frame = generate_split(CONFIRMATION_SPEC, base_seed=BASE_SEED)
    decisions = runtime.predict_dataframe(frame[["chip_id", "lot_id", *FEATURE_NAMES]])
    labels = frame["label"].to_numpy(dtype=np.int64)
    flags = decisions["flag"].to_numpy(dtype=np.int64)
    metrics = evaluate_policy(labels, flags)
    gates = SafetyGates(**base_evaluation["selection_protocol"]["safety_gates"])
    gate_result = gate_candidate(metrics, gates)

    canonical_predictions = decisions[
        [
            "chip_id",
            "lot_id",
            "flag",
            "classifier_flag",
            "vae_flag",
            "sigma_flag",
            "lot_drift_blocked",
        ]
    ].copy()
    canonical_predictions.insert(2, "label", labels)
    prediction_bytes = canonical_predictions.to_csv(
        index=False,
        lineterminator="\n",
    ).encode("utf-8")
    lot_summary = {}
    for lot_id, lot in decisions.groupby("lot_id", sort=True):
        lot_summary[str(lot_id)] = {
            "chips": int(len(lot)),
            "drift_score": float(lot["lot_drift_score"].iloc[0]),
            "drift_blocked": bool(lot["lot_drift_blocked"].iloc[0]),
            "drift_reason": str(lot["lot_drift_reason"].iloc[0]),
        }

    payload = {
        "schema_version": 1,
        "confirmation_id": "post_freeze_operational_envelope_v1",
        "purpose": (
            "Measure the exact frozen bundle on new chronological lots inside the "
            "declared development drift envelope; no model or threshold selection occurs"
        ),
        "bundle_id": runtime.manifest["bundle_id"],
        "runtime_manifest_sha256": file_sha256(manifest_path),
        "base_evaluation_sha256": file_sha256(evaluation_path),
        "selection_or_tuning_performed": False,
        "dataset": build_dataset_manifest(
            {CONFIRMATION_SPEC.name: frame},
            [CONFIRMATION_SPEC],
            base_seed=BASE_SEED,
        ),
        "metrics": metrics,
        "gate_result": gate_result,
        "failure_modes": evaluate_by_failure_mode(
            frame["failure_mode"], labels, flags
        ),
        "lot_drift": {
            "blocked_lots": int(
                decisions.groupby("lot_id")["lot_drift_blocked"].first().sum()
            ),
            "total_lots": int(frame["lot_id"].nunique()),
            "lots": lot_summary,
        },
        "canonical_prediction_sha256": sha256(prediction_bytes).hexdigest(),
        "command": "python scripts/run_operational_envelope_confirmation.py",
        "targets": {
            "test_time_reduction_percent": 15.0,
            "observed_escapees": 0,
        },
    }
    example_lot = frame["lot_id"].iloc[0]
    example_records = [
        {
            "chip_id": str(row["chip_id"]),
            "lot_id": str(row["lot_id"]),
            **{name: float(row[name]) for name in FEATURE_NAMES},
        }
        for _, row in frame.loc[frame["lot_id"] == example_lot].iterrows()
    ]
    write_json(
        REPOSITORY_ROOT / "examples" / "public_synthetic_input.json",
        {
            "schema_version": 1,
            "note": "One unlabelled 500-chip lot from the accepted operational envelope",
            "records": example_records,
        },
    )
    write_json(output_path, payload)
    print(f"Bundle: {payload['bundle_id']}")
    print(f"Confirmation chips: {metrics['total_chips']:,}")
    print(f"Observed escapees: {metrics['escapees']}")
    print(f"Overtest rate: {metrics['overtest_rate']:.2%}")
    print(f"Test-time reduction: {metrics['time_reduction_percent']:.2f}%")
    print(f"Drift-blocked lots: {payload['lot_drift']['blocked_lots']}/20")
    print(f"Accepted: {gate_result['passed']}")


if __name__ == "__main__":
    main()