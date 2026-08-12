"""Validate Project 05 evidence identities and replay the frozen policy."""

from __future__ import annotations

import argparse
import json
import math
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.artifacts import file_sha256  # noqa: E402
from benchmark.policy_metrics import evaluate_policy  # noqa: E402
from benchmark.synthetic_data import (  # noqa: E402
    BASE_SEED,
    FEATURE_NAMES,
    SplitSpec,
    dataframe_sha256,
    generate_benchmark,
    generate_split,
)
from deployment.runtime import HybridPolicyRuntime  # noqa: E402


OPERATIONAL_SPEC = SplitSpec(
    "operational_envelope_confirmation",
    first_time_index=220,
    lot_count=20,
    chips_per_lot=500,
    fail_rate=0.06,
    phase_drift=0.28,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _matches(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _matches(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def _runtime_config_differences(left: Any, right: Any, path: str = "") -> list[dict]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            next_path = f"{path}/{key}"
            if key not in left or key not in right:
                differences.append({"path": next_path, "canonical": left.get(key), "replay": right.get(key)})
            else:
                differences.extend(
                    _runtime_config_differences(left[key], right[key], next_path)
                )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [{"path": path, "canonical": left, "replay": right}]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                _runtime_config_differences(
                    left_item,
                    right_item,
                    f"{path}/{index}",
                )
            )
        return differences
    return [] if left == right else [{"path": path, "canonical": left, "replay": right}]


def _summarize_runtime_differences(differences: list[dict]) -> dict:
    numeric_differences = [
        abs(float(item["canonical"]) - float(item["replay"]))
        for item in differences
        if isinstance(item["canonical"], (int, float))
        and isinstance(item["replay"], (int, float))
    ]
    return {
        "count": len(differences),
        "numeric_count": len(numeric_differences),
        "maximum_absolute_numeric_difference": (
            max(numeric_differences) if numeric_differences else None
        ),
        "sample": differences[:20],
    }


def validate(repository_root: Path = REPOSITORY_ROOT, replay_root: Path | None = None) -> dict:
    root = repository_root.resolve()
    errors = []
    public = _read_json(root / "evidence" / "public_synthetic_evaluation.json")
    dataset_manifest = _read_json(
        root / "evidence" / "public_synthetic_dataset_manifest.json"
    )
    operational = _read_json(
        root / "evidence" / "operational_envelope_confirmation.json"
    )
    claims = _read_json(root / "evidence" / "claims.json")
    runtime_manifest_path = root / "artifacts" / "public_v1" / "runtime_manifest.json"
    runtime = HybridPolicyRuntime(runtime_manifest_path)

    splits = generate_benchmark(base_seed=dataset_manifest["base_seed"])
    split_checks = {}
    for name, frame in splits.items():
        actual_hash = dataframe_sha256(frame)
        expected_hash = dataset_manifest["splits"][name]["sha256"]
        split_checks[name] = actual_hash == expected_hash
        if actual_hash != expected_hash:
            errors.append(f"Dataset split hash mismatch: {name}")

    operational_frame = generate_split(OPERATIONAL_SPEC, base_seed=BASE_SEED)
    predictions = runtime.predict_dataframe(
        operational_frame[["chip_id", "lot_id", *FEATURE_NAMES]]
    )
    labels = operational_frame["label"].to_numpy(dtype=np.int64)
    flags = predictions["flag"].to_numpy(dtype=np.int64)
    replayed_metrics = evaluate_policy(labels, flags)
    if not _matches(replayed_metrics, operational["metrics"]):
        errors.append("Operational metrics do not replay exactly")

    canonical_predictions = predictions[
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
    prediction_hash = sha256(
        canonical_predictions.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    if prediction_hash != operational["canonical_prediction_sha256"]:
        errors.append("Operational prediction hash mismatch")

    claim_by_id = {claim["id"]: claim for claim in claims["claims"]}
    claim_checks = {
        "bundle_id": (
            claims["champion"]["bundle_id"] == runtime.manifest["bundle_id"]
        ),
        "operational_metrics": _matches(
            claim_by_id["post_freeze_operational_confirmation"]["value"],
            {
                "samples": operational["metrics"]["total_chips"],
                "lots": operational["dataset"]["splits"][OPERATIONAL_SPEC.name]["lot_count"],
                "pass_chips": operational["metrics"]["pass_chips"],
                "failed_chips": operational["metrics"]["failed_chips"],
                "time_reduction_percent": operational["metrics"]["time_reduction_percent"],
                "time_reduction_ci_95_percent": operational["metrics"]["time_reduction_ci_95_percent"],
                "defect_recall": operational["metrics"]["defect_recall"],
                "escapees": operational["metrics"]["escapees"],
                "relative_escape_rate": operational["metrics"]["relative_escape_rate"],
                "absolute_escape_rate": operational["metrics"]["absolute_escape_rate"],
                "escape_rate_upper_95": operational["metrics"]["escape_rate_upper_95"],
                "overtest": operational["metrics"]["overtest"],
                "overtest_rate": operational["metrics"]["overtest_rate"],
                "mcc": operational["metrics"]["mcc"],
            },
        ),
        "selected_candidate": (
            claim_by_id["bounded_selection"]["value"]["selected_candidate"]
            == public["validation"]["selected_candidate"]["name"]
        ),
        "targets_unmet": all(
            target["status"] == "unmet" for target in claims["targets"]
        ),
    }
    for name, passed in claim_checks.items():
        if not passed:
            errors.append(f"Claim contract failed: {name}")

    source_paths = {
        "synthetic_data.py": root / "benchmark" / "synthetic_data.py",
        "policy_metrics.py": root / "benchmark" / "policy_metrics.py",
        "experiment.py": root / "benchmark" / "experiment.py",
    }
    source_hash_checks = {
        name: file_sha256(path)
        == public["reproducibility"]["source_hashes"][name]
        for name, path in source_paths.items()
    }

    replay_comparison = None
    if replay_root is not None:
        replay_path = replay_root.resolve()
        replay_public = _read_json(
            replay_path / "evidence" / "public_synthetic_evaluation.json"
        )
        replay_manifest = _read_json(
            replay_path / "artifacts" / "public_v1" / "runtime_manifest.json"
        )
        canonical_manifest = runtime.manifest
        replay_checks = {
            "selected_candidate": (
                replay_public["validation"]["selected_candidate"]["name"]
                == public["validation"]["selected_candidate"]["name"]
            ),
            "selected_metrics": _matches(
                replay_public["validation"]["selected_candidate"]["metrics"],
                public["validation"]["selected_candidate"]["metrics"],
            ),
            "confirmation_prediction_hash": (
                replay_public["prediction_evidence"]["confirmation"]["canonical_prediction_sha256"]
                == public["prediction_evidence"]["confirmation"]["canonical_prediction_sha256"]
            ),
            "ood_prediction_hash": (
                replay_public["prediction_evidence"]["ood_confirmation"]["canonical_prediction_sha256"]
                == public["prediction_evidence"]["ood_confirmation"]["canonical_prediction_sha256"]
            ),
            "artifact_hashes": replay_public["artifact_hashes"] == public["artifact_hashes"],
        }
        for name, passed in replay_checks.items():
            if not passed:
                errors.append(f"Training replay contract failed: {name}")
        runtime_differences = _runtime_config_differences(
            canonical_manifest["runtime_config"],
            replay_manifest["runtime_config"],
        )
        replay_comparison = {
            "checks": replay_checks,
            "behavioral_identity_passed": all(replay_checks.values()),
            "canonical_bundle_id": canonical_manifest["bundle_id"],
            "replay_bundle_id": replay_manifest["bundle_id"],
            "bundle_id_matches": (
                canonical_manifest["bundle_id"] == replay_manifest["bundle_id"]
            ),
            "runtime_config_difference_summary": _summarize_runtime_differences(
                runtime_differences
            ),
            "interpretation": (
                "Model and prediction identities are acceptance checks. A different "
                "bundle ID is retained as a disclosed serialization difference when "
                "all behavioral replay checks pass."
            ),
            "canonical_environment": public["reproducibility"],
            "replay_environment": replay_public["reproducibility"],
        }

    return {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "bundle_id": runtime.manifest["bundle_id"],
        "runtime_manifest_sha256": file_sha256(runtime_manifest_path),
        "dataset_split_checks": split_checks,
        "operational_metrics_replayed": _matches(
            replayed_metrics, operational["metrics"]
        ),
        "operational_prediction_sha256": prediction_hash,
        "claim_checks": claim_checks,
        "historical_source_hash_matches": source_hash_checks,
        "historical_source_hash_interpretation": (
            "These hashes identify the source files recorded during frozen artifact "
            "generation. Current source includes recovered implementation and delivery "
            "changes; model, metric, and prediction replay are the active acceptance gates."
        ),
        "training_replay": replay_comparison,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = validate(replay_root=args.replay_root)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())