"""Write hash-bound model, evidence, example, and visual artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.experiment import BenchmarkRun
from benchmark.synthetic_data import FEATURE_NAMES, build_dataset_manifest


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_base_sha(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _prediction_evidence(
    frame: pd.DataFrame,
    outputs: Mapping[str, np.ndarray],
) -> dict:
    decisions = frame[
        ["chip_id", "lot_id", "time_index", "failure_mode", "label"]
    ].copy()
    for name, values in outputs.items():
        decisions[name] = values
    canonical = decisions.to_csv(
        index=False,
        float_format="%.8f",
        lineterminator="\n",
    ).encode("utf-8")

    per_lot = []
    for lot_id, lot in decisions.groupby("lot_id", sort=True):
        labels = lot["label"].to_numpy(dtype=np.int64)
        flags = lot["ensemble_flags"].to_numpy(dtype=np.int64)
        per_lot.append(
            {
                "lot_id": str(lot_id),
                "time_index": int(lot["time_index"].iloc[0]),
                "chips": int(len(lot)),
                "failures": int(labels.sum()),
                "skipped": int((flags == 0).sum()),
                "escapees": int(((labels == 1) & (flags == 0)).sum()),
                "overtest": int(((labels == 0) & (flags == 1)).sum()),
            }
        )
    return {
        "row_count": int(len(decisions)),
        "canonical_prediction_sha256": sha256(canonical).hexdigest(),
        "per_lot": per_lot,
    }


def _write_confusion_matrix(
    labels: np.ndarray,
    flags: np.ndarray,
    output_path: Path,
) -> None:
    matrix = np.array(
        [
            [
                int(((labels == 0) & (flags == 0)).sum()),
                int(((labels == 0) & (flags == 1)).sum()),
            ],
            [
                int(((labels == 1) & (flags == 0)).sum()),
                int(((labels == 1) & (flags == 1)).sum()),
            ],
        ]
    )
    fig, axis = plt.subplots(figsize=(6.4, 4.6))
    image = axis.imshow(matrix, cmap="YlGnBu")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "#142b3a",
                fontsize=17,
                fontweight="bold",
            )
    axis.set_xticks([0, 1], labels=["SKIP optional test", "RUN optional test"])
    axis.set_yticks([0, 1], labels=["Actual pass", "Actual fail"])
    axis.set_xlabel("Frozen policy decision")
    axis.set_ylabel("Synthetic confirmation label")
    axis.set_title("Confirmation policy matrix")
    fig.colorbar(image, ax=axis, fraction=0.045)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _write_candidate_tradeoff(report: dict, output_path: Path) -> None:
    candidates = [
        candidate
        for candidate in report["validation"]["candidates"]
        if candidate["eligible_for_selection"]
    ]
    selected_name = report["validation"]["selected_candidate"]["name"]
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for candidate in candidates:
        passed = candidate["gate_result"]["passed"]
        selected = candidate["name"] == selected_name
        axis.scatter(
            candidate["metrics"]["time_reduction_percent"],
            candidate["metrics"]["overtest_rate"] * 100.0,
            color="#f3c969" if selected else ("#18745a" if passed else "#e56b46"),
            edgecolor="#142b3a" if selected else "none",
            linewidth=1.5,
            s=95 if selected else 38,
            alpha=0.88,
            zorder=3 if selected else 2,
        )
    axis.axhline(
        report["selection_protocol"]["safety_gates"]["maximum_overtest_rate"]
        * 100.0,
        color="#9381e8",
        linestyle="--",
        label="Overtest gate",
    )
    axis.set_xlabel("Validation test-time reduction (%)")
    axis.set_ylabel("Validation overtest rate (%)")
    axis.set_title("Bounded hybrid policy search")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _bundle_identity(runtime_config: dict, classifier_hash: str, vae_hash: str) -> str:
    payload = json.dumps(
        {
            "runtime_config": _json_ready(runtime_config),
            "classifier_sha256": classifier_hash,
            "vae_sha256": vae_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def save_benchmark_run(
    run: BenchmarkRun,
    splits: Mapping[str, pd.DataFrame],
    repository_root: Path,
    command: str,
) -> dict:
    """Persist the exact selected bundle and its machine-readable evidence."""
    repository_root = repository_root.resolve()
    evidence_dir = repository_root / "evidence"
    artifact_dir = repository_root / "artifacts" / "public_v1"
    asset_dir = repository_root / "docs" / "assets"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    classifier_path = artifact_dir / "classifier_state.pt"
    vae_path = artifact_dir / "vae_state.pt"
    torch.save(_copy_to_cpu(run.classifier.state_dict()), classifier_path)
    torch.save(_copy_to_cpu(run.vae.state_dict()), vae_path)
    classifier_hash = file_sha256(classifier_path)
    vae_hash = file_sha256(vae_path)
    bundle_id = _bundle_identity(run.runtime_config, classifier_hash, vae_hash)

    dataset_manifest_path = evidence_dir / "public_synthetic_dataset_manifest.json"
    dataset_manifest = build_dataset_manifest(splits)
    write_json(dataset_manifest_path, dataset_manifest)

    report: dict[str, Any] = _json_ready(run.report)
    report["bundle_id"] = bundle_id
    report["dataset_manifest"] = dataset_manifest_path.relative_to(
        repository_root
    ).as_posix()
    report["dataset_manifest_sha256"] = file_sha256(dataset_manifest_path)
    report["artifact_hashes"] = {
        "classifier_state.pt": classifier_hash,
        "vae_state.pt": vae_hash,
    }
    report["prediction_evidence"] = {
        split_name: _prediction_evidence(splits[split_name], outputs)
        for split_name, outputs in run.predictions.items()
    }
    report["reproducibility"] = {
        "base_commit_sha": _git_base_sha(repository_root),
        "command": command,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "not reported by operating system",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "source_hashes": {
            path.name: file_sha256(path)
            for path in (
                Path(__file__).with_name("synthetic_data.py"),
                Path(__file__).with_name("policy_metrics.py"),
                Path(__file__).with_name("experiment.py"),
            )
        },
    }
    evaluation_path = evidence_dir / "public_synthetic_evaluation.json"
    write_json(evaluation_path, report)

    runtime_manifest = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "model_id": run.runtime_config["model_id"],
        "runtime_config": run.runtime_config,
        "artifacts": {
            "classifier": {
                "path": classifier_path.name,
                "sha256": classifier_hash,
            },
            "vae": {
                "path": vae_path.name,
                "sha256": vae_hash,
            },
        },
        "evidence": {
            "evaluation_path": "../../evidence/public_synthetic_evaluation.json",
            "evaluation_sha256": file_sha256(evaluation_path),
            "dataset_manifest_path": "../../evidence/public_synthetic_dataset_manifest.json",
            "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        },
    }
    runtime_manifest_path = artifact_dir / "runtime_manifest.json"
    write_json(runtime_manifest_path, runtime_manifest)

    example_records = []
    example_lot = splits["confirmation"]["lot_id"].iloc[0]
    for _, row in splits["confirmation"].loc[
        splits["confirmation"]["lot_id"] == example_lot
    ].iterrows():
        example_records.append(
            {
                "chip_id": str(row["chip_id"]),
                "lot_id": str(row["lot_id"]),
                **{name: float(row[name]) for name in FEATURE_NAMES},
            }
        )
    example_path = repository_root / "examples" / "public_synthetic_input.json"
    write_json(
        example_path,
        {
            "schema_version": 1,
            "note": "One unlabelled 500-chip lot generated independently by the public benchmark",
            "records": example_records,
        },
    )

    confirmation_labels = splits["confirmation"]["label"].to_numpy(dtype=np.int64)
    _write_confusion_matrix(
        confirmation_labels,
        run.predictions["confirmation"]["ensemble_flags"],
        asset_dir / "confirmation_policy_matrix.png",
    )
    _write_candidate_tradeoff(report, asset_dir / "candidate_tradeoff.png")

    return {
        "bundle_id": bundle_id,
        "runtime_manifest": runtime_manifest_path,
        "evaluation": evaluation_path,
        "dataset_manifest": dataset_manifest_path,
        "example_input": example_path,
        "assets": [
            asset_dir / "confirmation_policy_matrix.png",
            asset_dir / "candidate_tradeoff.png",
        ],
    }


def _copy_to_cpu(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state_dict.items()}