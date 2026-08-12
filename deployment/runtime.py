"""Fail-closed runtime for the hash-bound public hybrid policy bundle."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.anomaly_detection.vae_model import VariationalAutoEncoder
from models.classification.classifier_model import ChipTestClassifier


class ArtifactIntegrityError(RuntimeError):
    """Raised when a model or evidence file does not match its manifest."""


class InputValidationError(ValueError):
    """Raised when an inference request does not match the frozen feature schema."""


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_identity(runtime_config: dict, classifier_hash: str, vae_hash: str) -> str:
    payload = json.dumps(
        {
            "runtime_config": runtime_config,
            "classifier_sha256": classifier_hash,
            "vae_sha256": vae_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


class HybridPolicyRuntime:
    """Load, verify, and execute one selected classifier/VAE/sigma OR policy."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ArtifactIntegrityError("Unsupported runtime manifest schema")
        self.config = self.manifest["runtime_config"]
        self.feature_names = list(self.config["feature_names"])
        self._verify_bundle()
        self._load_models()

    def _verify_file(self, relative_path: str, expected_hash: str) -> Path:
        path = (self.manifest_path.parent / relative_path).resolve()
        if not path.is_file():
            raise ArtifactIntegrityError(f"Required artifact is missing: {path.name}")
        actual_hash = _file_sha256(path)
        if actual_hash != expected_hash:
            raise ArtifactIntegrityError(f"SHA-256 mismatch for {path.name}")
        return path

    def _verify_bundle(self) -> None:
        classifier = self.manifest["artifacts"]["classifier"]
        vae = self.manifest["artifacts"]["vae"]
        self.classifier_path = self._verify_file(
            classifier["path"], classifier["sha256"]
        )
        self.vae_path = self._verify_file(vae["path"], vae["sha256"])
        for evidence_name, hash_name in (
            ("evaluation_path", "evaluation_sha256"),
            ("dataset_manifest_path", "dataset_manifest_sha256"),
        ):
            self._verify_file(
                self.manifest["evidence"][evidence_name],
                self.manifest["evidence"][hash_name],
            )
        expected_bundle = _bundle_identity(
            self.config,
            classifier["sha256"],
            vae["sha256"],
        )
        if expected_bundle != self.manifest["bundle_id"]:
            raise ArtifactIntegrityError("Bundle identity does not match its components")

    def _load_models(self) -> None:
        classifier_config = self.config["classifier"]
        self.classifier = ChipTestClassifier(
            input_size=classifier_config["input_size"],
            hidden_size=classifier_config["hidden_size"],
            dropout=classifier_config["dropout"],
        )
        self.classifier.load_state_dict(
            torch.load(self.classifier_path, map_location="cpu", weights_only=True)
        )
        self.classifier.eval()

        vae_config = self.config["vae"]
        self.vae = VariationalAutoEncoder(
            input_size=vae_config["input_size"],
            latent_size=vae_config["latent_size"],
            num_layers=vae_config["num_layers"],
            beta=vae_config["beta"],
        )
        self.vae.load_state_dict(
            torch.load(self.vae_path, map_location="cpu", weights_only=True)
        )
        self.vae.eval()

    def _validate(
        self, frame: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, bool, np.ndarray]:
        if frame.empty:
            raise InputValidationError("At least one chip record is required")
        if len(frame) > 10_000:
            raise InputValidationError("A request may contain at most 10,000 chips")
        allowed = set(self.feature_names) | {"chip_id", "lot_id"}
        missing = sorted(set(self.feature_names) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - allowed)
        if missing:
            raise InputValidationError(f"Missing features: {missing}")
        if unexpected:
            raise InputValidationError(f"Unexpected fields: {unexpected}")

        if "chip_id" in frame:
            chip_ids = frame["chip_id"].astype(str)
            if chip_ids.duplicated().any():
                raise InputValidationError(
                    "chip_id values must be unique within a request"
                )
        else:
            chip_ids = pd.Series(
                [f"REQUEST_CHIP_{index:05d}" for index in range(len(frame))],
                index=frame.index,
            )
        has_lot_context = "lot_id" in frame
        if has_lot_context:
            lot_ids = frame["lot_id"].astype(str)
        else:
            lot_ids = pd.Series("MISSING_LOT_CONTEXT", index=frame.index)
        try:
            values = frame[self.feature_names].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise InputValidationError("Every feature must be numeric") from error
        if not np.isfinite(values).all():
            raise InputValidationError(
                "Features must be finite; NaN and infinity are rejected"
            )
        return chip_ids, lot_ids, has_lot_context, values

    def predict_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        chip_ids, lot_ids, has_lot_context, raw_values = self._validate(frame)
        centre = np.asarray(self.config["scaler"]["centre"], dtype=np.float64)
        scale = np.asarray(self.config["scaler"]["scale"], dtype=np.float64)
        if (scale <= 0).any():
            raise ArtifactIntegrityError("Scaler contains a non-positive scale")
        scaled_values = ((raw_values - centre) / scale).astype(np.float32)
        tensor = torch.from_numpy(scaled_values)

        classifier_probabilities = self.classifier.predict_proba(tensor)[:, 1].numpy()
        vae_errors = self.vae.get_reconstruction_error(tensor).numpy()
        classifier_flags = (
            classifier_probabilities >= self.config["classifier"]["threshold"]
        ).astype(np.int64)
        vae_flags = (vae_errors > self.config["vae"]["threshold"]).astype(np.int64)

        sigma_config = self.config["sigma"]
        feature_rules = sigma_config["feature_rules"]
        sigma_flags = np.zeros(len(frame), dtype=np.int64)
        robust_z_columns = []
        for feature_index, feature_name in enumerate(self.feature_names):
            parameters = feature_rules[feature_name]
            lower = parameters["median"] - parameters["lower"] * parameters["rstd"]
            upper = parameters["median"] + parameters["upper"] * parameters["rstd"]
            sigma_flags |= (
                (scaled_values[:, feature_index] < lower)
                | (scaled_values[:, feature_index] > upper)
            ).astype(np.int64)
            robust_scale = max(parameters["rstd"], np.finfo(float).eps)
            robust_z_columns.append(
                (scaled_values[:, feature_index] - parameters["median"])
                / robust_scale
            )
        robust_z = np.column_stack(robust_z_columns)
        aggregate_scores = np.sqrt(np.mean(np.square(robust_z), axis=1))
        sigma_flags |= (
            aggregate_scores > sigma_config["aggregate_threshold"]
        ).astype(np.int64)
        correlation_location = np.asarray(
            sigma_config["correlation_location"], dtype=np.float64
        )
        correlation_precision = np.asarray(
            sigma_config["correlation_precision"], dtype=np.float64
        )
        centred = robust_z - correlation_location
        correlation_scores = np.sqrt(
            np.maximum(
                np.einsum(
                    "ij,jk,ik->i",
                    centred,
                    correlation_precision,
                    centred,
                ),
                0.0,
            )
        )
        sigma_flags |= (
            correlation_scores > sigma_config["correlation_threshold"]
        ).astype(np.int64)
        lot_guard = self.config["lot_drift_guard"]
        lot_drift_scores = np.full(len(frame), np.nan, dtype=np.float64)
        lot_drift_flags = np.ones(len(frame), dtype=np.int64)
        lot_drift_reasons = np.full(
            len(frame), "insufficient_lot_context", dtype=object
        )
        if has_lot_context:
            location = np.asarray(lot_guard["location"], dtype=np.float64)
            precision = np.asarray(lot_guard["precision"], dtype=np.float64)
            lot_drift_flags.fill(0)
            lot_drift_reasons.fill("within_development_envelope")
            lot_values = lot_ids.to_numpy()
            for lot_id in sorted(set(lot_values)):
                mask = lot_values == lot_id
                centred_mean = scaled_values[mask].mean(axis=0) - location
                score = float(
                    np.sqrt(max(float(centred_mean @ precision @ centred_mean), 0.0))
                )
                lot_drift_scores[mask] = score
                if int(mask.sum()) < lot_guard["minimum_lot_size"]:
                    lot_drift_flags[mask] = 1
                    lot_drift_reasons[mask] = "insufficient_lot_context"
                elif score > lot_guard["threshold"]:
                    lot_drift_flags[mask] = 1
                    lot_drift_reasons[mask] = "distribution_shift"
        decisions = classifier_flags | vae_flags | sigma_flags | lot_drift_flags

        return pd.DataFrame(
            {
                "chip_id": chip_ids.to_numpy(),
                "lot_id": lot_ids.to_numpy(),
                "decision": np.where(decisions == 1, "RUN", "SKIP"),
                "flag": decisions,
                "classifier_flag": classifier_flags,
                "vae_flag": vae_flags,
                "sigma_flag": sigma_flags,
                "classifier_fail_probability": classifier_probabilities,
                "vae_reconstruction_error": vae_errors,
                "sigma_aggregate_score": aggregate_scores,
                "sigma_correlation_score": correlation_scores,
                "lot_drift_score": lot_drift_scores,
                "lot_drift_blocked": lot_drift_flags,
                "lot_drift_reason": lot_drift_reasons,
                "model_id": self.config["model_id"],
                "bundle_id": self.manifest["bundle_id"],
            }
        )

    def predict_records(self, records: list[dict]) -> list[dict]:
        return self.predict_dataframe(pd.DataFrame.from_records(records)).to_dict(
            orient="records"
        )