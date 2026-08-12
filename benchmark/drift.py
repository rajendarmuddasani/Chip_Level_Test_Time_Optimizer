"""Lot-level distribution-shift guard that disables unsafe skipping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from benchmark.synthetic_data import FEATURE_NAMES


@dataclass(frozen=True)
class LotDriftGuard:
    location: np.ndarray
    precision: np.ndarray
    threshold: float
    minimum_lot_size: int
    calibration: dict

    def score_lots(
        self,
        metadata: pd.DataFrame,
        scaled_features: pd.DataFrame,
    ) -> dict[str, float]:
        if "lot_id" not in metadata:
            raise ValueError("lot_id is required for drift scoring")
        scores = {}
        lot_values = metadata["lot_id"].astype(str).to_numpy()
        feature_values = scaled_features[FEATURE_NAMES].to_numpy(dtype=np.float64)
        for lot_id in sorted(set(lot_values)):
            lot_mean = feature_values[lot_values == lot_id].mean(axis=0)
            centred = lot_mean - self.location
            distance = float(
                np.sqrt(max(float(centred @ self.precision @ centred), 0.0))
            )
            scores[lot_id] = distance
        return scores

    def apply(
        self,
        metadata: pd.DataFrame,
        scaled_features: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return row-aligned scores and block flags plus a lot summary."""
        lot_values = metadata["lot_id"].astype(str).to_numpy()
        lot_scores = self.score_lots(metadata, scaled_features)
        scores = np.zeros(len(metadata), dtype=np.float64)
        blocked = np.zeros(len(metadata), dtype=np.int64)
        summary = {}

        for lot_id, score in lot_scores.items():
            mask = lot_values == lot_id
            lot_size = int(mask.sum())
            reason = None
            if lot_size < self.minimum_lot_size:
                reason = "insufficient_lot_context"
            elif score > self.threshold:
                reason = "distribution_shift"
            scores[mask] = score
            if reason is not None:
                blocked[mask] = 1
            summary[lot_id] = {
                "score": score,
                "chips": lot_size,
                "blocked": reason is not None,
                "reason": reason,
            }

        return scores, blocked, summary

    def to_config(self) -> dict:
        return {
            "location": [float(value) for value in self.location],
            "precision": [
                [float(value) for value in row] for row in self.precision
            ],
            "threshold": self.threshold,
            "minimum_lot_size": self.minimum_lot_size,
            "calibration": self.calibration,
        }


def _lot_means(
    metadata: pd.DataFrame,
    scaled_features: pd.DataFrame,
) -> pd.DataFrame:
    frame = scaled_features[FEATURE_NAMES].copy()
    frame["lot_id"] = metadata["lot_id"].astype(str).to_numpy()
    return frame.groupby("lot_id", sort=True)[FEATURE_NAMES].mean()


def fit_lot_drift_guard(
    train_metadata: pd.DataFrame,
    train_scaled_features: pd.DataFrame,
    known_validation: Iterable[tuple[pd.DataFrame, pd.DataFrame]],
    ood_validation: Iterable[tuple[pd.DataFrame, pd.DataFrame]],
    minimum_lot_size: int = 100,
) -> LotDriftGuard:
    """Fit on training lots and select for 100% development OOD lot blocking."""
    train_means = _lot_means(train_metadata, train_scaled_features)
    covariance = LedoitWolf().fit(train_means.to_numpy(dtype=np.float64))
    provisional = LotDriftGuard(
        location=covariance.location_,
        precision=covariance.precision_,
        threshold=float("inf"),
        minimum_lot_size=minimum_lot_size,
        calibration={},
    )

    known_scores = {
        lot_id: score
        for metadata, features in known_validation
        for lot_id, score in provisional.score_lots(metadata, features).items()
    }
    ood_scores = {
        lot_id: score
        for metadata, features in ood_validation
        for lot_id, score in provisional.score_lots(metadata, features).items()
    }
    if not known_scores or not ood_scores:
        raise ValueError("Known and OOD validation lots are required")

    threshold = float(np.nextafter(min(ood_scores.values()), -np.inf))
    known_blocked = sum(score > threshold for score in known_scores.values())
    ood_blocked = sum(score > threshold for score in ood_scores.values())
    calibration = {
        "objective": "highest threshold with 100% development OOD lot detection",
        "training_lots": int(len(train_means)),
        "known_validation_lots": len(known_scores),
        "known_validation_blocked_lots": known_blocked,
        "known_validation_false_block_rate": known_blocked / len(known_scores),
        "ood_validation_lots": len(ood_scores),
        "ood_validation_blocked_lots": ood_blocked,
        "ood_validation_detection_rate": ood_blocked / len(ood_scores),
        "known_score_range": [
            min(known_scores.values()),
            max(known_scores.values()),
        ],
        "ood_score_range": [
            min(ood_scores.values()),
            max(ood_scores.values()),
        ],
    }
    if ood_blocked != len(ood_scores):
        raise RuntimeError("Drift threshold failed the complete OOD lot-detection gate")

    return LotDriftGuard(
        location=covariance.location_,
        precision=covariance.precision_,
        threshold=threshold,
        minimum_lot_size=minimum_lot_size,
        calibration=calibration,
    )
