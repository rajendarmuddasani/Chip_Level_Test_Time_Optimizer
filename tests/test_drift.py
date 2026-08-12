"""Tests for lot-level fail-closed drift detection."""

import numpy as np
import pandas as pd

from benchmark.drift import fit_lot_drift_guard
from benchmark.synthetic_data import FEATURE_NAMES, SplitSpec, generate_split
from preprocessing.preprocessing import FeatureScaler


def test_lot_drift_guard_blocks_every_development_ood_lot():
    train = generate_split(
        SplitSpec("train", 0, 8, 120, 0.06, 0.00), base_seed=41
    )
    known = generate_split(
        SplitSpec("known", 20, 3, 120, 0.06, 0.15), base_seed=41
    )
    ood = generate_split(
        SplitSpec("ood", 30, 2, 120, 0.20, 0.25, ood=True),
        base_seed=41,
    )
    scaler = FeatureScaler().fit(train[FEATURE_NAMES])
    train_scaled = scaler.transform(train[FEATURE_NAMES])
    known_scaled = scaler.transform(known[FEATURE_NAMES])
    ood_scaled = scaler.transform(ood[FEATURE_NAMES])

    guard = fit_lot_drift_guard(
        train,
        train_scaled,
        [(known, known_scaled)],
        [(ood, ood_scaled)],
        minimum_lot_size=100,
    )
    _, ood_blocked, summary = guard.apply(ood, ood_scaled)

    assert ood_blocked.mean() == 1.0
    assert all(item["reason"] == "distribution_shift" for item in summary.values())
    assert guard.calibration["ood_validation_detection_rate"] == 1.0


def test_lot_drift_guard_fails_closed_on_small_lot():
    rng = np.random.default_rng(3)
    train = pd.DataFrame(rng.normal(size=(400, 32)), columns=FEATURE_NAMES)
    train.insert(0, "lot_id", np.repeat(["A", "B", "C", "D"], 100))
    known = train.copy()
    known["lot_id"] = "KNOWN"
    ood = train.copy()
    ood.loc[:, FEATURE_NAMES] += 2.0
    ood["lot_id"] = "OOD"
    guard = fit_lot_drift_guard(
        train,
        train[FEATURE_NAMES],
        [(known, known[FEATURE_NAMES])],
        [(ood, ood[FEATURE_NAMES])],
        minimum_lot_size=100,
    )
    small = known.head(20).copy()
    _, blocked, summary = guard.apply(small, small[FEATURE_NAMES])

    assert blocked.mean() == 1.0
    assert summary["KNOWN"]["reason"] == "insufficient_lot_context"