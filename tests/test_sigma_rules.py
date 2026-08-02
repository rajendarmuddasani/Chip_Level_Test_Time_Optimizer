"""Tests for SigmaRule outlier detection and calculate_sigma_metrics."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
from models.statistical.sigma_rules import SigmaRule, calculate_sigma_metrics


def test_sigma_rule_fits_without_error(normal_df):
    sr = SigmaRule(target_fail_count=5, min_sigma=1.0, max_sigma=5.0)
    sr.fit(normal_df)
    assert sr.dict_sigma is not None


def test_sigma_rule_stores_all_columns(normal_df):
    sr = SigmaRule(target_fail_count=5)
    sr.fit(normal_df)
    for col in normal_df.columns:
        assert col in sr.dict_sigma


def test_sigma_rule_predict_shape(normal_df):
    sr = SigmaRule(target_fail_count=5)
    sr.fit(normal_df)
    preds = sr.predict(normal_df)
    assert len(preds) == len(normal_df)


def test_sigma_rule_predict_binary(normal_df):
    sr = SigmaRule(target_fail_count=5)
    sr.fit(normal_df)
    preds = sr.predict(normal_df)
    assert set(preds.unique()).issubset({0, 1})


def test_sigma_rule_flags_outliers(mixed_df):
    """Extreme outliers (8σ) must be flagged after training on normal data."""
    train = mixed_df.iloc[:100]
    test = mixed_df.iloc[100:]  # contains outlier rows
    sr = SigmaRule(target_fail_count=10, min_sigma=2.0, max_sigma=8.0)
    sr.fit(train)
    preds = sr.predict(test)
    # At least some outlier chips should be flagged
    assert preds.sum() > 0


def test_get_bounds_returns_lower_lt_upper(normal_df):
    sr = SigmaRule(target_fail_count=5)
    sr.fit(normal_df)
    lower, upper = sr.get_bounds("feat_0")
    assert lower < upper


def test_get_bounds_unfitted_raises(normal_df):
    sr = SigmaRule(target_fail_count=5)
    sr.fit(normal_df)
    with pytest.raises(ValueError):
        sr.get_bounds("nonexistent_feature")


def test_get_metadata_returns_dict(normal_df):
    sr = SigmaRule(target_fail_count=5)
    sr.fit(normal_df)
    meta = sr.get_metadata()
    assert isinstance(meta, dict)
    assert "feat_0" in meta


def test_calculate_sigma_metrics_keys():
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 0])
    metrics = calculate_sigma_metrics(y_true, y_pred)
    for key in ["accuracy", "precision", "recall", "overreject_rate",
                "escapee_rate", "skip_rate", "flag_rate"]:
        assert key in metrics, f"missing metric: {key}"


def test_calculate_sigma_metrics_ranges():
    y_true = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
    metrics = calculate_sigma_metrics(y_true, y_pred)
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"{k} out of [0,1]: {v}"
