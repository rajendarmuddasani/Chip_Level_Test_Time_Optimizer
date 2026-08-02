"""Tests for TestTimeEvaluator business and ML metrics."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from evaluation.metrics import TestTimeEvaluator


@pytest.fixture
def perfect_preds():
    y_true = np.array([0]*80 + [1]*20)
    y_pred = y_true.copy()
    return y_true, y_pred


@pytest.fixture
def mixed_preds():
    rng = np.random.default_rng(7)
    y_true = np.array([0]*80 + [1]*20)
    y_pred = rng.integers(0, 2, size=100)
    return y_true, y_pred


def test_evaluator_constructs():
    ev = TestTimeEvaluator()
    assert ev is not None


def test_evaluate_returns_all_keys(mixed_preds):
    y_true, y_pred = mixed_preds
    ev = TestTimeEvaluator()
    m = ev.evaluate(y_true, y_pred)
    for key in ["accuracy", "precision", "recall", "f1",
                "skip_rate", "test_rate", "escapee_rate", "overreject_rate",
                "true_negatives", "false_positives", "false_negatives",
                "true_positives", "total_samples"]:
        assert key in m, f"missing: {key}"


def test_evaluate_ml_metrics_range(mixed_preds):
    y_true, y_pred = mixed_preds
    m = TestTimeEvaluator().evaluate(y_true, y_pred)
    for k in ["accuracy", "precision", "recall", "f1"]:
        assert 0.0 <= m[k] <= 1.0


def test_evaluate_perfect_accuracy(perfect_preds):
    y_true, y_pred = perfect_preds
    m = TestTimeEvaluator().evaluate(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(1.0)


def test_evaluate_perfect_escapee_rate_zero(perfect_preds):
    y_true, y_pred = perfect_preds
    m = TestTimeEvaluator().evaluate(y_true, y_pred)
    assert m["escapee_rate"] == pytest.approx(0.0)


def test_evaluate_total_samples(mixed_preds):
    y_true, y_pred = mixed_preds
    m = TestTimeEvaluator().evaluate(y_true, y_pred)
    assert m["total_samples"] == 100
    assert m["true_positives"] + m["false_positives"] + \
           m["true_negatives"] + m["false_negatives"] == 100


def test_evaluate_with_proba():
    y_true = np.array([0]*40 + [1]*10)
    y_pred = np.array([0]*40 + [1]*10)
    y_proba = np.array([0.1]*40 + [0.9]*10)
    m = TestTimeEvaluator().evaluate(y_true, y_pred, y_proba)
    assert "roc_auc" in m
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_calculate_time_savings():
    ev = TestTimeEvaluator()
    y_pred = np.array([0]*600 + [1]*400)  # 60% skip
    savings = ev.calculate_time_savings(
        y_pred, test_time_per_chip=30.0, cost_per_hour=100.0,
        n_lots=1, chips_per_lot=1000)
    assert isinstance(savings, dict)
    # Skipping 600 chips of 1000 should save time
    assert any(v > 0 for v in savings.values() if isinstance(v, (int, float)))
