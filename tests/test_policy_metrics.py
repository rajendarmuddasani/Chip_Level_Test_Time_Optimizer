"""Tests for canonical safety metrics and validation-only selection."""

import numpy as np
import pytest

from benchmark.policy_metrics import (
    evaluate_by_failure_mode,
    evaluate_policy,
    gate_candidate,
    one_sided_binomial_upper,
    select_candidate,
)


def test_policy_metrics_use_total_population_and_explicit_stage_cost():
    labels = np.array([0, 0, 0, 1])
    flags = np.array([0, 0, 0, 1])

    metrics = evaluate_policy(labels, flags)

    assert metrics["skip_rate"] == pytest.approx(0.75)
    assert metrics["test_rate"] == pytest.approx(0.25)
    assert metrics["time_reduction_percent"] == pytest.approx(11.25)
    assert metrics["escapees"] == 0
    assert metrics["overtest_rate"] == pytest.approx(0.0)


def test_policy_reports_relative_and_absolute_escape_rates():
    labels = np.array([0, 0, 1, 1])
    flags = np.array([0, 0, 0, 1])

    metrics = evaluate_policy(labels, flags)

    assert metrics["relative_escape_rate"] == pytest.approx(0.5)
    assert metrics["absolute_escape_rate"] == pytest.approx(0.25)
    assert metrics["defect_recall"] == pytest.approx(0.5)


def test_zero_observed_escapees_still_has_uncertainty():
    upper = one_sided_binomial_upper(events=0, trials=240)

    assert upper == pytest.approx(1.0 - 0.05 ** (1.0 / 240), rel=1e-9)
    assert 0.01 < upper < 0.015


def test_unsafe_candidate_fails_escape_gate():
    metrics = evaluate_policy(
        np.array([0] * 760 + [1] * 240),
        np.array([0] * 760 + [0] + [1] * 239),
    )

    result = gate_candidate(metrics)

    assert result["passed"] is False
    assert result["checks"]["observed_escapees"] is False
    assert result["checks"]["relative_escape_rate"] is False


def test_selection_maximizes_reduction_only_among_safe_candidates():
    labels = np.array([0] * 760 + [1] * 240)
    candidates = [
        {
            "name": "safe_low",
            "eligible_for_selection": True,
            "metrics": evaluate_policy(labels, np.array([0] * 600 + [1] * 400)),
        },
        {
            "name": "safe_high",
            "eligible_for_selection": True,
            "metrics": evaluate_policy(labels, np.array([0] * 700 + [1] * 300)),
        },
        {
            "name": "unsafe",
            "eligible_for_selection": True,
            "metrics": evaluate_policy(labels, np.array([0] * 761 + [1] * 239)),
        },
    ]

    selected, evaluated = select_candidate(candidates)

    assert selected["name"] == "safe_high"
    assert len(evaluated) == 3
    assert (
        next(item for item in evaluated if item["name"] == "unsafe")["gate_result"][
            "passed"
        ]
        is False
    )


def test_failure_mode_metrics_expose_weakest_mode():
    modes = ["pass", "mode_a", "mode_a", "mode_b", "mode_b"]
    labels = np.array([0, 1, 1, 1, 1])
    flags = np.array([0, 1, 1, 1, 0])

    metrics = evaluate_by_failure_mode(modes, labels, flags)

    assert metrics["mode_a"]["recall"] == pytest.approx(1.0)
    assert metrics["mode_b"]["recall"] == pytest.approx(0.5)


def test_policy_rejects_nonbinary_flags():
    with pytest.raises(ValueError, match="Flags must be binary"):
        evaluate_policy(np.array([0, 1]), np.array([0, 2]))