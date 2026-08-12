"""Canonical policy metrics and validation-only candidate selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from statistics import NormalDist
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import beta
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef


@dataclass(frozen=True)
class CostModel:
    early_stage_units: float = 85.0
    optional_stage_units: float = 15.0

    @property
    def full_flow_units(self) -> float:
        return self.early_stage_units + self.optional_stage_units


@dataclass(frozen=True)
class SafetyGates:
    maximum_observed_escapees: int = 0
    maximum_relative_escape_rate: float = 0.0
    maximum_escape_rate_upper_95: float = 0.015
    maximum_overtest_rate: float = 0.2


DEFAULT_COST_MODEL = CostModel()
DEFAULT_SAFETY_GATES = SafetyGates()


def one_sided_binomial_upper(
    events: int,
    trials: int,
    confidence: float = 0.95,
) -> float:
    if trials < 0 or events < 0 or events > trials:
        raise ValueError("Require 0 <= events <= trials")
    if trials == 0 or events == trials:
        return 1.0
    return float(beta.ppf(confidence, events + 1, trials - events))


def wilson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Require 0 <= successes <= trials and trials > 0")
    z_score = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z_score**2 / trials
    centre = (proportion + z_score**2 / (2.0 * trials)) / denominator
    margin = (
        z_score
        * sqrt(
            proportion * (1.0 - proportion) / trials
            + z_score**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def evaluate_policy(
    y_true: np.ndarray,
    flags: np.ndarray,
    cost_model: CostModel = DEFAULT_COST_MODEL,
) -> dict:
    labels = np.asarray(y_true, dtype=np.int64)
    decisions = np.asarray(flags, dtype=np.int64)
    if labels.ndim != 1 or decisions.ndim != 1 or len(labels) != len(decisions):
        raise ValueError("Labels and flags must be one-dimensional and equal length")
    if not len(labels):
        raise ValueError("Cannot evaluate an empty policy result")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must be binary")
    if not set(np.unique(decisions)).issubset({0, 1}):
        raise ValueError("Flags must be binary")

    tn, fp, fn, tp = confusion_matrix(labels, decisions, labels=[0, 1]).ravel()
    total = len(labels)
    failures = int(tp + fn)
    passes = int(tn + fp)
    skipped = int(tn + fn)
    tested = int(tp + fp)
    skip_rate = skipped / total
    time_reduction = (
        skip_rate * cost_model.optional_stage_units / cost_model.full_flow_units
    )
    skip_ci = wilson_interval(skipped, total)

    return {
        "total_chips": total,
        "pass_chips": passes,
        "failed_chips": failures,
        "true_skip": int(tn),
        "overtest": int(fp),
        "escapees": int(fn),
        "true_test": int(tp),
        "skip_rate": skip_rate,
        "test_rate": tested / total,
        "defect_recall": tp / failures if failures else 0.0,
        "relative_escape_rate": fn / failures if failures else 0.0,
        "absolute_escape_rate": fn / total,
        "escape_rate_upper_95": one_sided_binomial_upper(int(fn), failures),
        "overtest_rate": fp / passes if passes else 0.0,
        "f1": f1_score(labels, decisions, zero_division=0),
        "mcc": matthews_corrcoef(labels, decisions),
        "time_reduction": time_reduction,
        "time_reduction_percent": time_reduction * 100.0,
        "time_reduction_ci_95_percent": [
            skip_ci[0]
            * cost_model.optional_stage_units
            / cost_model.full_flow_units
            * 100.0,
            skip_ci[1]
            * cost_model.optional_stage_units
            / cost_model.full_flow_units
            * 100.0,
        ],
        "cost_model": asdict(cost_model),
    }


def evaluate_by_failure_mode(
    failure_modes: Iterable[str],
    y_true: np.ndarray,
    flags: np.ndarray,
) -> dict:
    modes = np.asarray(list(failure_modes), dtype=object)
    labels = np.asarray(y_true, dtype=np.int64)
    decisions = np.asarray(flags, dtype=np.int64)
    if len(modes) != len(labels) or len(labels) != len(decisions):
        raise ValueError("Failure modes, labels, and flags must be equal length")

    result = {}
    for mode in sorted(set(modes[labels == 1])):
        mask = (modes == mode) & (labels == 1)
        support = int(mask.sum())
        captured = int(decisions[mask].sum())
        result[str(mode)] = {
            "support": support,
            "captured": captured,
            "escaped": support - captured,
            "recall": captured / support if support else 0.0,
        }
    return result


def gate_candidate(
    metrics: Mapping[str, float | int],
    gates: SafetyGates = DEFAULT_SAFETY_GATES,
) -> dict:
    checks = {
        "observed_escapees": int(metrics["escapees"])
        <= gates.maximum_observed_escapees,
        "relative_escape_rate": float(metrics["relative_escape_rate"])
        <= gates.maximum_relative_escape_rate,
        "escape_uncertainty": float(metrics["escape_rate_upper_95"])
        <= gates.maximum_escape_rate_upper_95,
        "overtest": float(metrics["overtest_rate"])
        <= gates.maximum_overtest_rate,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "gates": asdict(gates),
    }


def select_candidate(
    candidates: Iterable[dict],
    gates: SafetyGates = DEFAULT_SAFETY_GATES,
) -> tuple[dict, list[dict]]:
    evaluated = []
    for candidate in candidates:
        record = dict(candidate)
        record["gate_result"] = gate_candidate(record["metrics"], gates)
        evaluated.append(record)

    eligible = [
        record
        for record in evaluated
        if record["eligible_for_selection"] and record["gate_result"]["passed"]
    ]
    if not eligible:
        raise RuntimeError("No eligible candidate passed all safety gates")

    selected = sorted(
        eligible,
        key=lambda record: (
            -float(record["metrics"]["time_reduction"]),
            float(record["metrics"]["overtest_rate"]),
            str(record["name"]),
        ),
    )[0]
    return selected, evaluated
