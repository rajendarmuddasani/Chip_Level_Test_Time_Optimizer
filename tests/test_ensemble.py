"""Tests for HybridEnsemble OR-logic combining sigma rules + optional DL models."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from models.ensemble import HybridEnsemble, calculate_time_savings
from models.statistical.sigma_rules import SigmaRule


@pytest.fixture
def fitted_sigma(normal_df):
    sr = SigmaRule(target_fail_count=5)
    sr.fit(normal_df)
    return sr


def test_ensemble_constructs_no_models():
    ens = HybridEnsemble()
    assert ens is not None


def test_ensemble_constructs_with_sigma(fitted_sigma):
    ens = HybridEnsemble(sigma=fitted_sigma)
    assert ens.sigma is not None


def test_ensemble_predict_shape(normal_df, fitted_sigma):
    ens = HybridEnsemble(sigma=fitted_sigma)
    flags = ens.predict(normal_df)
    assert len(flags) == len(normal_df)


def test_ensemble_predict_binary(normal_df, fitted_sigma):
    ens = HybridEnsemble(sigma=fitted_sigma)
    flags = ens.predict(normal_df)
    assert set(flags).issubset({0, 1})


def test_ensemble_no_models_all_skip(normal_df):
    """With no sub-models, ensemble should flag nothing."""
    ens = HybridEnsemble(classifier=None, vae=None, sigma=None)
    flags = ens.predict(normal_df)
    assert flags.sum() == 0


def test_ensemble_or_logic_with_sigma(mixed_df):
    """Sigma rule flags outliers; ensemble with only sigma should reproduce those flags."""
    train = mixed_df.iloc[:100]
    test = mixed_df.iloc[100:]
    sr = SigmaRule(target_fail_count=10, min_sigma=2.0)
    sr.fit(train)

    sigma_flags = sr.predict(test).values
    ens = HybridEnsemble(sigma=sr)
    ens_flags = ens.predict(test)

    # OR logic: ensemble flags must be superset of sigma flags
    assert ((sigma_flags == 1) <= (ens_flags == 1)).all()


def test_ensemble_conservative_default_threshold():
    ens = HybridEnsemble()
    assert ens.classifier_threshold <= 0.5  # conservative (lower threshold)


def test_time_savings_only_counts_optional_stage():
    flags = np.array([0] * 75 + [1] * 25)

    savings = calculate_time_savings(flags)

    assert savings["skip_rate"] == pytest.approx(0.75)
    assert savings["time_reduction_percent"] == pytest.approx(11.25)
    assert savings["actual_units"] == pytest.approx(8875.0)
