"""Tests for DataValidator input validation logic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest
from preprocessing.preprocessing import DataValidator


def test_validator_constructs():
    dv = DataValidator()
    assert dv is not None


def test_valid_df_passes(normal_df):
    dv = DataValidator()
    is_valid, errors = dv.validate(normal_df)
    assert is_valid
    assert errors == []


def test_empty_df_fails():
    dv = DataValidator()
    is_valid, errors = dv.validate(pd.DataFrame())
    assert not is_valid
    assert len(errors) > 0


def test_missing_required_feature_fails(normal_df):
    dv = DataValidator(expected_features=["feat_0", "feat_99"])
    is_valid, errors = dv.validate(normal_df)
    assert not is_valid
    assert any("feat_99" in e for e in errors)


def test_all_required_features_present_passes(normal_df):
    required = ["feat_0", "feat_1", "feat_2"]
    dv = DataValidator(expected_features=required)
    is_valid, errors = dv.validate(normal_df)
    assert is_valid


def test_no_expected_features_always_passes_nonempty(normal_df):
    dv = DataValidator(expected_features=None)
    is_valid, _ = dv.validate(normal_df)
    assert is_valid


def test_errors_list_when_invalid():
    dv = DataValidator(expected_features=["missing_col"])
    is_valid, errors = dv.validate(pd.DataFrame({"a": [1, 2]}))
    assert not is_valid
    assert isinstance(errors, list)
    assert len(errors) > 0
