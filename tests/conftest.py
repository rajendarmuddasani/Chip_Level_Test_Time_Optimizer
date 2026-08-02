"""Shared fixtures for Chip-Level Test Time Optimizer tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def normal_df():
    """100-sample normal distribution DataFrame with 5 features."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((100, 5)),
                        columns=[f"feat_{i}" for i in range(5)])


@pytest.fixture
def mixed_df():
    """200-sample DataFrame: 180 normal + 20 outlier rows."""
    rng = np.random.default_rng(0)
    normal = rng.standard_normal((180, 5))
    outliers = rng.standard_normal((20, 5)) * 8  # very large variance
    data = np.vstack([normal, outliers])
    return pd.DataFrame(data, columns=[f"feat_{i}" for i in range(5)])


@pytest.fixture
def binary_labels():
    """Ground-truth labels: 0 = pass, 1 = fail."""
    rng = np.random.default_rng(1)
    labels = np.zeros(200, dtype=int)
    # 10% fail
    labels[rng.choice(200, 20, replace=False)] = 1
    return labels
