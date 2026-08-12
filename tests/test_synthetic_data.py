"""Contracts for the independently generated public benchmark data."""

import pandas as pd
import pytest

from benchmark.synthetic_data import (
    FEATURE_NAMES,
    OOD_FAILURE_MODE,
    SplitSpec,
    assert_split_isolation,
    build_dataset_manifest,
    generate_benchmark,
    generate_split,
)


SMALL_SPECS = (
    SplitSpec("train", 0, 2, 40, 0.10, 0.00),
    SplitSpec("validation", 5, 1, 40, 0.10, 0.10),
    SplitSpec("confirmation", 10, 1, 40, 0.10, 0.20),
    SplitSpec("ood_challenge", 15, 1, 40, 0.20, 0.30, ood=True),
)


def test_split_generation_is_deterministic():
    first = generate_split(SMALL_SPECS[0], base_seed=7)
    second = generate_split(SMALL_SPECS[0], base_seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_split_generation_changes_with_seed():
    first = generate_split(SMALL_SPECS[0], base_seed=7)
    second = generate_split(SMALL_SPECS[0], base_seed=8)

    assert not first[FEATURE_NAMES].equals(second[FEATURE_NAMES])


def test_benchmark_has_disjoint_lots_chips_and_time_ranges():
    splits = generate_benchmark(SMALL_SPECS, base_seed=11)

    assert_split_isolation(splits)
    assert sum(frame["lot_id"].nunique() for frame in splits.values()) == 5


def test_isolation_rejects_reused_lot():
    train = generate_split(SMALL_SPECS[0], base_seed=11)
    reused = train.iloc[:20].copy()
    reused["split"] = "validation"

    with pytest.raises(ValueError, match="Chip overlap"):
        assert_split_isolation({"train": train, "validation": reused})


def test_ood_split_contains_only_unseen_failure_mode_for_failed_chips():
    ood = generate_split(SMALL_SPECS[-1], base_seed=13)

    assert set(ood.loc[ood["label"] == 1, "failure_mode"]) == {OOD_FAILURE_MODE}


def test_manifest_records_counts_hashes_and_zero_overlap():
    splits = generate_benchmark(SMALL_SPECS, base_seed=17)
    manifest = build_dataset_manifest(splits, SMALL_SPECS, base_seed=17)

    assert manifest["feature_count"] == 32
    assert manifest["splits"]["train"]["samples"] == 80
    assert len(manifest["splits"]["train"]["sha256"]) == 64
    assert set(manifest["isolation"].values()) == {0}