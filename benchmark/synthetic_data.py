"""Deterministic, independently generated chip-test benchmark data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd

GENERATOR_VERSION = "public_grouped_synthetic_v1"
BASE_SEED = 20260807
FEATURE_GROUPS = {
    "voltage": [f"V_{index:02d}" for index in range(8)],
    "current": [f"I_{index:02d}" for index in range(8)],
    "timing": [f"T_{index:02d}" for index in range(8)],
    "resistance": [f"R_{index:02d}" for index in range(8)],
}
FEATURE_NAMES = [name for group in FEATURE_GROUPS.values() for name in group]
KNOWN_FAILURE_MODES = (
    "voltage_drift",
    "leakage_spike",
    "timing_shift",
    "resistance_bridge",
)
OOD_FAILURE_MODE = "novel_coupled_drift"


@dataclass(frozen=True)
class SplitSpec:
    name: str
    first_time_index: int
    lot_count: int
    chips_per_lot: int
    fail_rate: float
    phase_drift: float
    ood: bool = False


DEFAULT_SPLITS = (
    SplitSpec("train", 0, 32, 500, 0.06, 0.0),
    SplitSpec("validation", 40, 8, 500, 0.06, 0.1),
    SplitSpec("known_shift_validation", 60, 12, 500, 0.06, 0.18),
    SplitSpec("ood_validation", 80, 4, 500, 0.2, 0.26, ood=True),
    SplitSpec("known_shift_validation_2", 100, 20, 500, 0.06, 0.3),
    SplitSpec("ood_validation_2", 130, 10, 500, 0.2, 0.36, ood=True),
    SplitSpec("confirmation", 160, 20, 500, 0.06, 0.42),
    SplitSpec("ood_confirmation", 190, 10, 500, 0.2, 0.48, ood=True),
)


def _apply_known_failure(
    features: np.ndarray,
    row_indices: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> None:
    if not len(row_indices):
        return
    if mode == "voltage_drift":
        features[row_indices, 0:4] += rng.normal(
            2.7, 0.25, (len(row_indices), 4)
        )
    elif mode == "leakage_spike":
        features[row_indices, 8:12] += rng.normal(
            3.0, 0.3, (len(row_indices), 4)
        )
    elif mode == "timing_shift":
        features[row_indices, 16:20] -= rng.normal(
            2.5, 0.25, (len(row_indices), 4)
        )
    elif mode == "resistance_bridge":
        features[row_indices, 24:28] += rng.normal(
            2.2, 0.25, (len(row_indices), 4)
        )
        features[row_indices, 28:32] -= rng.normal(
            2.2, 0.25, (len(row_indices), 4)
        )
    else:
        raise ValueError(f"Unknown failure mode: {mode}")


def _apply_ood_failure(
    features: np.ndarray,
    row_indices: np.ndarray,
    rng: np.random.Generator,
) -> None:
    if not len(row_indices):
        return
    coupled = rng.normal(1.65, 0.18, (len(row_indices), 1))
    features[row_indices, 4:8] += coupled
    features[row_indices, 12:16] -= coupled
    features[row_indices, 20:24] += coupled * 0.85
    features[row_indices, 28:32] -= coupled * 0.85


def generate_split(
    spec: SplitSpec,
    base_seed: int = BASE_SEED,
) -> pd.DataFrame:
    """Generate one split using independent seeds for every lot."""
    frames = []
    for lot_offset in range(spec.lot_count):
        time_index = spec.first_time_index + lot_offset
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, time_index]))
        lot_id = f"{spec.name.upper()}_LOT_{time_index:03d}"
        n_fail = int(round(spec.chips_per_lot * spec.fail_rate))

        process_offset = rng.normal(0.0, 0.09, len(FEATURE_NAMES))
        drift_vector = np.linspace(-1.0, 1.0, len(FEATURE_NAMES)) * spec.phase_drift
        features = rng.normal(
            0.0,
            1.0,
            (spec.chips_per_lot, len(FEATURE_NAMES)),
        )
        features += process_offset + drift_vector

        labels = np.zeros(spec.chips_per_lot, dtype=np.int64)
        fail_indices = rng.choice(spec.chips_per_lot, size=n_fail, replace=False)
        labels[fail_indices] = 1
        failure_modes = np.full(spec.chips_per_lot, "pass", dtype=object)

        if spec.ood:
            failure_modes[fail_indices] = OOD_FAILURE_MODE
            _apply_ood_failure(features, fail_indices, rng)
        else:
            shuffled_failures = rng.permutation(fail_indices)
            mode_groups = np.array_split(shuffled_failures, len(KNOWN_FAILURE_MODES))
            for mode, mode_indices in zip(KNOWN_FAILURE_MODES, mode_groups):
                failure_modes[mode_indices] = mode
                _apply_known_failure(features, mode_indices, mode, rng)

        frame = pd.DataFrame(features, columns=FEATURE_NAMES)
        frame.insert(0, "failure_mode", failure_modes)
        frame.insert(0, "label", labels)
        frame.insert(0, "time_index", time_index)
        frame.insert(0, "lot_id", lot_id)
        frame.insert(
            0,
            "chip_id",
            [f"{lot_id}_CHIP_{index:04d}" for index in range(spec.chips_per_lot)],
        )
        frame.insert(0, "split", spec.name)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def generate_benchmark(
    split_specs: Iterable[SplitSpec] = DEFAULT_SPLITS,
    base_seed: int = BASE_SEED,
) -> Dict[str, pd.DataFrame]:
    """Generate every benchmark split and fail if group isolation is broken."""
    splits = {spec.name: generate_split(spec, base_seed) for spec in split_specs}
    assert_split_isolation(splits)
    return splits


def assert_split_isolation(splits: Mapping[str, pd.DataFrame]) -> None:
    """Require unique chips/lots and non-overlapping chronological phases."""
    seen_chips = set()
    seen_lots = set()
    time_ranges = []

    for split_name, frame in splits.items():
        chip_ids = set(frame["chip_id"].astype(str))
        lot_ids = set(frame["lot_id"].astype(str))
        if chip_ids & seen_chips:
            raise ValueError(f"Chip overlap detected in {split_name}")
        if lot_ids & seen_lots:
            raise ValueError(f"Lot overlap detected in {split_name}")
        seen_chips.update(chip_ids)
        seen_lots.update(lot_ids)
        time_ranges.append(
            (
                int(frame["time_index"].min()),
                int(frame["time_index"].max()),
                split_name,
            )
        )

    ordered_ranges = sorted(time_ranges)
    for previous, current in zip(ordered_ranges, ordered_ranges[1:]):
        if previous[1] >= current[0]:
            raise ValueError(
                f"Time overlap detected between {previous[2]} and {current[2]}"
            )


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash a stable CSV representation without committing generated rows."""
    canonical = frame.to_csv(
        index=False,
        float_format="%.8f",
        lineterminator="\n",
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def build_dataset_manifest(
    splits: Mapping[str, pd.DataFrame],
    split_specs: Iterable[SplitSpec] = DEFAULT_SPLITS,
    base_seed: int = BASE_SEED,
) -> dict:
    """Build the machine-readable data and split evidence contract."""
    specs_by_name = {spec.name: spec for spec in split_specs}
    return {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "provenance": "independently generated synthetic data; no external dataset",
        "license": "MIT repository code; generated rows are not redistributed",
        "base_seed": base_seed,
        "feature_count": len(FEATURE_NAMES),
        "feature_groups": FEATURE_GROUPS,
        "known_failure_modes": list(KNOWN_FAILURE_MODES),
        "ood_failure_mode": OOD_FAILURE_MODE,
        "splits": {
            name: {
                "spec": asdict(specs_by_name[name]),
                "samples": int(len(frame)),
                "pass_samples": int((frame["label"] == 0).sum()),
                "fail_samples": int((frame["label"] == 1).sum()),
                "lot_count": int(frame["lot_id"].nunique()),
                "time_index_min": int(frame["time_index"].min()),
                "time_index_max": int(frame["time_index"].max()),
                "failure_mode_counts": {
                    str(key): int(value)
                    for key, value in frame["failure_mode"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "sha256": dataframe_sha256(frame),
            }
            for name, frame in splits.items()
        },
        "isolation": {
            "chip_overlap": 0,
            "lot_overlap": 0,
            "time_overlap": 0,
        },
    }
