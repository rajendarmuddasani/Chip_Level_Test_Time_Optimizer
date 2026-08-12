"""Executable evidence walkthrough for the frozen public synthetic policy.

Run from the repository root:
    python notebooks/01_train_and_evaluate.py

The complete training and selection command is intentionally separate:
    python scripts/run_public_benchmark.py --output-root tmp/replay
"""

# %% [markdown]
# # Chip-Level Test Time Optimizer
#
# This walkthrough reads canonical evidence, executes the frozen policy on a public
# unlabelled lot, and replays every evidence contract. All data is independently
# generated synthetic data. The 15% reduction and zero-escape objectives are unmet.

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.runtime import HybridPolicyRuntime
from scripts.validate_evidence import validate


def read_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


public = read_json("evidence/public_synthetic_evaluation.json")
dataset = read_json("evidence/public_synthetic_dataset_manifest.json")
operational = read_json("evidence/operational_envelope_confirmation.json")
claims = read_json("evidence/claims.json")

print("Data scope:", claims["data_scope"]["provenance"])
print("Frozen bundle:", claims["champion"]["bundle_id"])
print("Benchmark splits:", len(dataset["splits"]))

# %% [markdown]
# ## Data roles
#
# Training, known-mode development, OOD drift calibration, frozen confirmation,
# and post-freeze confirmation have separate lots and time ranges.

# %%
split_rows = []
for name, details in dataset["splits"].items():
    split_rows.append(
        {
            "split": name,
            "chips": details["samples"],
            "lots": details["lot_count"],
            "failures": details["fail_samples"],
            "time_range": f"{details['time_index_min']}-{details['time_index_max']}",
        }
    )
split_table = pd.DataFrame(split_rows).sort_values("time_range")
print(split_table.to_string(index=False))
assert dataset["isolation"] == {
    "chip_overlap": 0,
    "lot_overlap": 0,
    "time_overlap": 0,
}

# %% [markdown]
# ## Candidate tradeoff
#
# Component ablations were measured but were not eligible for promotion. The hybrid
# was selected from 180 eligible OR policies by maximizing validation simulated
# reduction while satisfying every predeclared safety gate.

# %%
selected = public["validation"]["selected_candidate"]
comparison_names = {
    "all_test_baseline",
    "classifier_t0.20",
    "vae_q990",
    "sigma_tail_0.00010",
    selected["name"],
}
comparison_rows = []
for candidate in public["validation"]["candidates"]:
    if candidate["name"] not in comparison_names:
        continue
    metrics = candidate["metrics"]
    comparison_rows.append(
        {
            "candidate": candidate["name"],
            "policy_family": candidate["policy_family"],
            "eligible": candidate["eligible_for_selection"],
            "reduction_percent": metrics["time_reduction_percent"],
            "relative_escape_rate": metrics["relative_escape_rate"],
            "overtest_rate": metrics["overtest_rate"],
            "gate_passed": candidate["gate_result"]["passed"],
        }
    )
comparison = pd.DataFrame(comparison_rows)
print(comparison.to_string(index=False))
print("Selected:", selected["name"])
assert public["validation"]["candidate_count"] == 198
assert public["validation"]["eligible_candidate_count"] == 180

# %% [markdown]
# ## Retained failed confirmations
#
# The first selected-policy confirmation safely blocked every lot, but that meant
# 100% over-test and zero reduction. It was rejected on utility. The OOD confirmation
# also runs every chip by design when all shifted lots are blocked.

# %%
first_confirmation = public["confirmation"]["metrics"]
ood_confirmation = public["ood_confirmation"]["metrics"]
print(
    {
        "first_confirmation_reduction_percent": first_confirmation[
            "time_reduction_percent"
        ],
        "first_confirmation_overtest_rate": first_confirmation["overtest_rate"],
        "first_confirmation_escapes": first_confirmation["escapees"],
        "ood_reduction_percent": ood_confirmation["time_reduction_percent"],
        "ood_escapes": ood_confirmation["escapees"],
    }
)
assert public["acceptance"]["passed"] is False
assert first_confirmation["time_reduction_percent"] == 0.0
assert first_confirmation["overtest_rate"] == 1.0

# %% [markdown]
# ## Accepted post-freeze evidence
#
# The exact frozen bundle was evaluated on 10,000 new chronological synthetic chips
# inside its declared drift envelope with no model or threshold selection.

# %%
accepted = operational["metrics"]
accepted_table = pd.DataFrame(
    [
        ("Chips", accepted["total_chips"]),
        ("Failures", accepted["failed_chips"]),
        ("Simulated reduction (%)", accepted["time_reduction_percent"]),
        ("Defect recall", accepted["defect_recall"]),
        ("Escapes", accepted["escapees"]),
        ("Relative escape rate", accepted["relative_escape_rate"]),
        ("Escape upper 95%", accepted["escape_rate_upper_95"]),
        ("Over-test rate", accepted["overtest_rate"]),
        ("MCC", accepted["mcc"]),
    ],
    columns=["metric", "value"],
)
print(accepted_table.to_string(index=False))
print("Failure modes:", json.dumps(operational["failure_modes"], indent=2))
assert accepted["time_reduction_percent"] == 13.524
assert accepted["escapees"] == 6
assert accepted["failed_chips"] == 600

# %% [markdown]
# ## Live frozen inference
#
# The public fixture is one unlabelled 500-chip lot from the accepted operational
# envelope. Runtime loading verifies model, evidence, dataset, and bundle hashes.

# %%
runtime = HybridPolicyRuntime(ROOT / "artifacts/public_v1/runtime_manifest.json")
fixture = read_json("examples/public_synthetic_input.json")
fixture_frame = pd.DataFrame.from_records(fixture["records"])
predictions = runtime.predict_dataframe(fixture_frame)

skipped = int((predictions["flag"] == 0).sum())
run = len(predictions) - skipped
optional_fraction = 15 / (85 + 15)
live_reduction = skipped / len(predictions) * optional_fraction * 100
print(
    {
        "chips": len(predictions),
        "skip": skipped,
        "run": run,
        "simulated_reduction_percent": live_reduction,
        "bundle_id": predictions["bundle_id"].iloc[0],
    }
)
assert skipped == 450
assert run == 50
assert live_reduction == 13.5
assert predictions["bundle_id"].nunique() == 1

# %% [markdown]
# ## Exact evidence replay
#
# This regenerates all eight benchmark split hashes, verifies the frozen bundle,
# replays post-freeze decisions, checks the prediction SHA-256, and validates claims.

# %%
report = validate(ROOT)
print(
    {
        "status": report["status"],
        "split_hashes_passed": sum(report["dataset_split_checks"].values()),
        "claim_checks_passed": sum(report["claim_checks"].values()),
        "prediction_sha256": report["operational_prediction_sha256"],
    }
)
assert report["status"] == "passed", report["errors"]
assert all(report["dataset_split_checks"].values())
assert all(report["claim_checks"].values())

# %% [markdown]
# ## Truth boundary
#
# - 13.524% is simulated optional-stage reduction under an 85/15 cost model.
# - Six of 600 synthetic known-mode failures escaped post-freeze.
# - Zero observed escapes and 15% reduction remain unmet objectives.
# - OOD safety is lot-level full testing, not per-chip novel-defect recognition.
# - No production savings, physical-defect coverage, or deployment approval is proven.

print("Evidence walkthrough complete.")
