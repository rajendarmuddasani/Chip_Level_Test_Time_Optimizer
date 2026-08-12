"""Streamlit control room for the frozen public chip test policy."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from deployment.runtime import (
    ArtifactIntegrityError,
    HybridPolicyRuntime,
    InputValidationError,
)


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "artifacts" / "public_v1" / "runtime_manifest.json"
PUBLIC_INPUT = ROOT / "examples" / "public_synthetic_input.json"
OPERATIONAL_EVIDENCE = ROOT / "evidence" / "operational_envelope_confirmation.json"
PUBLIC_EVIDENCE = ROOT / "evidence" / "public_synthetic_evaluation.json"


st.set_page_config(
    page_title="Chip Test Policy Control Room",
    page_icon="CP",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #102a43;
        --teal: #087f8c;
        --yellow: #f2c14e;
        --coral: #ef6f6c;
        --blue: #2f6fed;
        --violet: #725ac1;
        --paper: #f7f9fc;
        --ink: #172b4d;
    }
    .stApp {
        background-color: var(--paper);
        background-image: linear-gradient(#dce5ef 1px, transparent 1px),
                          linear-gradient(90deg, #dce5ef 1px, transparent 1px);
        background-size: 32px 32px;
        color: var(--ink);
        font-family: Bahnschrift, "Trebuchet MS", sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(247, 249, 252, 0.94); }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #cfdae7;
        border-top: 4px solid var(--teal);
        border-radius: 6px;
        padding: 0.8rem 1rem;
        min-height: 112px;
    }
    [data-testid="stMetricValue"] { color: var(--navy); }
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px solid #cfdae7;
        border-radius: 6px;
        padding: 0.5rem;
    }
    .policy-strip {
        display: flex;
        gap: 0.65rem;
        flex-wrap: wrap;
        padding: 0.7rem 0 1rem;
    }
    .policy-chip {
        background: #ffffff;
        border: 1px solid #c9d6e4;
        border-left: 5px solid var(--blue);
        border-radius: 4px;
        color: var(--navy);
        padding: 0.45rem 0.7rem;
        font-size: 0.86rem;
        overflow-wrap: anywhere;
    }
    .truth-band {
        background: #fff8df;
        border: 1px solid #efd47f;
        border-left: 6px solid var(--yellow);
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.35rem 0 1rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
    .stTabs [data-baseweb="tab"] { border-radius: 4px 4px 0 0; }
    h1, h2, h3 { color: var(--navy); letter-spacing: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_runtime() -> HybridPolicyRuntime:
    return HybridPolicyRuntime(MANIFEST)


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    payload = json.load(uploaded_file)
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise InputValidationError("JSON must be a list or contain a records list")
    return pd.DataFrame.from_records(records)


def live_summary(runtime: HybridPolicyRuntime, predictions: pd.DataFrame) -> dict:
    total = len(predictions)
    skipped = int((predictions["flag"] == 0).sum())
    cost_model = runtime.config["cost_model"]
    optional_fraction = cost_model["optional_stage_units"] / (
        cost_model["early_stage_units"] + cost_model["optional_stage_units"]
    )
    return {
        "total": total,
        "skipped": skipped,
        "run": total - skipped,
        "skip_rate": skipped / total,
        "time_reduction_percent": skipped / total * optional_fraction * 100,
        "blocked": int(predictions["lot_drift_blocked"].sum()),
    }


st.title("Chip Test Policy Control Room")
st.caption("Classifier + VAE + sigma guardrails + lot drift blocking")

try:
    runtime = load_runtime()
    operational = load_json(OPERATIONAL_EVIDENCE)
    public_evidence = load_json(PUBLIC_EVIDENCE)
except (ArtifactIntegrityError, OSError, ValueError) as error:
    st.error(f"Runtime unavailable: {error}")
    st.stop()

st.markdown(
    f"""
    <div class="policy-strip">
      <span class="policy-chip">Model {runtime.config['model_id']}</span>
      <span class="policy-chip">Bundle {runtime.manifest['bundle_id'][:12]}...</span>
      <span class="policy-chip">32 synthetic ATE features</span>
      <span class="policy-chip">85 + 15 unit cost model</span>
    </div>
    """,
    unsafe_allow_html=True,
)

accepted = operational["metrics"]
metric_columns = st.columns(5)
metric_columns[0].metric("Post-freeze reduction", f"{accepted['time_reduction_percent']:.3f}%")
metric_columns[1].metric("Defect recall", f"{accepted['defect_recall']:.2%}")
metric_columns[2].metric("Escaped failures", f"{accepted['escapees']} / {accepted['failed_chips']}")
metric_columns[3].metric("Over-test rate", f"{accepted['overtest_rate']:.2%}")
metric_columns[4].metric("Lots confirmed", operational["dataset"]["splits"]["operational_envelope_confirmation"]["lot_count"])

st.markdown(
    """
    <div class="truth-band">
      <strong>Accepted boundary:</strong> the post-freeze synthetic envelope gate passed,
      but the 15% reduction and zero-escape objectives remain unmet. Production behavior,
      cost savings, and unseen physical failure modes are not proven.
    </div>
    """,
    unsafe_allow_html=True,
)

source = st.radio(
    "Input source",
    ("Accepted-envelope sample", "Upload lot"),
    horizontal=True,
)
if source == "Accepted-envelope sample":
    input_frame = pd.DataFrame.from_records(load_json(PUBLIC_INPUT)["records"])
else:
    uploaded = st.file_uploader("Lot input", type=("csv", "json"))
    input_frame = read_upload(uploaded) if uploaded is not None else None

if input_frame is None:
    st.info("No lot loaded")
    st.stop()

try:
    predictions = runtime.predict_dataframe(input_frame)
except InputValidationError as error:
    st.error(str(error))
    st.stop()

summary = live_summary(runtime, predictions)
live_columns = st.columns(5)
live_columns[0].metric("Loaded chips", f"{summary['total']:,}")
live_columns[1].metric("SKIP", f"{summary['skipped']:,}")
live_columns[2].metric("RUN", f"{summary['run']:,}")
live_columns[3].metric("Simulated reduction", f"{summary['time_reduction_percent']:.3f}%")
live_columns[4].metric("Drift-blocked chips", f"{summary['blocked']:,}")

operations_tab, evidence_tab, guardrails_tab = st.tabs(
    ("Live lot", "Evidence", "Guardrails")
)

with operations_tab:
    distribution, causes = st.columns((1, 2))
    with distribution:
        st.subheader("Decision mix")
        decision_counts = predictions["decision"].value_counts().rename("chips")
        st.bar_chart(decision_counts, color="#087f8c")
    with causes:
        st.subheader("RUN contributors")
        contributor_counts = pd.DataFrame(
            {
                "chips": [
                    int(predictions["classifier_flag"].sum()),
                    int(predictions["vae_flag"].sum()),
                    int(predictions["sigma_flag"].sum()),
                    int(predictions["lot_drift_blocked"].sum()),
                ]
            },
            index=("Classifier", "VAE", "Sigma", "Lot guard"),
        )
        st.bar_chart(contributor_counts, color="#ef6f6c")

    displayed_columns = [
        "chip_id",
        "lot_id",
        "decision",
        "classifier_fail_probability",
        "vae_reconstruction_error",
        "sigma_aggregate_score",
        "sigma_correlation_score",
        "lot_drift_reason",
    ]
    st.dataframe(
        predictions[displayed_columns],
        width="stretch",
        hide_index=True,
        height=360,
    )
    st.download_button(
        "Download decisions",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name="chip_policy_decisions.csv",
        mime="text/csv",
    )

with evidence_tab:
    selection_column, rejection_column = st.columns(2)
    with selection_column:
        st.subheader("Validation tradeoff")
        st.image(ROOT / "docs" / "assets" / "candidate_tradeoff.png", width="stretch")
        st.caption(
            "Candidates beyond the 25% over-test gate were rejected; the highlighted "
            "validation policy maximized simulated reduction among eligible policies."
        )
    with rejection_column:
        st.subheader("Rejected first confirmation")
        st.image(
            ROOT / "docs" / "assets" / "confirmation_policy_matrix.png",
            width="stretch",
        )
        st.caption(
            "All 20 shifted lots were blocked, producing 0% reduction and 100% over-test. "
            "This result is retained as a failed trial, not promoted as the champion."
        )

    post_freeze = pd.DataFrame(
        [
            ("Total chips", accepted["total_chips"]),
            ("Failures", accepted["failed_chips"]),
            ("Escapes", accepted["escapees"]),
            ("Escape upper 95%", accepted["escape_rate_upper_95"]),
            ("Over-tests", accepted["overtest"]),
            ("MCC", accepted["mcc"]),
        ],
        columns=("Measure", "Value"),
    )
    st.subheader("Post-freeze operational envelope")
    st.dataframe(post_freeze, width="stretch", hide_index=True)

with guardrails_tab:
    checks = operational["gate_result"]["checks"]
    gates = operational["gate_result"]["gates"]
    gate_rows = [
        {
            "Gate": "Observed escapes",
            "Measured": accepted["escapees"],
            "Limit": gates["maximum_observed_escapees"],
            "Passed": checks["observed_escapees"],
        },
        {
            "Gate": "Relative escape rate",
            "Measured": accepted["relative_escape_rate"],
            "Limit": gates["maximum_relative_escape_rate"],
            "Passed": checks["relative_escape_rate"],
        },
        {
            "Gate": "Escape upper 95%",
            "Measured": accepted["escape_rate_upper_95"],
            "Limit": gates["maximum_escape_rate_upper_95"],
            "Passed": checks["escape_uncertainty"],
        },
        {
            "Gate": "Over-test rate",
            "Measured": accepted["overtest_rate"],
            "Limit": gates["maximum_overtest_rate"],
            "Passed": checks["overtest"],
        },
    ]
    st.dataframe(pd.DataFrame(gate_rows), width="stretch", hide_index=True)
    st.subheader("Policy behavior")
    st.write(
        "A chip runs the optional stage when the classifier, VAE, sigma guardrails, "
        "or lot drift guard flags it. Missing or undersized lot context forces RUN."
    )
    st.subheader("Unmet objectives")
    st.write(
        f"15% simulated time reduction and zero observed escapes. The frozen "
        f"post-confirmation values are {accepted['time_reduction_percent']:.3f}% and "
        f"{accepted['escapees']} escapes."
    )

st.caption(
    f"Evidence acceptance: {public_evidence['acceptance']['passed']} | "
    "Public synthetic reconstruction; no production or confidential data"
)