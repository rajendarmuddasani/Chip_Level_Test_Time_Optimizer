"""Canonical evidence identities and claim ledger replay."""

import json
from pathlib import Path

from scripts.validate_evidence import validate


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_evidence_replays_exactly():
    report = validate()

    assert report["status"] == "passed", report["errors"]
    assert all(report["dataset_split_checks"].values())
    assert report["operational_metrics_replayed"] is True
    assert all(report["claim_checks"].values())
    assert report["operational_prediction_sha256"] == (
        "d5d50071d10de5bf0dfb531b843c49567a42cb170b595cc1b990f4c2a30661db"
    )


def test_local_api_load_is_bounded_and_not_a_production_slo():
    load = json.loads(
        (ROOT / "evidence" / "local_api_load.json").read_text(encoding="utf-8")
    )
    claims = json.loads(
        (ROOT / "evidence" / "claims.json").read_text(encoding="utf-8")
    )
    load_claim = next(
        claim for claim in claims["claims"] if claim["id"] == "local_api_load"
    )

    assert load["scope"] == "local bounded API measurement; not a production SLO"
    assert load["configuration"] == {
        "url": "http://127.0.0.1:8015/v1/predict",
        "requests": 100,
        "concurrency": 5,
        "records_per_request": 1,
    }
    assert load["results"]["successes"] == 100
    assert load["results"]["failures"] == 0
    assert load["results"]["bundle_ids"] == [
        "53ce0e9ccbd63b3c84c581a0dedc325782e8c09b72847977626f41aa6ad3d1fe"
    ]
    assert load_claim["value"]["latency_ms"] == {
        percentile: load["results"]["latency_ms"][percentile]
        for percentile in ("p50", "p95", "p99")
    }
    assert "not a production SLO" in load_claim["allowed_wording"]


def test_secure_lock_reproduces_behavior_with_disclosed_float_serialization():
    replay = json.loads(
        (ROOT / "evidence" / "reproducibility_validation.json").read_text(
            encoding="utf-8"
        )
    )["training_replay"]

    assert replay["behavioral_identity_passed"] is True
    assert all(replay["checks"].values())
    assert replay["bundle_id_matches"] is False
    assert replay["runtime_config_difference_summary"]["count"] == 1026
    assert (
        replay["runtime_config_difference_summary"][
            "maximum_absolute_numeric_difference"
        ]
        < 3e-7
    )
    assert replay["replay_environment"]["numpy"] == "1.26.4"
    assert replay["replay_environment"]["torch"] == "2.13.0+cpu"