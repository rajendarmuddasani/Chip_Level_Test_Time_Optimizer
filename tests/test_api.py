"""Authenticated API contracts for the frozen policy runtime."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from deployment.api import app


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples" / "public_synthetic_input.json"


class ASGITestClient:
    def request(self, method: str, path: str, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.request("POST", path, **kwargs)


@pytest.fixture(scope="module")
def records():
    return json.loads(INPUT.read_text(encoding="utf-8"))["records"]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHIP_OPTIMIZER_API_KEY", "test-only-key")
    return ASGITestClient()


def test_health_and_readiness_expose_bundle_identity(client):
    health = client.get("/health", headers={"X-Request-ID": "health-check-1"})
    ready = client.get("/ready")

    assert health.status_code == 200
    assert health.json() == {"status": "alive"}
    assert health.headers["X-Request-ID"] == "health-check-1"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["bundle_id"] == (
        "53ce0e9ccbd63b3c84c581a0dedc325782e8c09b72847977626f41aa6ad3d1fe"
    )


def test_predict_requires_api_key(client, records):
    response = client.post("/v1/predict", json={"records": records[:1]})

    assert response.status_code == 401


def test_predict_replays_public_fixture(client, records):
    response = client.post(
        "/v1/predict",
        json={"records": records},
        headers={"X-API-Key": "test-only-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total_chips"] == 500
    assert payload["summary"]["skip_count"] == 450
    assert payload["summary"]["run_count"] == 50
    assert payload["summary"]["simulated_time_reduction_percent"] == pytest.approx(
        13.5
    )
    assert {record["bundle_id"] for record in payload["records"]} == {
        payload["summary"]["bundle_id"]
    }


def test_predict_without_lot_context_fails_closed(client, records):
    no_lot = [
        {key: value for key, value in record.items() if key != "lot_id"}
        for record in records[:5]
    ]

    response = client.post(
        "/v1/predict",
        json={"records": no_lot},
        headers={"X-API-Key": "test-only-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert {record["decision"] for record in payload["records"]} == {"RUN"}
    assert {record["lot_drift_reason"] for record in payload["records"]} == {
        "insufficient_lot_context"
    }
    assert {record["lot_drift_score"] for record in payload["records"]} == {None}


def test_predict_rejects_missing_feature(client, records):
    invalid = dict(records[0])
    invalid.pop("V_00")

    response = client.post(
        "/v1/predict",
        json={"records": [invalid]},
        headers={"X-API-Key": "test-only-key"},
    )

    assert response.status_code == 422
    assert "Missing features" in response.json()["detail"]


def test_metrics_are_prometheus_parseable_text(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "chip_optimizer_http_requests_total" in response.text
    assert 'path="/v1/predict"' in response.text