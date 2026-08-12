"""Authenticated HTTP service for the frozen chip test policy."""

from __future__ import annotations

import hmac
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from deployment.runtime import (
    ArtifactIntegrityError,
    HybridPolicyRuntime,
    InputValidationError,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "artifacts" / "public_v1" / "runtime_manifest.json"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class PredictionRequest(BaseModel):
    """A bounded batch of public-schema chip measurements."""

    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)


class ServiceMetrics:
    """Thread-safe process-local request counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = Counter()
        self.latency_seconds = Counter()

    def observe(self, method: str, path: str, response_status: int, latency: float) -> None:
        key = (method, path, str(response_status))
        with self._lock:
            self.requests[key] += 1
            self.latency_seconds[(method, path)] += latency

    def render(self) -> str:
        lines = [
            "# HELP chip_optimizer_http_requests_total HTTP requests by endpoint.",
            "# TYPE chip_optimizer_http_requests_total counter",
        ]
        with self._lock:
            for (method, path, response_status), count in sorted(self.requests.items()):
                lines.append(
                    "chip_optimizer_http_requests_total"
                    f'{{method="{method}",path="{path}",status="{response_status}"}} {count}'
                )
            lines.extend(
                [
                    "# HELP chip_optimizer_http_latency_seconds_total "
                    "Cumulative request latency by endpoint.",
                    "# TYPE chip_optimizer_http_latency_seconds_total counter",
                ]
            )
            for (method, path), latency in sorted(self.latency_seconds.items()):
                lines.append(
                    "chip_optimizer_http_latency_seconds_total"
                    f'{{method="{method}",path="{path}"}} {latency:.9f}'
                )
        return "\n".join(lines) + "\n"


app = FastAPI(
    title="Chip Test Policy API",
    version="1.0.0",
    description="Hash-bound inference for the public synthetic policy bundle.",
)
metrics = ServiceMetrics()
_runtime: HybridPolicyRuntime | None = None
_runtime_lock = threading.Lock()


def _manifest_path() -> Path:
    configured = os.environ.get("CHIP_OPTIMIZER_MANIFEST")
    return Path(configured) if configured else DEFAULT_MANIFEST


def get_runtime() -> HybridPolicyRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = HybridPolicyRuntime(_manifest_path())
    return _runtime


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("CHIP_OPTIMIZER_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured",
        )
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def _prediction_summary(runtime: HybridPolicyRuntime, frame: pd.DataFrame) -> dict:
    total = len(frame)
    skipped = int((frame["flag"] == 0).sum())
    cost_model = runtime.config["cost_model"]
    optional_fraction = cost_model["optional_stage_units"] / (
        cost_model["early_stage_units"] + cost_model["optional_stage_units"]
    )
    return {
        "total_chips": total,
        "skip_count": skipped,
        "run_count": total - skipped,
        "skip_rate": skipped / total,
        "simulated_time_reduction_percent": skipped / total * optional_fraction * 100,
        "blocked_chips": int(frame["lot_drift_blocked"].sum()),
        "model_id": runtime.config["model_id"],
        "bundle_id": runtime.manifest["bundle_id"],
        "scope": "local policy output; not a production outcome",
    }


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    supplied_request_id = request.headers.get("X-Request-ID", "")
    request_id = (
        supplied_request_id
        if REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
        else str(uuid4())
    )
    started = time.perf_counter()
    response: Response = await call_next(request)
    latency = time.perf_counter() - started
    response.headers["X-Request-ID"] = request_id
    metrics.observe(request.method, request.url.path, response.status_code, latency)
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict:
    try:
        runtime = get_runtime()
    except (ArtifactIntegrityError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Runtime is not ready: {error}",
        ) from error
    return {
        "status": "ready",
        "model_id": runtime.config["model_id"],
        "bundle_id": runtime.manifest["bundle_id"],
    }


@app.post("/v1/predict", dependencies=[Depends(require_api_key)])
def predict(payload: PredictionRequest) -> dict:
    try:
        runtime = get_runtime()
        predictions = runtime.predict_records(payload.records)
        frame = pd.DataFrame.from_records(predictions)
    except InputValidationError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
    except (ArtifactIntegrityError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Runtime unavailable: {error}",
        ) from error
    return {
        "summary": _prediction_summary(runtime, frame),
        "records": _records_for_json(frame),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    return metrics.render()