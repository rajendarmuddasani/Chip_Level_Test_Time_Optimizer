"""Run a bounded authenticated local API load measurement."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def run_load(
    url: str,
    api_key: str,
    requests: int,
    concurrency: int,
    records_per_request: int,
) -> dict:
    fixture = json.loads(
        (REPOSITORY_ROOT / "examples" / "public_synthetic_input.json").read_text(
            encoding="utf-8"
        )
    )
    if not 1 <= records_per_request <= len(fixture["records"]):
        raise ValueError("records-per-request must be between 1 and 500")
    body = {"records": fixture["records"][:records_per_request]}

    def send(index: int) -> dict:
        started = time.perf_counter()
        try:
            response = httpx.post(
                url,
                json=body,
                headers={
                    "X-API-Key": api_key,
                    "X-Request-ID": f"load-{index:05d}",
                },
                timeout=30.0,
            )
            latency = time.perf_counter() - started
            payload = response.json() if response.status_code == 200 else None
            return {
                "status": response.status_code,
                "latency_seconds": latency,
                "bundle_id": payload["summary"]["bundle_id"] if payload else None,
            }
        except (httpx.HTTPError, ValueError, KeyError):
            return {
                "status": 0,
                "latency_seconds": time.perf_counter() - started,
                "bundle_id": None,
            }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send, index) for index in range(requests)]
        results = [future.result() for future in as_completed(futures)]
    elapsed = time.perf_counter() - started

    successes = [result for result in results if result["status"] == 200]
    latencies = [result["latency_seconds"] * 1000.0 for result in successes]
    statuses = {}
    for result in results:
        status = str(result["status"])
        statuses[status] = statuses.get(status, 0) + 1
    bundles = sorted(
        {result["bundle_id"] for result in successes if result["bundle_id"]}
    )
    return {
        "schema_version": 1,
        "scope": "local bounded API measurement; not a production SLO",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            "python scripts/run_api_load.py "
            f"--url {url} --requests {requests} --concurrency {concurrency} "
            f"--records-per-request {records_per_request}"
        ),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported by operating system",
            "python": platform.python_version(),
        },
        "configuration": {
            "url": url,
            "requests": requests,
            "concurrency": concurrency,
            "records_per_request": records_per_request,
        },
        "results": {
            "successes": len(successes),
            "failures": requests - len(successes),
            "status_counts": dict(sorted(statuses.items())),
            "elapsed_seconds": elapsed,
            "requests_per_second": len(successes) / elapsed if elapsed else 0.0,
            "chips_per_second": (
                len(successes) * records_per_request / elapsed if elapsed else 0.0
            ),
            "latency_ms": {
                "p50": percentile(latencies, 50) if latencies else None,
                "p95": percentile(latencies, 95) if latencies else None,
                "p99": percentile(latencies, 99) if latencies else None,
                "mean": statistics.fmean(latencies) if latencies else None,
            },
            "bundle_ids": bundles,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8015/v1/predict")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--records-per-request", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "evidence" / "local_api_load.json",
    )
    args = parser.parse_args()
    if not 1 <= args.requests <= 10_000:
        parser.error("requests must be between 1 and 10,000")
    if not 1 <= args.concurrency <= 100:
        parser.error("concurrency must be between 1 and 100")

    report = run_load(
        args.url,
        args.api_key,
        args.requests,
        args.concurrency,
        args.records_per_request,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["results"], indent=2, sort_keys=True))
    return 0 if report["results"]["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())