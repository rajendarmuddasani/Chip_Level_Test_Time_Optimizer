# Deployment Reference

## Scope

The deployment surfaces execute only the frozen `public_synthetic_hybrid_v1` bundle. They are local reference implementations, not a production manufacturing release.

## Runtime Integrity

`deployment/runtime.py` verifies:

1. classifier and VAE SHA-256 values;
2. canonical evaluation and dataset-manifest SHA-256 values;
3. the bundle identity derived from runtime configuration and model hashes;
4. the exact 32-feature schema;
5. finite numeric values, unique request chip IDs, and a 10,000-chip batch limit.

Artifact or evidence mismatch prevents readiness and inference.

## CLI

```powershell
python deployment/generate_flags.py `
  --input examples/public_synthetic_input.json `
  --output tmp/predictions.json
```

Input may be CSV or JSON. JSON is either a list of records or an object with a `records` list. Output may be detailed CSV, standards-compliant JSON, or a minimal `chip_id flag` sortfile. Missing lot context forces every decision to RUN.

## API

Set a non-empty key and bind locally:

```powershell
$env:CHIP_OPTIMIZER_API_KEY = "choose-a-local-key"
python -m uvicorn deployment.api:app --host 127.0.0.1 --port 8005
```

| Endpoint | Authentication | Purpose |
|---|---|---|
| `GET /health` | None | Process liveness only |
| `GET /ready` | None | Hash-bound model readiness and identity |
| `POST /v1/predict` | `X-API-Key` | Bounded batch inference |
| `GET /metrics` | None | Process-local Prometheus-format request counters |

The API refuses inference with HTTP 503 when `CHIP_OPTIMIZER_API_KEY` is not configured. Invalid keys receive HTTP 401. Invalid data receives HTTP 422. A caller-supplied `X-Request-ID` is accepted only when it contains 1-128 letters, digits, periods, underscores, or hyphens; otherwise a UUID is generated.

Example request body:

```json
{
  "records": [
    {
      "chip_id": "PUBLIC_LOT_001_CHIP_0001",
      "lot_id": "PUBLIC_LOT_001",
      "V_00": 0.1,
      "V_01": 0.2
    }
  ]
}
```

The abbreviated example shows envelope shape only. All 32 canonical features are required; use `examples/public_synthetic_input.json` for an executable request.

## Dashboard

```powershell
python -m streamlit run app.py --server.port 8505
```

The dashboard loads the same hash-bound runtime. It defaults to the public 500-chip lot, accepts CSV/JSON uploads, displays component contributions, exposes downloadable detailed decisions, and separates accepted post-freeze evidence from rejected confirmations.

## Containers

```powershell
$env:CHIP_OPTIMIZER_API_KEY = "choose-a-local-key"
docker compose up --build
```

| Service | Local URL | Container behavior |
|---|---|---|
| API | `http://127.0.0.1:8005/docs` | Digest-pinned Distroless runtime with CPython 3.12, non-root UID 65532, read-only root, dropped capabilities |
| Dashboard | `http://127.0.0.1:8505` | Digest-pinned Distroless runtime with CPython 3.12, non-root UID 65532, read-only root, dropped capabilities |

Both services receive a bounded writable `/tmp` filesystem. The API key is supplied at runtime and is never embedded in the image or repository.

CI builds the digest-pinned Distroless image, fails on high or critical Trivy findings, verifies UID 65532, and then runs authenticated API plus dashboard health smoke tests.

## Fail-Closed Rules

- Missing, extra, non-numeric, NaN, or infinite features: reject request.
- Duplicate chip IDs: reject request.
- More than 10,000 records: reject request.
- Missing lot ID: RUN all chips.
- Fewer than 100 chips in a lot: RUN that lot.
- Lot drift above the frozen threshold: RUN that lot.
- Artifact, evidence, or bundle mismatch: fail readiness and inference.

## Monitoring Boundary

`/metrics` records request count and cumulative latency by method, path, and response status. It does not log features, chip IDs, API keys, or predictions. These counters are process-local and reset on restart; no production persistence, alerting, or SLO is claimed.

## Local Load Evidence

`scripts/run_api_load.py` sends a bounded number of authenticated requests and stores only aggregate timing, status, environment, and bundle identity. The accepted local artifact used 100 one-chip requests at concurrency five on Windows CPU:

| Measure | Local result |
|---|---:|
| Successful requests | 100 / 100 |
| Throughput | 10.22 requests/s |
| Latency p50 | 470.50 ms |
| Latency p95 | 565.37 ms |
| Latency p99 | 721.73 ms |

Canonical artifact: `evidence/local_api_load.json`. This is not a production SLO and does not represent batch throughput, multi-worker deployment, network distance, or sustained load.

## Promotion Checklist

- Replace synthetic-only evaluation with approved representative data.
- Run hardware timing, concurrency, soak, and recovery tests.
- Integrate managed secret storage, TLS termination, rate limiting, and audited identity.
- Define drift response, human review, rollback, and incident ownership.
- Obtain manufacturing, product-quality, security, privacy, and release approval.