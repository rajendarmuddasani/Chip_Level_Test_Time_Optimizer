# Evidence Quick Start

This guide uses only the frozen public synthetic Project 05 evidence. It does not generate random predictions or project production cost savings.

## Environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-test.txt
python -m pip check
```

## Validate Canonical Evidence

```powershell
python scripts/validate_evidence.py
```

The validator:

- regenerates all eight benchmark split hashes;
- verifies classifier, VAE, evaluation, dataset, and bundle identities;
- replays the 10,000-chip post-freeze policy;
- compares the canonical prediction SHA-256;
- checks public claim values and unmet-target wording.

Expected accepted result:

- 13.524% simulated optional-stage reduction;
- 99.0% defect recall;
- six escapes among 600 failures;
- 4.149% over-test;
- timing shift weakest at 98.571% recall.

## Run The Evidence Walkthrough

```powershell
python notebooks/01_train_and_evaluate.py
```

This reads canonical JSON, shows the meaningful candidate comparison, explains the rejected all-RUN confirmation, executes the frozen model on the public 500-chip lot, and runs exact evidence replay.

## Reproduce Training And Selection

Write a disposable replay under ignored `tmp/`:

```powershell
python scripts/run_public_benchmark.py --output-root tmp/replay
python scripts/validate_evidence.py --replay-root tmp/replay
```

The secure lock (`NumPy 1.26.4`, `Torch 2.13.0`) reproduces the selected candidate, selected metrics, both model hashes, and both frozen confirmation prediction hashes. Low-level floating-point drift-matrix values serialize at slightly different last digits, so the derived bundle ID differs even though evaluated decisions are identical. The accepted frozen manifest remains unchanged and the difference is recorded in `evidence/reproducibility_validation.json`.

## Generate Decisions

```powershell
python deployment/generate_flags.py `
  --input examples/public_synthetic_input.json `
  --output tmp/predictions.json
```

Expected public fixture result: 500 chips, 450 SKIP, 50 RUN, and 13.5% simulated reduction under the 85/15 cost model.

## API And Dashboard

```powershell
$env:CHIP_OPTIMIZER_API_KEY = "choose-a-local-key"
python -m uvicorn deployment.api:app --host 127.0.0.1 --port 8005
```

```powershell
python -m streamlit run app.py --server.port 8505
```

The API refuses unauthenticated inference. The dashboard uses the same hash-bound runtime and keeps accepted evidence separate from failed trials.

## Truth Boundary

- Data is independently generated synthetic data only.
- 15% reduction and zero observed escapes are unmet objectives.
- OR logic does not guarantee zero escapes.
- OOD lots fail closed by running every chip, so blocked lots receive 0% reduction.
- No physical-defect coverage, production savings, real ATE timing, or deployment approval is established.
