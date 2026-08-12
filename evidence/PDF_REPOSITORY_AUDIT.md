# Project 05 Claim-to-Evidence Audit

Audit base: `8d59cd3e60b6d57efbf5454c8a3bc75ab3e9e721` plus the current local Project 05 worktree. Public evidence uses independently generated synthetic data only.

## Claim Classification

| Claim | Code / artifact | Verdict | Allowed public wording |
|---|---|---|---|
| Neural classifier participates in skip/run decisions | `models/classification/classifier_model.py`, frozen classifier state, runtime tests | Reproduced | Neural classifier is one input to the frozen OR policy |
| VAE detects unusual signatures | `models/anomaly_detection/vae_model.py`, frozen VAE state, deterministic repeatability test | Reproduced | Deterministic VAE reconstruction score contributes a risk flag |
| Sigma rules provide statistical guardrails | `models/statistical/sigma_rules.py`, aggregate/correlation tests | Reproduced | Per-feature, aggregate robust-z, and correlation guardrails contribute flags |
| Conservative OR logic is used | `deployment/runtime.py`, `tests/test_runtime.py` | Reproduced | Any component flag or lot block forces the optional stage to RUN |
| Up to 15% test-time reduction | 85/15 synthetic cost model | Target | 15% is the theoretical maximum and remains an unmet objective |
| Zero escapees by design | Contradicted by post-freeze confirmation: 6/600 escapes | Unsupported as outcome | Zero observed escapes is an unmet objective; OR logic does not guarantee it |
| End-to-end measured reduction | `evidence/operational_envelope_confirmation.json` | Reproduced | 13.524% simulated reduction on 10,000 disjoint synthetic chips |
| End-to-end safety | Same artifact and exact runtime replay | Reproduced with scope | 99.0% known-mode recall, 6/600 escapes, 1.964% one-sided upper bound |
| OOD safety | 10/10 shifted lots blocked; all chips RUN | Reproduced blocking behavior | Synthetic OOD lots fail closed with no time reduction |
| Production savings or deployment | No representative data, hardware timing, or production run | Unsupported | Local reference runtime only; no production outcome |
| MLflow production registry | The obsolete unvalidated helper was removed; no registry artifact or server is part of the accepted path | Unsupported | Do not claim an active registry or promotion workflow |

## Evidence Chain

| Surface | Canonical identity |
|---|---|
| Dataset | Eight split SHA-256 values in `public_synthetic_dataset_manifest.json`; all replay exactly |
| Classifier | `52da84b9972d3ede2b3f1860f1c07080dfcb74db03882c120a07d7e696ff5c02` |
| VAE | `0fc8a9bb9f9d02cb94d9b56c3a30fc6e4233410ea5e83569640870189ba9d9d2` |
| Bundle | `53ce0e9ccbd63b3c84c581a0dedc325782e8c09b72847977626f41aa6ad3d1fe` |
| Post-freeze predictions | `d5d50071d10de5bf0dfb531b843c49567a42cb170b595cc1b990f4c2a30661db` |
| Validation command | `python scripts/validate_evidence.py` |

## Experiment Integrity

- Train, known-mode development, OOD calibration, confirmation, and post-freeze roles are separate.
- The selected policy maximizes development reduction among eligible hybrid policies under predeclared gates.
- Component ablations are retained but cannot win selection.
- Two predecessor failures and the all-RUN first frozen confirmation remain versioned.
- The post-freeze confirmation performs no model, threshold, or policy selection.

## Current Replay Note

The exact secure lock (`NumPy 1.26.4`, `Torch 2.13.0`) selected the same candidate and reproduced selected validation metrics, both confirmation prediction hashes, and both model state hashes exactly. Low-level floating-point drift-matrix values serialize at slightly different last digits, producing a different derived bundle ID despite identical evaluated decisions. The accepted frozen manifest is retained rather than silently replaced; `evidence/reproducibility_validation.json` records the comparison.

The source hashes embedded in the frozen evaluation identify its original generation files. Current source includes recovered implementation and delivery changes, so those historical source hashes do not match byte for byte. Active acceptance depends on exact model hashes, selected metrics, split identities, and prediction hashes, all of which replay.

## Remaining Gates

- Exact canonical training environment is not fully locked beyond versions recorded in the evaluation artifact.
- Local container execution and browser QA must pass before publication approval.
- Representative physical data, hardware timing, stress lots, and manufacturing review remain external promotion gates.
- Resume wording must keep 13.524% explicitly synthetic and keep 15% / zero escapes as unmet targets.