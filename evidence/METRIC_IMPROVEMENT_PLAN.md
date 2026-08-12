# Metric Improvement Plan

## Current Accepted Boundary

The frozen post-freeze result is 13.524% simulated reduction, 99.0% known-mode recall, 6/600 escaped failures, a 1.964% one-sided 95% relative escape upper bound, and 4.149% over-test on 10,000 disjoint synthetic chips. Timing shift is weakest at 98.571% recall.

The 15% reduction and zero observed escape objectives are unmet.

## Optimization Rule

Future work must maximize validation simulated time reduction subject to safety gates. It must not select by repeatedly inspecting the current post-freeze confirmation. Any changed generator, model, threshold, cost model, or drift rule requires new development splits and a newly sealed confirmation range.

## Priority Experiments

| Priority | Experiment | Selection evidence | Safety requirement |
|---:|---|---|---|
| 1 | Add timing-shift boundary cases to development only | Timing recall and calibrated score distribution | Do not alter current confirmation labels or thresholds after opening |
| 2 | Calibrate classifier probabilities on a dedicated calibration split | Brier score, ECE, PR-AUC, gate-qualified reduction | No increase in relative or upper-bound escape risk |
| 3 | Compare monotonic or cost-sensitive tree baseline | Same 32 features and split roles | Must pass every gate and artifact-identity check |
| 4 | Evaluate conformal or abstention policy | Coverage versus escape upper bound | Abstention means RUN, never unchecked SKIP |
| 5 | Stress lot-size and mixed-drift behavior | Block detection, false block rate, recovery | Missing/undersized context remains fail closed |
| 6 | Replace fixed 85/15 assumption with measured stage timing | Hardware p50/p95/p99 by lot and site | Report setup, multisite, retest, and queueing separately |

## Data Needs

- Approved representative early-test and optional-stage outcomes.
- Independent lot, wafer, time, package, site, tester, and product grouping fields.
- Label provenance, retest policy, disposition outcome, and known label uncertainty.
- Explicit rare and latent failure coverage, including failure modes absent from development.
- Hardware stage-time traces and tester resource envelope.

No confidential data may enter the public repository. A private work outcome and this synthetic reconstruction must retain separate evidence chains.

## Required Metrics

- Relative and absolute escape rate, observed escapes, and one-sided confidence bound.
- Per-failure-mode recall and support, with weakest behavior displayed first.
- Over-test and false-stop counts and rates.
- Simulated and hardware-measured time reduction kept separate.
- MCC, F1, PR-AUC, ROC-AUC, Brier score, and ECE for relevant scoring components.
- Lot drift true-block and false-block rates, minimum lot size, and blocked utility.
- Inference p50/p95/p99, throughput, concurrency, memory, startup, and recovery under a declared machine envelope.

## Promotion Gates

1. Freeze generator/data version, split manifest, seeds, environment, and objective.
2. Select on development and calibration data only.
3. Require every safety gate plus artifact/evidence replay.
4. Open one new chronological confirmation exactly once.
5. Reject candidates that improve reduction by weakening safety or hiding blocked utility.
6. Run shadow-mode hardware validation with human review and rollback.
7. Obtain manufacturing, product-quality, security, privacy, and release approval.

## Stopping Rule

If no candidate improves gate-qualified validation reduction, retain the frozen bundle and record the no-improvement result. If a new confirmation fails, preserve it and reopen development with new data; do not tune on that confirmation.