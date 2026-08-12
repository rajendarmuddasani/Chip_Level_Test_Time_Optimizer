# Model Card: Public Synthetic Hybrid Policy v1

## Identity

| Field | Value |
|---|---|
| Model ID | `public_synthetic_hybrid_v1` |
| Bundle ID | `53ce0e9ccbd63b3c84c581a0dedc325782e8c09b72847977626f41aa6ad3d1fe` |
| Classifier SHA-256 | `52da84b9972d3ede2b3f1860f1c07080dfcb74db03882c120a07d7e696ff5c02` |
| VAE SHA-256 | `0fc8a9bb9f9d02cb94d9b56c3a30fc6e4233410ea5e83569640870189ba9d9d2` |
| Runtime manifest SHA-256 | `7eda203201f9ce650c666ef64cef8e8ebd9f69a64e517a5bad1fa29e32972d29` |

Serving verifies both model files, the evaluation JSON, the dataset manifest, and the derived bundle identity before accepting inference.

## Components

| Component | Frozen configuration | Decision role |
|---|---|---|
| Neural classifier | 32 inputs, 24 hidden units, dropout 0.15, fail threshold 0.20 | Flags likely known failures |
| Beta-VAE | latent size 4, 3 layers, beta 0.02, reconstruction threshold 0.8552235365 | Flags unusual chip-level signatures |
| Sigma guardrails | tail variant 0.00010 plus aggregate robust-z and correlation distance | Flags feature, aggregate, and correlation excursions |
| Lot drift guard | shrinkage precision over lot means, minimum lot size 100 | Blocks shifted or undersized lots |
| Ensemble | logical OR | Runs the optional stage if any component flags |

Missing lot context also forces RUN.

## Selection

The training split fits model parameters. Three known-mode development splits totaling 20,000 chips select the policy. Two separate OOD development splits calibrate lot blocking. Neither confirmation split participates in model or threshold selection.

The catalog contains 198 measured candidates. Only 180 hybrid OR candidates are eligible. Selection maximizes simulated time reduction subject to limits on observed escapes, relative escape rate, one-sided 95% escape uncertainty, and over-test.

Selected candidate: `hybrid_c0.20_vae_q990_sigma_tail_0.00010`.

## Evaluation

### Development selection

| Metric | Value |
|---|---:|
| Chips / failures | 20,000 / 1,200 |
| Simulated reduction | 13.1085% |
| Defect recall | 99.1667% |
| Escapes | 10 |
| Relative escape rate | 0.8333% |
| One-sided 95% escape upper bound | 1.4094% |
| Over-test rate | 7.0851% |

### First frozen confirmation

All 20 lots exceeded the selected drift threshold. The policy correctly failed closed but produced 100% over-test and 0% reduction. This confirmation was rejected on utility and retained as evidence.

### OOD confirmation

All ten novel coupled-drift lots were blocked. All 5,000 chips ran, so recall was 100%, escapes were zero, and reduction was 0%. This tests lot blocking only.

### Post-freeze operational envelope

| Metric | Value |
|---|---:|
| Chips / failures | 10,000 / 600 |
| Simulated reduction | 13.524% |
| Reduction 95% interval | 13.434% to 13.609% |
| Defect recall | 99.0% |
| Escapes | 6 |
| Relative / absolute escape rate | 1.0% / 0.06% |
| One-sided 95% escape upper bound | 1.964% |
| Over-test rate | 4.149% |
| MCC | 0.7563 |

Timing shift was weakest at 98.571% recall. The zero-escape and 15% reduction objectives remain unmet.

## Intended Use

This bundle is a public synthetic demonstration of constrained policy selection, fail-closed lot handling, and model/evidence identity. It supports local CLI, API, dashboard, and container testing.

## Prohibited Interpretation

- It is not production-qualified.
- OR logic does not guarantee zero escapes.
- OOD blocking does not establish novel-defect classification.
- The measured percentage is synthetic and cost-model dependent.
- Local service behavior is not a production SLO.

## Promotion Requirements

Promotion requires approved representative data, leakage and label review, broader stress lots, hardware timing, multisite and retest modeling, independent confirmation, manufacturing risk review, monitored shadow deployment, rollback criteria, and explicit release approval.