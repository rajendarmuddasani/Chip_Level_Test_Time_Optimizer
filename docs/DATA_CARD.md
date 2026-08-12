# Data Card: Public Grouped Synthetic Chip Tests

## Summary

Project 05 uses independently generated synthetic data only. No production, customer, employee, proprietary, or externally licensed measurements are included. The generator models a deliberately bounded decision problem: use 32 early-stage features to decide whether a 15-unit optional stage should run after 85 mandatory units.

## Schema

| Group | Features | Intended synthetic meaning |
|---|---:|---|
| Voltage | `V_00` to `V_07` | Early parametric voltage signatures |
| Current | `I_00` to `I_07` | Current and leakage signatures |
| Timing | `T_00` to `T_07` | Timing and frequency signatures |
| Resistance | `R_00` to `R_07` | Resistance and analog signatures |

Each row also has a synthetic `chip_id`, `lot_id`, chronological `time_index`, binary `label`, and generator `failure_mode`. Label and failure mode are evaluation fields and are never accepted by the serving runtime.

## Splits

The canonical benchmark has 58,000 rows across eight isolated splits. A later post-freeze confirmation adds 10,000 rows without changing the model or thresholds.

| Split | Chips | Lots | Failures | Time indices |
|---|---:|---:|---:|---|
| Train | 16,000 | 32 | 960 | 0-31 |
| Validation | 4,000 | 8 | 240 | 40-47 |
| Known-shift validation | 6,000 | 12 | 360 | 60-71 |
| OOD validation | 2,000 | 4 | 400 | 80-83 |
| Known-shift validation 2 | 10,000 | 20 | 600 | 100-119 |
| OOD validation 2 | 5,000 | 10 | 1,000 | 130-139 |
| First frozen confirmation | 10,000 | 20 | 600 | 160-179 |
| OOD confirmation | 5,000 | 10 | 1,000 | 190-199 |
| Post-freeze operational envelope | 10,000 | 20 | 600 | 220-239 |

The manifest asserts zero chip, lot, and time overlap. `scripts/validate_evidence.py` regenerates every benchmark split and compares its canonical SHA-256.

## Failure Modes

Known-mode failures apply shifts to one feature family:

- `voltage_drift`
- `leakage_spike`
- `timing_shift`
- `resistance_bridge`

The OOD challenge uses `novel_coupled_drift`, a synthetic coupled pattern reserved for drift calibration and confirmation. OOD safety is evaluated as lot-level blocking, not as per-chip novel-defect recognition.

## Generation And Isolation

- Generator: `benchmark/synthetic_data.py`
- Generator version: `public_grouped_synthetic_v1`
- Base seed: `20260807`
- Canonical manifest: `evidence/public_synthetic_dataset_manifest.json`
- Post-freeze manifest: embedded in `evidence/operational_envelope_confirmation.json`
- Public input fixture: one unlabelled 500-chip post-freeze lot in `examples/public_synthetic_input.json`

No generated training or labelled confirmation rows are redistributed as a separate dataset. The compact public fixture supports runtime demonstrations only.

## Appropriate Use

- Reproducing bounded classifier/VAE/statistical policy selection.
- Testing escape, over-test, drift-block, and cost-model calculations.
- Demonstrating hash-bound inference, fail-closed validation, and evidence replay.
- Comparing methods under this exact synthetic generator.

## Inappropriate Use

- Estimating real manufacturing yield, test cost, escape risk, or customer quality.
- Claiming coverage of physical failure mechanisms not represented by the generator.
- Training a production disposition model.
- Combining these results with confidential work outcomes as one evidence chain.

## Limitations

The generator omits tester noise, multisite interactions, retest, handler effects, package and environmental variation, latent defects, label errors, hardware timing, and changing test content. Correlation and drift are simplified. Strong performance here may not transfer to any physical process.

## License And Privacy

The generator is MIT-licensed repository code. The rows are independently generated and contain no personal or confidential data. Secret and provenance scans must still run before publication because operational logs or local `.env` files are outside this data contract.