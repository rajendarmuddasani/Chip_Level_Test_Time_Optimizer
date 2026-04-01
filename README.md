# Chip-Level Test Time Optimizer

Reduces semiconductor test time by generating per-chip skip/run flags using a hybrid ensemble of neural classification, VAE anomaly detection, and statistical sigma-rule screening. Targets up to 15% test-time reduction while maintaining zero-escapee safety through conservative decision logic.

## Problem

Back-end semiconductor testing is one of the most expensive steps in chip manufacturing. Every chip passes through dozens of test programs, many of which are redundant for chips that show healthy signatures on earlier tests. The challenge is to identify which tests can be safely skipped for each individual chip — without letting a single defective chip escape to the customer.

## Approach

Three independent screening methods vote on each chip. A conservative OR policy ensures that if **any** method flags a chip as risky, the full test suite runs. Only chips cleared by all three methods get the optimized (shorter) test flow.

```
                    ┌─────────────────────┐
  Per-chip          │  Neural Classifier  │──── skip / test ──┐
  feature     ────▶ │  (feedforward NN)   │                   │
  vector            └─────────────────────┘                   │
                    ┌─────────────────────┐                   │   Conservative
                ──▶ │  VAE Anomaly        │──── normal / OOD ─┼──── OR Logic ──▶ SKIP or TEST
                    │  Detector           │                   │
                    └─────────────────────┘                   │
                    ┌─────────────────────┐                   │
                ──▶ │  Sigma-Rule         │──── pass / flag ──┘
                    │  Screening          │
                    └─────────────────────┘
```

| Component | Role |
|-----------|------|
| **Neural classifier** | Feedforward binary classifier trained on labeled pass/fail chip data |
| **VAE anomaly detector** | Variational autoencoder that flags chips with unusual feature distributions |
| **Sigma-rule screening** | Statistical bounds (3σ, 6σ) on key test parameters for interpretable guardrails |
| **Hybrid ensemble** | Conservative OR logic — any flag triggers full testing |

## Repository Structure

```
├── models/
│   ├── classification/    # Feedforward NN classifier
│   ├── anomaly_detection/ # VAE-based anomaly detector
│   ├── statistical/       # Sigma-rule screening
│   └── ensemble.py        # Hybrid ensemble with OR logic
├── preprocessing/         # Feature validation, scaling, outlier detection
├── evaluation/            # Metrics: escapee rate, overreject, test-time savings
├── deployment/            # Flag file generation + MLflow model registry
├── examples/              # Training example script
├── notebooks/             # Exploratory analysis
└── requirements.txt
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Usage

**Training:**
```python
from models.classification.classifier_model import ChipTestClassifier
from models.anomaly_detection.vae_model import VAEAnomalyDetector
from models.ensemble import HybridEnsemble

# Train individual models, then combine
ensemble = HybridEnsemble(classifier=clf, vae=vae, sigma=sigma_rules)
ensemble.save("models/ensemble_v1.pt")
```

**Inference (flag generation):**
```python
from deployment.generate_flags import FlagGenerator

generator = FlagGenerator(model_path="models/ensemble_v1.pt")
flags = generator.generate(chip_features)  # Returns per-chip SKIP/TEST flags
```

## Key Metrics

| Metric | Target |
|--------|--------|
| Test-time reduction | Up to 15% |
| Escapee rate | 0% (by design — conservative OR policy) |
| Overreject rate | Minimized through model calibration |

## Requirements

- Python 3.10+
- PyTorch 2.x
- scikit-learn, MLflow
- See `requirements.txt` for full list

## License

MIT
