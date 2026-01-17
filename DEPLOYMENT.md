# Deployment Guide

This guide covers deploying the DTFS system to production.

## 🎯 Deployment Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Test Data   │────▶│ Preprocessing│────▶│ Ensemble    │
│ (STDF)      │     │ Pipeline     │     │ Inference   │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │ Flag        │
                                          │ Generation  │
                                          └─────────────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │ SORTFILE    │
                                          │ Output      │
                                          └─────────────┘
```

## 📦 Prerequisites

### System Requirements
- Python 3.8+
- 16GB RAM minimum (32GB recommended)
- GPU optional (speeds up inference)
- Linux/Unix environment (LSF cluster for batch processing)

### Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
```bash
# Copy and configure environment variables
cp env.example .env

# Edit .env with your credentials
vim .env
```

## 🚀 Deployment Steps

### 1. Model Preparation

#### Load Models from MLflow Registry
```python
from deployment.mlflow_registry import ModelRegistry

# Initialize registry
registry = ModelRegistry(tracking_uri='http://mlflow-server:5000')

# Load production models
loaded = registry.load_model('TF_ADC_1', stage='Production')
classifier = loaded['model']
scaler = loaded['scaler']
```

#### Alternative: Load from Local Files
```python
import torch
from preprocessing.preprocessing import FeatureScaler

# Load models
classifier = torch.load('models/classifier.pth')
vae = torch.load('models/vae.pth')
scaler = FeatureScaler.load('models/scaler.pkl')
```

### 2. Batch Processing Setup

#### LSF Job Submission
```bash
#!/bin/bash
#BSUB -J dtfs_inference
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 02:00
#BSUB -o logs/inference_%J.out
#BSUB -e logs/inference_%J.err

# Activate environment
source venv/bin/activate

# Run inference
python deployment/generate_flags.py \
    --input data/lot_${LOT_ID}.csv \
    --output sortfiles/DTFS_${LOT_ID}.txt \
    --ensemble-path models/ensemble_v5.pth \
    --scaler-path models/scaler_v5.pkl
```

Submit job:
```bash
bsub < scripts/run_inference.sh
```

### 3. Real-time Inference API (Optional)

#### Flask API Server
```python
# api/server.py
from flask import Flask, request, jsonify
from deployment.generate_flags import FlagGenerator

app = Flask(__name__)

# Load models at startup
generator = FlagGenerator(ensemble, scaler)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    flags = generator.generate_flags(pd.DataFrame(data))
    return jsonify(flags.to_dict())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Start server:
```bash
python api/server.py
```

### 4. Production Monitoring

#### Metrics Collection
```python
from evaluation.metrics import DTFSEvaluator, calculate_production_metrics

# Generate flags
flags_df = generator.generate_flags(data)

# Calculate metrics
metrics = calculate_production_metrics(
    flags_df, 
    test_suite='TF_ADC_1',
    test_time_minutes=30.0
)

# Log to monitoring system
print(f"Skip rate: {metrics['skip_rate']:.2%}")
print(f"Time saved: {metrics['time_saved_hours']:.1f} hours")
```

#### Drift Detection
```python
# Check for distribution drift
from models.drift_detection import MMDDriftDetector

detector = MMDDriftDetector()
drift_score = detector.compute_mmd(X_production, X_training)

if drift_score > threshold:
    logger.warning(f"Drift detected: {drift_score:.4f}")
    # Trigger retraining workflow
```

### 5. Output Format

#### SORTFILE Structure
```
CHIP_00001 1
CHIP_00002 0
CHIP_00003 0
CHIP_00004 1
...
```

- Column 1: Chip identifier
- Column 2: Flag (0=SKIP, 1=TEST)

## 🔄 Update & Rollback

### Deploy New Model Version
```python
# Register new version
registry = ModelRegistry()
new_version = registry.register_model(
    model=new_classifier,
    model_name='TF_ADC_1',
    metadata={'version': '6', 'accuracy': 0.96}
)

# Promote to staging for A/B testing
registry.promote_to_staging('TF_ADC_1', new_version)

# After validation, promote to production
registry.promote_to_production('TF_ADC_1', new_version)
```

### Rollback to Previous Version
```python
# Load previous production version
previous_model = registry.load_model('TF_ADC_1', version=5)

# Promote back to production
registry.promote_to_production('TF_ADC_1', version=5)
```

## 📊 Performance Tuning

### Batch Size Optimization
```python
# Tune batch size for inference
BATCH_SIZE = 4096  # Adjust based on available memory

for i in range(0, len(data), BATCH_SIZE):
    batch = data[i:i+BATCH_SIZE]
    flags = ensemble.predict(batch)
```

### GPU Acceleration
```python
# Enable GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Use mixed precision for faster inference
with torch.cuda.amp.autocast():
    predictions = model(X_tensor)
```

## 🛡️ Security Considerations

1. **Credentials**: Never commit credentials to git
2. **Data Access**: Use read-only database credentials
3. **Output Validation**: Verify sortfile integrity before deployment
4. **Audit Logging**: Log all inference runs with timestamps

## 📝 Troubleshooting

### Common Issues

**Issue**: Out of memory errors
```bash
# Solution: Reduce batch size or increase memory allocation
#BSUB -R "rusage[mem=32GB]"
```

**Issue**: Model loading fails
```python
# Solution: Check model compatibility
print(f"Model trained with PyTorch {torch.__version__}")
```

**Issue**: Inference too slow
```python
# Solution: Use GPU or reduce model complexity
model.to('cuda')
model.eval()
```

## 📞 Support

For deployment issues:
1. Check logs in `logs/` directory
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
3. Open an issue on GitHub
4. Contact maintainers

---

**Next Steps**: Review [WALKTHROUGH.md](WALKTHROUGH.md) for interview preparation
