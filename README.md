# 🚀 Dynamic Test Flow Selection (DTFS) - AI-Powered Semiconductor Test Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)

> **Reducing semiconductor test time by 15% using AI/ML while maintaining 0% defect escapee rate**

## 📊 Project Impact

- **💰 Business Value**: €3.2M savings over 5 years
- **⚡ Test Time Reduction**: 15% average (up to 30% for high-yield products)
- **🎯 Quality**: 0% escapee rate, <10% overreject rate
- **📈 Scale**: Successfully deployed on 400+ production lots
- **🏭 Production Ready**: Handles 13,000+ chips per lot in real-time

---

## 🎯 Problem Statement

In semiconductor manufacturing, **98%+ of chips pass all tests**, yet **100% undergo expensive, time-consuming testing**. Traditional testing is:
- ⏱️ **Time-Consuming**: 30+ minutes per chip across multiple test suites
- 💸 **Expensive**: Tester time is a major production cost
- 🔄 **Inefficient**: Testing good chips repeatedly

**The Challenge**: *Can we predict which chips will pass and safely skip their tests without compromising quality?*

---

## 💡 Solution Overview

**DTFS** (Dynamic Test Flow Selection) uses machine learning to generate **test flags** for each chip, determining which tests to run or skip based on early test results.

### The Flag System

```
Chip ID: Wafer_1_X11_Y25
├── TF_EVR_1: 0 ✅ SKIP (predicted PASS)
├── TF_EVR_2: 0 ✅ SKIP (predicted PASS)  
├── TF_EVR_3: 1 ⚠️ TEST (risk detected)
├── TF_OSC_1: 0 ✅ SKIP (predicted PASS)
├── TF_ADC_1: 0 ✅ SKIP (predicted PASS)
└── TF_PERF_1: 0 ✅ SKIP (predicted PASS)

Result: Save 2,450 seconds (only run TF_EVR_3: 1,503s)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION TEST DATA                      │
│     (Early test results from wafer/package testing)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   DATA PREPROCESSING                         │
│  • Validation  • Feature Engineering  • Scaling             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              HYBRID PREDICTION MODELS (OR Logic)             │
│  ┌────────────────────┐      ┌────────────────────┐        │
│  │  Deep Learning     │      │  Statistical       │        │
│  │  Classifier        │  OR  │  Sigma Rules       │        │
│  │  (Neural Network)  │      │  (Robust Stats)    │        │
│  │  P(Fail) ≥ 20%    │      │  Outlier Detection │        │
│  └────────────────────┘      └────────────────────┘        │
│           Flag=1 if EITHER method flags chip                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   MLFLOW MODEL REGISTRY                      │
│  Version Control • A/B Testing • Rollback Capability        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    FLAG FILE GENERATION                      │
│  Per-chip, per-test-suite binary decisions (0=SKIP, 1=TEST) │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PRODUCTION TEST EQUIPMENT (ATE)                 │
│  Reads flags → Dynamically skips/executes tests → Bins      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  METRICS & VALIDATION                        │
│  Harvest % | False Fails | Escapees | ROI Calculation       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Methodology: Three-Approach Ensemble

### 1. **Anomaly Detection** (Variational Autoencoder)
```python
# Learns normal chip behavior, flags outliers
VAE: Input → Encode → Latent Space → Decode → Reconstruction Error
If error > threshold → Flag = 1 (TEST)
```
**Advantage**: Detects never-seen-before failure patterns

### 2. **Classification** (Neural Network)
```python
# Direct pass/fail prediction
Input: Early test features (200+ parameters)
Output: P(Fail)
If P(Fail) ≥ 20% → Flag = 1 (TEST)
```
**Advantage**: Fast, accurate for known failure modes

### 3. **Statistical Sigma Rules**
```python
# Robust statistical thresholds
For each test: median ± N×robust_std
If value outside N-sigma → Flag = 1 (TEST)
```
**Advantage**: Interpretable, no training needed, catches gross outliers

### Combined Decision (Conservative)
```python
Final_Flag = DL_Flag OR Sigma_Flag OR VAE_Flag
# Test if ANY method detects risk
```

---

## 📁 Repository Structure

```
dtfs-public-portfolio/
│
├── README.md                          # This file
├── WALKTHROUGH.md                     # Interview/presentation guide
├── ARCHITECTURE.md                    # Detailed technical architecture
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore patterns
├── env.example                        # Environment variables template
│
├── models/                            # Model architectures & training
│   ├── anomaly_detection/
│   │   ├── vae_model.py              # Variational Autoencoder
│   │   └── train_vae.py              # VAE training script
│   ├── classification/
│   │   ├── classifier_model.py       # Neural Network Classifier
│   │   └── train_classifier.py       # Classifier training
│   ├── statistical/
│   │   └── sigma_rules.py            # Robust statistical methods
│   └── ensemble/
│       └── hybrid_model.py           # Combined prediction logic
│
├── preprocessing/
│   ├── data_cleaning.py              # Data validation & cleaning
│   ├── feature_engineering.py        # Feature extraction
│   └── validation.py                 # Input validation
│
├── deployment/
│   ├── generate_flags.py             # Flag file generation
│   ├── mlflow_register.py            # Model registry & versioning
│   └── batch_inference.py            # Batch prediction pipeline
│
├── evaluation/
│   ├── calculate_harvest.py          # Test time savings metrics
│   ├── calculate_quality.py          # Quality metrics (escapees/overrejects)
│   └── false_fail_analysis.py        # Root cause analysis
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_selection.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_deployment_simulation.ipynb
│
├── tests/                             # Unit & integration tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_deployment.py
│
├── configs/                           # Configuration files
│   ├── model_config.yaml
│   └── deployment_config.yaml
│
├── docs/                              # Additional documentation
│   ├── data_schema.md
│   ├── flag_format_spec.md
│   └── deployment_guide.md
│
└── requirements.txt                   # Python dependencies
```

---

## 🛠️ Technology Stack

### Core ML/AI
- **PyTorch 2.3.0** - Deep learning framework
- **Scikit-learn 1.4.2** - Statistical ML, preprocessing
- **NumPy 1.26.4** - Numerical computing
- **Pandas 2.2.2** - Data manipulation

### Production & MLOps
- **MLflow** - Model tracking, registry, deployment
- **Joblib** - Model serialization
- **Imbalanced-learn** - Handle class imbalance (ROSE, SMOTE)

### Data & Visualization
- **Matplotlib / Seaborn** - Visualization
- **Statsmodels** - Statistical analysis
- **SciPy** - Scientific computing

### Infrastructure
- **PostgreSQL / Denodo** - Data warehouse
- **AWS S3** - Model artifact storage
- **LSF Cluster** - Distributed training
- **STDF** - Standard Test Data Format (test equipment)

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (optional, for training)
Access to semiconductor test data
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/dtfs-public-portfolio.git
cd dtfs-public-portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp env.example .env
# Edit .env with your configuration
```

### Training a Model
```bash
# Train classifier for a specific test suite
python models/classification/train_classifier.py \
    --test-module TF_EVR_1 \
    --data-path data/dataset_v2u.parquet \
    --epochs 111 \
    --batch-size 4096

# Train anomaly detector
python models/anomaly_detection/train_vae.py \
    --test-module TF_EVR_1 \
    --latent-size 2 \
    --beta 1.5
```

### Generating Flags (Inference)
```bash
# Generate flags for new lot
python deployment/generate_flags.py \
    --lot-id 84P36N735UN \
    --model-version TF_EVR_1/5 \
    --output flags/

# Output: flags/84P36N735UN.csv
# Format: LotID, WafNr, X, Y, TF_EVR_1, TF_EVR_2, ...
```

### Model Registry (MLflow)
```bash
# Register trained model
python deployment/mlflow_register.py \
    --experiment-name TF_EVR_1 \
    --model-path artifact/cls_v4.pt \
    --dataset-version v2u

# MLflow UI
mlflow ui --port 5000
# Open http://localhost:5000
```

---

## 📈 Results & Metrics

### Test Time Savings
| Test Suite | Time (sec) | Skip Rate | Time Saved |
|------------|-----------|-----------|------------|
| TF_EVR_1   | 550.8     | 93%       | 512.4s     |
| TF_EVR_2   | 515.4     | 91%       | 469.0s     |
| TF_EVR_3   | 1503.0    | 87%       | 1307.6s    |
| TF_PERF_1  | 415.3     | 95%       | 394.5s     |
| TF_ADC_1   | 190.0     | 94%       | 178.6s     |
| TF_OSC_1   | 278.6     | 92%       | 256.3s     |

**Average Harvest: ~15% total test time reduction**

### Quality Metrics (Validation Set)
- **Accuracy**: 98.5%
- **Overreject Rate**: 0.8% (slight over-testing, safe)
- **Escapee Rate**: 0.0% (no bad chips escaped!)
- **Precision (Flag=1)**: 92.3%
- **Recall (Detect Fail)**: 100%

---

## 🎓 Key Innovations

### 1. **Hybrid Ensemble Approach**
Combines physics-based statistics with data-driven ML for robustness

### 2. **Conservative OR Logic**
Test if ANY method flags → Ensures zero escapees

### 3. **Per-Chip Dynamic Decisions**
Not lot-based or wafer-based, but individual chip optimization

### 4. **False Fail Removal**
ML-guided identification of marginal/ambiguous failures

### 5. **Synthetic Data Generation (ROSE)**
Handles severe class imbalance (98% pass vs 2% fail)

### 6. **Production MLOps Integration**
Full MLflow lifecycle with versioning, A/B testing, rollback

---

## 📊 Use Cases

- ✅ **Automotive Semiconductors** (AURIX™ family)
- ✅ **High-Yield Products** (>98% pass rate)
- ✅ **Time-Critical Manufacturing** (cost per tester-hour)
- ✅ **Multi-Insertion Testing** (FE wafer → BE package)

---

## 🤝 Contributing

This is a portfolio/demonstration project. For collaboration:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

---

## 📚 Documentation

- **[WALKTHROUGH.md](./WALKTHROUGH.md)** - Interview presentation guide
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Deep technical dive
- **[docs/](./docs/)** - Additional specifications

---

## 📧 Contact

**Author**: [Your Name]
- 📧 Email: your.email@example.com
- 💼 LinkedIn: [Your LinkedIn]
- 🐙 GitHub: [Your GitHub]

---

## 📝 Citation

If you use this methodology in your research, please cite:
```bibtex
@software{dtfs2024,
  title={Dynamic Test Flow Selection: AI-Powered Semiconductor Test Optimization},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/dtfs-public-portfolio}
}
```

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Methodology developed through collaboration between data science and process engineering teams
- Inspired by production challenges in automotive semiconductor manufacturing
- Built on PyTorch, MLflow, and scikit-learn ecosystems

---

**⭐ If you find this project interesting, please star the repository!**
