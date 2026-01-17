# 🏛️ DTFS Technical Architecture

> **Detailed technical architecture and design decisions**

---

## 📑 Table of Contents

1. [System Overview](#system-overview)
2. [Data Architecture](#data-architecture)
3. [Model Architecture](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Inference Pipeline](#inference-pipeline)
6. [MLOps Infrastructure](#mlops-infrastructure)
7. [Performance Optimization](#performance-optimization)
8. [Security & Compliance](#security-compliance)

---

## <a name="system-overview"></a>🔭 System Overview

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Denodo DWH  │  │ PostgreSQL  │  │ Test Equip  │             │
│  │ (Historical)│  │ (Metadata)  │  │ (Real-time) │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
└─────────┼─────────────────┼─────────────────┼────────────────────┘
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │    DATA PREPROCESSING LAYER       │
          │  • Validation • Cleaning          │
          │  • Feature Engineering • Scaling  │
          └─────────────────┬─────────────────┘
                            │
          ┌─────────────────▼─────────────────┐
          │      MODEL SERVING LAYER          │
          │  ┌──────────┐  ┌──────────┐      │
          │  │ DL Model │  │ Sigma    │      │
          │  │ (PyTorch)│  │ Rules    │      │
          │  └────┬─────┘  └────┬─────┘      │
          │       └──────┬───────┘            │
          │              │ OR Logic           │
          └──────────────┬───────────────────┘
                         │
          ┌──────────────▼───────────────────┐
          │    MLFLOW MODEL REGISTRY         │
          │  • Versioning • A/B Testing      │
          │  • Artifacts (S3) • Monitoring   │
          └──────────────┬───────────────────┘
                         │
          ┌──────────────▼───────────────────┐
          │    FLAG GENERATION SERVICE       │
          │  • Batch Processing              │
          │  • Parallel Execution            │
          │  • Output: CSV Flag Files        │
          └──────────────┬───────────────────┘
                         │
          ┌──────────────▼───────────────────┐
          │    PRODUCTION TEST FLOOR         │
          │  • ATE (Automatic Test Equip)    │
          │  • Flag Reader • STDF Writer     │
          └──────────────┬───────────────────┘
                         │
          ┌──────────────▼───────────────────┐
          │   MONITORING & FEEDBACK LOOP     │
          │  • Metrics Tracking              │
          │  • Drift Detection • Alerting    │
          └──────────────────────────────────┘
```

### Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Storage** | PostgreSQL, Denodo | Historical test data warehouse |
| **Data Processing** | Pandas, NumPy | Data manipulation, feature engineering |
| **ML Framework** | PyTorch 2.3.0 | Deep learning models (VAE, Classifier) |
| **Statistical** | Scikit-learn, SciPy, Statsmodels | Sigma rules, preprocessing, metrics |
| **MLOps** | MLflow | Model registry, versioning, tracking |
| **Artifact Storage** | AWS S3 | Model weights, scalers, metadata |
| **Orchestration** | LSF Cluster | Distributed training, batch jobs |
| **Monitoring** | Custom Python + MLflow | Metrics tracking, drift detection |
| **Production Format** | STDF, CSV | Test equipment interface |

---

## <a name="data-architecture"></a>📊 Data Architecture

### Data Schema

#### Input Data (Test Results)

```python
# Raw test data structure
{
    'chip_id': str,           # Format: "{wafer}_{x}_{y}"
    'test_insertion': str,     # B6, B9, S2, etc.
    'test_number': int,        # Unique test identifier
    'test_name': str,          # Human-readable test name
    'test_value': float,       # Measured value
    'classification': str,     # 'P' (pass) or 'F' (fail)
    'timestamp': datetime,     # Test execution time
    'lot_id': str,            # Manufacturing lot
    'tester_id': str,         # Equipment identifier
    'socket': int,            # Test socket number
    'temperature': str        # Test temperature condition
}
```

#### Feature Engineering Pipeline

```python
# Step 1: Raw data → Pivot table
def pivot_test_data(df):
    """
    Transform long-format test data to wide-format features
    
    Input:  N rows × (chip_id, test_num, value)
    Output: M chips × 200+ features
    """
    return df.pivot_table(
        index='chip_id',
        columns='test_parameter',  # e.g., "B9_1763019_EVR_V_VOUTT_3V3_TRIM_MIN"
        values='test_value',
        aggfunc='first'  # Handle duplicates
    )

# Step 2: Feature naming convention
# Format: {insertion}_{test_num}_{test_name}
# Example: "B9_1763019_EVR_V_VOUTT_3V3_TRIM_MIN"
#          ├─ B9: Back-end insertion
#          ├─ 1763019: Test number
#          └─ EVR_V_VOUTT_3V3_TRIM_MIN: Test description

# Step 3: Target engineering
def create_targets(df):
    """
    Create per-test-suite binary targets
    
    If ANY test in suite fails → Target = 1 (FAIL)
    If ALL tests in suite pass → Target = 0 (PASS)
    """
    targets = {}
    for suite in TEST_SUITES:
        suite_tests = df.filter(regex=f"{suite}_.*_C")  # _C suffix = classification
        targets[f"{suite}_OVERALL_C"] = (suite_tests == 1).any(axis=1).astype(int)
    return pd.DataFrame(targets, index=df.index)
```

#### Data Quality Checks

```python
class DataValidator:
    """Comprehensive data validation for production"""
    
    def validate_chip_id(self, chip_id: str) -> bool:
        """
        Validate chip ID format
        Pattern: <wafer>_<x>_<y>
        Example: "1_11_25"
        """
        pattern = r'^\w+_\d+_\d+_\d+$'
        return bool(re.match(pattern, chip_id))
    
    def validate_test_values(self, df: pd.DataFrame) -> None:
        """Check for invalid values"""
        # Check for NaN
        if df.isna().any().any():
            raise ValueError("NaN values detected")
        
        # Check for infinite
        if df.isin([float('inf'), float('-inf')]).any().any():
            raise ValueError("Infinite values detected")
        
        # Check for placeholder values (99999, -99999)
        outliers = df[(df > 88888) | (df < -88888)]
        if not outliers.empty:
            raise ValueError(f"Outlier placeholders detected: {outliers.shape[0]} values")
    
    def validate_required_columns(self, df: pd.DataFrame, required: List[str]) -> None:
        """Ensure all required features present"""
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
```

### Data Preprocessing Pipeline

```python
class PreprocessingPipeline:
    """End-to-end preprocessing for DTFS"""
    
    def __init__(self, scaler_type='robust'):
        self.scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        self.validator = DataValidator()
        self.required_columns = []
    
    def fit(self, X_train: pd.DataFrame) -> 'PreprocessingPipeline':
        """Learn preprocessing parameters from training data"""
        # 1. Validate
        self.validator.validate_test_values(X_train)
        
        # 2. Store column order
        self.required_columns = X_train.columns.tolist()
        
        # 3. Fit scaler
        self.scaler.fit(X_train)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply preprocessing to new data"""
        # 1. Validate
        self.validator.validate_required_columns(X, self.required_columns)
        self.validator.validate_test_values(X)
        
        # 2. Ensure column order
        X = X[self.required_columns]
        
        # 3. Scale
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
```

---

## <a name="model-architecture"></a>🧠 Model Architecture

### 1. Classification Model (Neural Network)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DTFSClassifier(nn.Module):
    """
    Binary classifier for test skip prediction
    
    Architecture:
    - Simple feedforward network
    - Minimal layers for interpretability
    - Dropout for regularization
    """
    
    def __init__(self, input_size: int = 202, hidden_size: int = 4, dropout: float = 0.5):
        super(DTFSClassifier, self).__init__()
        
        # Layer 1: Input → Hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2: Hidden → Output
        self.fc2 = nn.Linear(hidden_size, 2)  # 2 classes: PASS, FAIL
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (batch_size, input_size)
        
        Returns:
            Logits (batch_size, 2)
        """
        # Hidden layer with ReLU activation
        out = F.relu(self.fc1(x))
        
        # Dropout (training only)
        out = self.dropout1(out)
        
        # Output layer (no activation, will apply softmax later)
        out = self.fc2(out)
        
        return out
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
```

#### Training Configuration

```python
# Hyperparameters
CONFIG = {
    'input_size': 202,        # Number of features
    'hidden_size': 4,         # Hidden layer neurons (tuned)
    'dropout': 0.5,           # Dropout rate
    'batch_size': 4096,       # Large batch for stability
    'epochs': 111,            # Tuned via early stopping
    'learning_rate': 1e-3,    # Initial LR
    'weight_decay': 0,        # L2 regularization (not used, using dropout instead)
    'random_seed': 4096,      # Reproducibility
    'device': 'cuda'          # Use GPU if available
}

# Loss function with class weights
class_weights = torch.tensor([1.0, 2100.0])  # [PASS, FAIL]
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
```

### 2. Variational Autoencoder (Anomaly Detection)

```python
class VariationalAutoEncoder(nn.Module):
    """
    β-VAE for anomaly detection
    
    Architecture:
    - Progressive compression encoder
    - Reparameterization trick
    - Progressive decompression decoder
    - β-VAE loss (reconstruction + KL divergence)
    """
    
    def __init__(self, input_size: int, latent_size: int = 2, 
                 num_layers: int = 2, beta: float = 1.5):
        super(VariationalAutoEncoder, self).__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.beta = beta
        
        # Generate layer sizes (progressive compression)
        layer_sizes = self._generate_layer_sizes()
        
        # Encoder: Input → Latent
        self.encoder = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.encoder.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.encoder.append(nn.ReLU())
        
        # Latent space parameters
        self.fc_mu = nn.Linear(layer_sizes[-1], self.latent_size)
        self.fc_logvar = nn.Linear(layer_sizes[-1], self.latent_size)
        
        # Decoder: Latent → Reconstruction
        self.decoder = nn.ModuleList()
        for i in range(len(layer_sizes) - 1, 0, -1):
            self.decoder.append(nn.Linear(layer_sizes[i], layer_sizes[i-1]))
            if i > 1:
                self.decoder.append(nn.ReLU())
    
    def _generate_layer_sizes(self) -> List[int]:
        """Generate progressive layer sizes"""
        sizes = [self.input_size]
        for i in range(1, self.num_layers):
            # Exponential decay to latent size
            size = int(self.input_size * (
                (self.latent_size / self.input_size) ** (i / (self.num_layers - 1))
            ))
            sizes.append(max(self.latent_size, size))
        return sizes
    
    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for backprop through sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters"""
        for layer in self.encoder:
            x = layer(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction"""
        for layer in self.decoder:
            z = layer(z)
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass"""
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        β-VAE loss: Reconstruction + β * KL Divergence
        
        β > 1: Emphasize disentanglement
        β = 1: Standard VAE
        β < 1: Emphasize reconstruction
        """
        # Reconstruction loss (MSE)
        BCE = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD / x.size(0)  # Normalize by batch size
        
        return BCE + self.beta * KLD
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction error for anomaly detection"""
        self.eval()
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            error = F.mse_loss(recon, x, reduction='none').mean(dim=1)
        return error
```

### 3. Statistical Sigma Rules

```python
import numpy as np
from scipy import stats

class SigmaRule:
    """
    Robust statistical outlier detection
    
    Uses robust statistics (median, quantile-based std)
    to detect chips outside normal operating range
    """
    
    def __init__(self, target_fail_count: int, 
                 min_sigma: float = 3.0, 
                 max_sigma: float = 10.0, 
                 step: float = 0.1):
        """
        Args:
            target_fail_count: Desired number of flagged samples
            min_sigma: Minimum sigma multiplier
            max_sigma: Maximum sigma multiplier
            step: Sigma increment for search
        """
        self.target_fail_count = target_fail_count
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.step = step
        self.dict_sigma = {}
    
    def robust_std(self, val: np.ndarray, q_left: float = 0.05, 
                   q_right: float = 0.95) -> float:
        """
        Calculate robust standard deviation using quantiles
        
        More stable than sample std in presence of outliers
        """
        numerator = np.quantile(val, q_right) - np.quantile(val, q_left)
        denominator = (stats.norm.ppf(q_right) - stats.norm.ppf(q_left))
        return numerator / denominator
    
    def fit(self, X: pd.DataFrame) -> 'SigmaRule':
        """
        Learn sigma multipliers for each feature
        
        For each feature:
        1. Calculate median and robust_std
        2. Search for sigma multiplier that yields target_fail_count
        """
        self.dict_sigma = {}
        
        for col in X.columns:
            col_val = X[col].dropna()
            median = col_val.median()
            rstd = self.robust_std(col_val)
            
            # Search for optimal sigma multipliers
            upper_rejects = []
            lower_rejects = []
            sigmas = []
            
            for sigma in np.arange(self.max_sigma, self.min_sigma, -self.step):
                upper_bound = median + (sigma * rstd)
                lower_bound = median - (sigma * rstd)
                
                upper_rejects.append(len(col_val[col_val > upper_bound]))
                lower_rejects.append(len(col_val[col_val < lower_bound]))
                sigmas.append(sigma)
            
            # Interpolate to find exact sigma for target fail count
            self.dict_sigma[col] = {
                'upper': round(np.interp(self.target_fail_count, upper_rejects, sigmas), 2),
                'lower': round(np.interp(self.target_fail_count, lower_rejects, sigmas), 2),
                'rstd': rstd,
                'median': median
            }
        
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Flag chips with any feature outside sigma bounds
        
        Returns:
            Binary series: 1 if flagged, 0 if normal
        """
        result = pd.Series(0, index=X.index)
        
        for col in X.columns:
            if col not in self.dict_sigma:
                continue
            
            col_val = X[col]
            params = self.dict_sigma[col]
            
            # Calculate bounds
            upper_bound = params['median'] + (params['upper'] * params['rstd'])
            lower_bound = params['median'] - (params['lower'] * params['rstd'])
            
            # Flag outliers
            outliers = (col_val > upper_bound) | (col_val < lower_bound)
            result = result | outliers.astype(int)
        
        return result
```

### 4. Ensemble Model (Hybrid Approach)

```python
class DynamicTestFlowModel:
    """
    Hybrid ensemble combining DL + Sigma rules
    
    Decision Logic: Test if ANY method flags (conservative OR)
    """
    
    def __init__(self, sigma_columns: List[str], 
                 DL_columns: List[str], 
                 is_DL_required: bool = True, 
                 DL_threshold: float = 0.2):
        self.sigma_columns = sigma_columns
        self.DL_columns = DL_columns
        self.is_DL_required = is_DL_required
        self.DL_threshold = DL_threshold
        
        # Will be loaded in load_context
        self.sigma_rule = None
        self.DL_scaler = None
        self.DL_model = None
    
    def load_context(self, context):
        """Load model artifacts (called by MLflow)"""
        self.sigma_rule = joblib.load(context.artifacts["sigma_rule"])
        
        if self.is_DL_required:
            self.DL_scaler = joblib.load(context.artifacts["DL_scaler"])
            self.DL_model = torch.jit.load(context.artifacts["DL_model"])
            self.DL_model.eval()
    
    def predict(self, context, df_input: pd.DataFrame) -> pd.Series:
        """
        Generate test flags
        
        Args:
            df_input: Chip features (indexed by chip_id)
        
        Returns:
            Binary flags: 0 = SKIP, 1 = TEST
        """
        try:
            # 1. Deep Learning prediction (if enabled)
            if self.is_DL_required:
                X_dl = df_input[self.DL_columns]
                X_scaled = self.DL_scaler.transform(X_dl)
                X_tensor = torch.from_numpy(X_scaled.astype(np.float32))
                
                with torch.no_grad():
                    logits = self.DL_model(X_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                
                # Flag if P(Fail) >= threshold
                DL_flags = pd.Series((probs[:, 1] >= self.DL_threshold).astype(int), 
                                    index=df_input.index)
            else:
                DL_flags = pd.Series(0, index=df_input.index)
            
            # 2. Sigma rule prediction
            X_sigma = df_input[self.sigma_columns]
            X_sigma_normalized = X_sigma - X_sigma.median()  # Lot-level normalization
            sigma_flags = self.sigma_rule.predict(X_sigma_normalized)
            
            # 3. Combine with OR logic (conservative)
            final_flags = DL_flags | sigma_flags.astype(int)
            
            return final_flags
        
        except Exception as e:
            # Fail-safe: flag all chips if error occurs
            logging.error(f"Prediction error: {e}")
            return pd.Series(1, index=df_input.index)
```

---

*Continued in next message due to length...*
