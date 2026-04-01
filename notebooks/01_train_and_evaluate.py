"""
Chip-Level Test Time Optimizer — Training & Evaluation
=======================================================

End-to-end pipeline: generate synthetic semiconductor test data,
train all three models (classifier, VAE, sigma rules),
build the hybrid ensemble, evaluate, and save artifacts.

Run: python notebooks/01_train_and_evaluate.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json, time, warnings
warnings.filterwarnings("ignore")

from models.classification.classifier_model import ChipTestClassifier, CustomDataset
from models.anomaly_detection.vae_model import VariationalAutoEncoder
from models.statistical.sigma_rules import SigmaRule
from models.ensemble import HybridEnsemble
from preprocessing.preprocessing import FeatureScaler

# ---------------------------------------------------------------------------
# 0  Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_CHIPS = 50_000
N_FEATURES = 202
FAIL_RATE = 0.04        # ~4 % fail rate (realistic for mature semiconductor process)
BATCH_SIZE = 4096
CLASSIFIER_EPOCHS = 100
CLASSIFIER_LR = 1e-3
CLASSIFIER_HIDDEN = 4
CLASSIFIER_DROPOUT = 0.5
CLASSIFIER_PATIENCE = 12
CLASSIFIER_THRESHOLD = 0.2
VAE_EPOCHS = 80
VAE_LR = 1e-3
VAE_LATENT = 2
VAE_BETA = 1.5
VAE_PATIENCE = 10
SIGMA_TARGET_FAIL = 100
SIGMA_MIN = 3.0
SIGMA_MAX = 10.0

np.random.seed(SEED)
torch.manual_seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

out_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
os.makedirs(out_dir, exist_ok=True)
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)
fig_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'figures')
os.makedirs(fig_dir, exist_ok=True)

print(f"Device: {device}")
print(f"Generating {N_CHIPS:,} chips × {N_FEATURES} features …")

# ---------------------------------------------------------------------------
# 1  Generate Synthetic Semiconductor Test Data
# ---------------------------------------------------------------------------
# Feature groups mimic real ATE parametric measurements
# Group 1 : Voltage measurements      (features 0-49)
# Group 2 : Current / leakage         (features 50-99)
# Group 3 : Timing / frequency        (features 100-149)
# Group 4 : Resistance / misc analog  (features 150-201)

feature_names = (
    [f"V_{i}" for i in range(50)] +
    [f"I_{i}" for i in range(50)] +
    [f"T_{i}" for i in range(50)] +
    [f"R_{i}" for i in range(52)]
)

n_fail = int(N_CHIPS * FAIL_RATE)
n_pass = N_CHIPS - n_fail

# --- Normal (passing) chips ---
X_pass = np.random.randn(n_pass, N_FEATURES) * 0.5  # tight distribution

# --- Failing chips: correlated shifts in certain feature groups ---
X_fail = np.random.randn(n_fail, N_FEATURES) * 0.5
# Failure mechanism A: voltage drift (~40 % of fails)
nA = int(0.4 * n_fail)
X_fail[:nA, 0:50] += np.random.uniform(2.0, 4.0, size=(nA, 50))
# Failure mechanism B: leakage spike (~30 %)
nB = int(0.3 * n_fail)
X_fail[nA:nA+nB, 50:100] += np.random.uniform(3.0, 6.0, size=(nB, 50))
# Failure mechanism C: timing violations (~20 %)
nC = int(0.2 * n_fail)
X_fail[nA+nB:nA+nB+nC, 100:150] += np.random.uniform(2.5, 5.0, size=(nC, 50))
# Failure mechanism D: mixed random shift (~10 %)
nD = n_fail - nA - nB - nC
shifts = np.random.choice(N_FEATURES, size=30, replace=False)
X_fail[nA+nB+nC:, shifts] += np.random.uniform(1.5, 3.5, size=(nD, 30))

X_all = np.vstack([X_pass, X_fail])
y_all = np.concatenate([np.zeros(n_pass), np.ones(n_fail)])

# Shuffle
perm = np.random.permutation(N_CHIPS)
X_all = X_all[perm]
y_all = y_all[perm]

df = pd.DataFrame(X_all, columns=feature_names)
df['label'] = y_all.astype(int)
df.insert(0, 'CHIP_ID', [f"CHIP_{i:06d}" for i in range(N_CHIPS)])

csv_path = os.path.join(data_dir, 'chip_test_data.csv')
df.to_csv(csv_path, index=False)
print(f"Saved {csv_path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
print(f"  Pass: {(y_all==0).sum():,}  |  Fail: {(y_all==1).sum():,}  ({y_all.mean():.2%} fail rate)")

# ---------------------------------------------------------------------------
# 2  Preprocessing
# ---------------------------------------------------------------------------
X = df.drop(columns=['CHIP_ID', 'label']).values
y = y_all.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler = FeatureScaler(method='robust')
X_train_s = scaler.fit_transform(pd.DataFrame(X_train, columns=feature_names))
X_test_s = scaler.transform(pd.DataFrame(X_test, columns=feature_names))

X_train_np = X_train_s.values if isinstance(X_train_s, pd.DataFrame) else X_train_s
X_test_np = X_test_s.values if isinstance(X_test_s, pd.DataFrame) else X_test_s

print(f"\nTrain: {X_train_np.shape}  |  Test: {X_test_np.shape}")

# ---------------------------------------------------------------------------
# 3  Train Neural-Network Classifier
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TRAINING: ChipTestClassifier")
print("="*60)

# Class weights for imbalanced data
n0 = (y_train == 0).sum()
n1 = (y_train == 1).sum()
w0 = 1.0
w1 = n0 / n1
class_weights = torch.FloatTensor([w0, w1]).to(device)
print(f"  Class weights: PASS={w0:.2f}, FAIL={w1:.2f}")

train_ds = CustomDataset(X_train_np, y_train)
test_ds = CustomDataset(X_test_np, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

classifier = ChipTestClassifier(
    input_size=N_FEATURES,
    hidden_size=CLASSIFIER_HIDDEN,
    dropout=CLASSIFIER_DROPOUT,
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(classifier.parameters(), lr=CLASSIFIER_LR)

best_val_loss = float('inf')
patience_count = 0
train_losses, val_losses = [], []

t0 = time.time()
for epoch in range(1, CLASSIFIER_EPOCHS + 1):
    classifier.train()
    running_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device).long()
        optimizer.zero_grad()
        logits = classifier(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * Xb.size(0)
    train_loss = running_loss / len(train_ds)
    train_losses.append(train_loss)

    # Validation
    classifier.eval()
    val_running = 0.0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device).long()
            logits = classifier(Xb)
            val_running += criterion(logits, yb).item() * Xb.size(0)
    val_loss = val_running / len(test_ds)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_count = 0
        torch.save(classifier.state_dict(), os.path.join(out_dir, 'classifier.pth'))
    else:
        patience_count += 1

    if epoch % 10 == 0 or patience_count == 0:
        print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  {'*' if patience_count==0 else ''}")

    if patience_count >= CLASSIFIER_PATIENCE:
        print(f"  Early stopping at epoch {epoch}")
        break

classifier.load_state_dict(torch.load(os.path.join(out_dir, 'classifier.pth'), weights_only=True))
print(f"  Classifier training done in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# 4  Train VAE Anomaly Detector (on passing chips only)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TRAINING: Variational Autoencoder (β-VAE)")
print("="*60)

# Train VAE only on normal (pass) data — anomalies = high reconstruction error
pass_mask = y_train == 0
X_pass_train = X_train_np[pass_mask]
X_pass_val = X_test_np[y_test == 0]

vae = VariationalAutoEncoder(
    input_size=N_FEATURES,
    latent_size=VAE_LATENT,
    num_layers=2,
    beta=VAE_BETA,
).to(device)

vae_optimizer = optim.Adam(vae.parameters(), lr=VAE_LR)
vae_train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_pass_train))
vae_val_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_pass_val))
vae_train_loader = DataLoader(vae_train_ds, batch_size=BATCH_SIZE, shuffle=True)
vae_val_loader = DataLoader(vae_val_ds, batch_size=BATCH_SIZE, shuffle=False)

best_vae_loss = float('inf')
vae_patience = 0
vae_train_losses, vae_val_losses = [], []

t0 = time.time()
for epoch in range(1, VAE_EPOCHS + 1):
    vae.train()
    running = 0.0
    for (Xb,) in vae_train_loader:
        Xb = Xb.to(device)
        recon, mu, logvar = vae(Xb)
        loss = vae.loss_function(recon, Xb, mu, logvar)
        vae_optimizer.zero_grad()
        loss.backward()
        vae_optimizer.step()
        running += loss.item() * Xb.size(0)
    t_loss = running / len(X_pass_train)
    vae_train_losses.append(t_loss)

    vae.eval()
    val_running = 0.0
    with torch.no_grad():
        for (Xb,) in vae_val_loader:
            Xb = Xb.to(device)
            recon, mu, logvar = vae(Xb)
            val_running += vae.loss_function(recon, Xb, mu, logvar).item() * Xb.size(0)
    v_loss = val_running / len(X_pass_val)
    vae_val_losses.append(v_loss)

    if v_loss < best_vae_loss:
        best_vae_loss = v_loss
        vae_patience = 0
        torch.save(vae.state_dict(), os.path.join(out_dir, 'vae.pth'))
    else:
        vae_patience += 1

    if epoch % 10 == 0 or vae_patience == 0:
        print(f"  Epoch {epoch:3d}  train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  {'*' if vae_patience==0 else ''}")

    if vae_patience >= VAE_PATIENCE:
        print(f"  Early stopping at epoch {epoch}")
        break

vae.load_state_dict(torch.load(os.path.join(out_dir, 'vae.pth'), weights_only=True))
print(f"  VAE training done in {time.time()-t0:.1f}s")

# Determine VAE threshold from training pass data (95th percentile of errors)
vae.eval()
with torch.no_grad():
    pass_errors = vae.get_reconstruction_error(
        torch.FloatTensor(X_pass_train).to(device)
    ).cpu().numpy()
vae_threshold = float(np.percentile(pass_errors, 95))
print(f"  VAE anomaly threshold (95th %ile of pass): {vae_threshold:.4f}")

# ---------------------------------------------------------------------------
# 5  Fit Sigma Rules
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("FITTING: Sigma Rules (Robust Statistical Bounds)")
print("="*60)

sigma = SigmaRule(
    target_fail_count=SIGMA_TARGET_FAIL,
    min_sigma=SIGMA_MIN,
    max_sigma=SIGMA_MAX,
)

X_train_df = pd.DataFrame(X_train_np, columns=feature_names)
X_test_df = pd.DataFrame(X_test_np, columns=feature_names)

t0 = time.time()
sigma.fit(X_train_df)
print(f"  Sigma rules fitted in {time.time()-t0:.1f}s  ({len(sigma.dict_sigma)} feature rules)")

# Save sigma
import joblib
joblib.dump(sigma, os.path.join(out_dir, 'sigma_rules.joblib'))

# ---------------------------------------------------------------------------
# 6  Build Ensemble & Evaluate
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("ENSEMBLE EVALUATION")
print("="*60)

ensemble = HybridEnsemble(
    classifier=classifier,
    vae=vae,
    sigma=sigma,
    classifier_threshold=CLASSIFIER_THRESHOLD,
    vae_threshold=vae_threshold,
)

# Per-model breakdown
details = ensemble.predict_with_details(X_test_df)

for name in ['classifier', 'vae', 'sigma', 'ensemble']:
    preds = details[name]
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    skip = tn / (tn + fp) if (tn + fp) > 0 else 0
    escape = fn / (tp + fn) if (tp + fn) > 0 else 0
    flag = preds.mean()
    print(f"\n  --- {name.upper()} ---")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Skip rate:    {skip:.2%}")
    print(f"  Escapee rate: {escape:.2%}")
    print(f"  Flag rate:    {flag:.2%}")

# Full ensemble report
y_pred = details['ensemble']
print("\n\nClassification Report (Ensemble):")
print(classification_report(y_test, y_pred, target_names=['PASS (skip)', 'FAIL (test)']))

# ROC-AUC for classifier component
if 'classifier_probs' in details:
    roc_auc = roc_auc_score(y_test, details['classifier_probs'])
    print(f"Classifier ROC-AUC: {roc_auc:.4f}")

# Business metrics
total_test = len(y_test)
skipped = (y_pred == 0).sum()
tested = (y_pred == 1).sum()
test_time_per_chip = 30.0  # seconds
time_saved = skipped * test_time_per_chip / 3600
print(f"\nBusiness Impact (on {total_test:,} chip test set):")
print(f"  Chips skipped:     {skipped:,} ({skipped/total_test:.1%})")
print(f"  Chips tested:      {tested:,} ({tested/total_test:.1%})")
print(f"  Test time saved:   {time_saved:.1f} hours")
print(f"  Escapees (missed): {(y_test[y_pred == 0] == 1).sum()}")

# ---------------------------------------------------------------------------
# 7  Save Artifacts
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("SAVING ARTIFACTS")
print("="*60)

# Save scaler
joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))

# Save ensemble config
ensemble_config = {
    'classifier_threshold': CLASSIFIER_THRESHOLD,
    'vae_threshold': vae_threshold,
    'sigma_target_fail': SIGMA_TARGET_FAIL,
    'n_features': N_FEATURES,
    'feature_names': feature_names,
    'device': device,
    'train_samples': len(X_train_np),
    'test_samples': len(X_test_np),
    'fail_rate': float(y_all.mean()),
}
with open(os.path.join(out_dir, 'ensemble_config.json'), 'w') as f:
    json.dump(ensemble_config, f, indent=2)

# Save evaluation results
eval_results = {}
for name in ['classifier', 'vae', 'sigma', 'ensemble']:
    preds = details[name]
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    eval_results[name] = {
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'skip_rate': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        'escapee_rate': float(fn / (tp + fn)) if (tp + fn) > 0 else 0,
        'flag_rate': float(preds.mean()),
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
    }
if 'classifier_probs' in details:
    eval_results['classifier']['roc_auc'] = float(roc_auc_score(y_test, details['classifier_probs']))

with open(os.path.join(out_dir, 'evaluation_results.json'), 'w') as f:
    json.dump(eval_results, f, indent=2)

print(f"  classifier.pth          → {out_dir}")
print(f"  vae.pth                 → {out_dir}")
print(f"  sigma_rules.joblib      → {out_dir}")
print(f"  scaler.joblib           → {out_dir}")
print(f"  ensemble_config.json    → {out_dir}")
print(f"  evaluation_results.json → {out_dir}")

# ---------------------------------------------------------------------------
# 8  Generate Figures
# ---------------------------------------------------------------------------
print("\nGenerating figures …")

# 8a. Training loss curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, label='Train')
axes[0].plot(val_losses, label='Val')
axes[0].set_title('Classifier Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(vae_train_losses, label='Train')
axes[1].plot(vae_val_losses, label='Val')
axes[1].set_title('VAE Loss (β-VAE)')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'training_curves.png'), dpi=150)
plt.close()

# 8b. Confusion matrix (ensemble)
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['SKIP', 'TEST'], yticklabels=['PASS', 'FAIL'], ax=ax)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
ax.set_title('Ensemble Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'confusion_matrix.png'), dpi=150)
plt.close()

# 8c. ROC curve (classifier)
if 'classifier_probs' in details:
    fpr, tpr, _ = roc_curve(y_test, details['classifier_probs'])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Classifier ROC Curve')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'roc_curve.png'), dpi=150)
    plt.close()

# 8d. VAE reconstruction error distribution
vae.eval()
with torch.no_grad():
    all_errors = vae.get_reconstruction_error(
        torch.FloatTensor(X_test_np).to(device)
    ).cpu().numpy()
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(all_errors[y_test == 0], bins=60, alpha=0.6, label='Pass', density=True)
ax.hist(all_errors[y_test == 1], bins=60, alpha=0.6, label='Fail', density=True)
ax.axvline(vae_threshold, color='red', linestyle='--', label=f'Threshold={vae_threshold:.3f}')
ax.set_xlabel('Reconstruction Error')
ax.set_title('VAE Anomaly Score Distribution')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'vae_error_dist.png'), dpi=150)
plt.close()

# 8e. Per-model flag Venn breakdown
fig, ax = plt.subplots(figsize=(6, 4))
model_names = ['Classifier', 'VAE', 'Sigma', 'Ensemble']
flag_rates = [details[n.lower()].mean() for n in model_names]
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
bars = ax.bar(model_names, flag_rates, color=colors)
ax.set_ylabel('Flag Rate')
ax.set_title('Per-Model Flag Rate Comparison')
for bar, rate in zip(bars, flag_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{rate:.1%}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'flag_rate_comparison.png'), dpi=150)
plt.close()

print(f"  Figures saved to {fig_dir}/")

print("\n" + "="*60)
print("DONE — All models trained, evaluated, and artifacts saved.")
print("="*60)
