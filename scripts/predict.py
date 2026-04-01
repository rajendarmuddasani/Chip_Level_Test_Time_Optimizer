"""
Predict on Unseen Chips — Chip-Level Test Time Optimizer

Loads the trained 3-model ensemble (Classifier + VAE + Sigma Rules)
and predicts SKIP/RUN flags for new chip test data.

Usage:
    cd Chip_Level_Test_Time_Optimizer
    python scripts/predict.py                       # synthetic unseen chips
    python scripts/predict.py --csv path/to/data.csv  # CSV with 202 features
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.classification.classifier_model import ChipTestClassifier

ARTIFACTS = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
FIGURES = os.path.join(os.path.dirname(__file__), '..', 'figures')
DEVICE = 'mps' if torch.backends.mps.is_available() else (
         'cuda' if torch.cuda.is_available() else 'cpu')


# ── Load artefacts ───────────────────────────────────────────────────

def load_artefacts():
    """Load trained models, scaler and config."""
    with open(os.path.join(ARTIFACTS, 'ensemble_config.json')) as f:
        config = json.load(f)

    n_features = config['n_features']

    # Classifier
    classifier = ChipTestClassifier(input_size=n_features, hidden_size=4, dropout=0.5)
    classifier.load_state_dict(
        torch.load(os.path.join(ARTIFACTS, 'classifier.pth'),
                   map_location=DEVICE, weights_only=False))
    classifier.to(DEVICE).eval()

    # Scaler
    scaler = joblib.load(os.path.join(ARTIFACTS, 'scaler.joblib'))

    # Sigma rules
    sigma_rules = joblib.load(os.path.join(ARTIFACTS, 'sigma_rules.joblib'))

    print(f'Loaded classifier ({n_features} features), scaler, sigma rules on {DEVICE}')
    return classifier, scaler, sigma_rules, config


# ── Generate synthetic unseen data ───────────────────────────────────

def generate_unseen_chips(n=200, n_features=202, fail_rate=0.04):
    """Generate n realistic but unseen chip test vectors.

    90% are normal-distribution pass chips (tight spread);
    ~fail_rate are anomalous (shifted means, wider spread).
    """
    rng = np.random.RandomState(999)
    n_fail = int(n * fail_rate)
    n_pass = n - n_fail

    # Normal chips — tight cluster around 0
    X_pass = rng.randn(n_pass, n_features) * 0.5

    # Anomalous chips — shifted mean + wider spread (simulate real defects)
    X_fail = rng.randn(n_fail, n_features) * 2.0 + rng.choice([-3, 3], size=(n_fail, n_features))

    X = np.vstack([X_pass, X_fail])
    y_true = np.array([0] * n_pass + [1] * n_fail)

    # Shuffle
    idx = rng.permutation(n)
    return X[idx], y_true[idx]


# ── Predict ──────────────────────────────────────────────────────────

def predict_ensemble(classifier, scaler, sigma_rules, config, X_raw):
    """Run 3-model ensemble prediction on raw features.

    Returns (flags, details) where flags: 0=SKIP, 1=RUN.
    """
    feature_names = config['feature_names']
    thresh_cls = config['classifier_threshold']
    sigma_target = config.get('sigma_target_fail', 100)

    # Wrap in DataFrame (scaler expects it)
    df_raw = pd.DataFrame(X_raw, columns=feature_names)

    # Scale
    X_scaled = scaler.transform(df_raw)
    if isinstance(X_scaled, pd.DataFrame):
        X_np = X_scaled.values
    else:
        X_np = X_scaled

    # -- 1. Classifier
    X_t = torch.FloatTensor(X_np).to(DEVICE)
    with torch.no_grad():
        probs = classifier.predict_proba(X_t)
        cls_flags = (probs[:, 1] >= thresh_cls).cpu().numpy().astype(int)
        cls_probs = probs[:, 1].cpu().numpy()

    # -- 2. Sigma rules
    sigma_flags = np.zeros(len(X_raw), dtype=int)
    df_scaled = pd.DataFrame(X_np, columns=feature_names)
    if isinstance(sigma_rules, dict) and 'means' in sigma_rules and 'stds' in sigma_rules:
        means = np.array(sigma_rules['means'])
        stds  = np.array(sigma_rules['stds'])
        stds[stds < 1e-8] = 1.0
        z = np.abs((X_raw - means) / stds)
        sigma_flags = (z.max(axis=1) > 3.0).astype(int)
    elif hasattr(sigma_rules, 'predict'):
        sigma_flags = sigma_rules.predict(df_scaled).values.astype(int)

    # -- 3. Ensemble OR
    flags = cls_flags | sigma_flags

    details = {
        'classifier_flags': cls_flags,
        'classifier_probs': cls_probs,
        'sigma_flags': sigma_flags,
        'ensemble_flags': flags,
    }
    return flags, details


# ── Visualisation ────────────────────────────────────────────────────

def visualise(flags, details, y_true, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1. Flag distribution
    skip = (flags == 0).sum()
    run  = (flags == 1).sum()
    axes[0].bar(['SKIP', 'RUN'], [skip, run], color=['#27ae60', '#e74c3c'])
    axes[0].set_title(f'Ensemble Decision  (n={len(flags)})')
    axes[0].set_ylabel('# Chips')
    for i, v in enumerate([skip, run]):
        axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

    # 2. Classifier probability histogram
    axes[1].hist(details['classifier_probs'], bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    axes[1].axvline(0.2, color='red', ls='--', label='threshold=0.2')
    axes[1].set_title('Classifier P(fail)')
    axes[1].set_xlabel('Probability')
    axes[1].legend()

    # 3. Safety check — escapees
    if y_true is not None:
        escapees = ((flags == 0) & (y_true == 1)).sum()
        over_rejects = ((flags == 1) & (y_true == 0)).sum()
        correct = ((flags == 0) & (y_true == 0)).sum() + ((flags == 1) & (y_true == 1)).sum()
        labels = ['Correct\nSkip/Run', 'Over-reject\n(safe)', 'Escapee\n(dangerous)']
        vals = [correct, over_rejects, escapees]
        colours = ['#27ae60', '#f39c12', '#e74c3c']
        axes[2].bar(labels, vals, color=colours)
        axes[2].set_title(f'Safety: {escapees} escapees')
        for i, v in enumerate(vals):
            axes[2].text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, 'No ground truth\nprovided', ha='center',
                     va='center', transform=axes[2].transAxes, fontsize=14)
        axes[2].set_title('Safety Check')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved visualisation → {save_path}')


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Predict test skip/run flags')
    parser.add_argument('--csv', type=str, help='CSV file with 202 feature columns')
    parser.add_argument('-n', type=int, default=200, help='Number of synthetic chips')
    args = parser.parse_args()

    classifier, scaler, sigma_rules, config = load_artefacts()
    os.makedirs(FIGURES, exist_ok=True)

    if args.csv:
        df = pd.read_csv(args.csv)
        X_raw = df.values[:, :config['n_features']]
        y_true = None
        print(f'Loaded {len(X_raw)} chips from {args.csv}')
    else:
        print(f'Generating {args.n} synthetic unseen chips …')
        X_raw, y_true = generate_unseen_chips(args.n, config['n_features'])

    flags, details = predict_ensemble(classifier, scaler, sigma_rules, config, X_raw)

    skip_rate = (flags == 0).mean()
    run_rate  = (flags == 1).mean()
    print(f'\n  Skip rate: {skip_rate:.1%}  |  Run rate: {run_rate:.1%}')
    if y_true is not None:
        escapees = ((flags == 0) & (y_true == 1)).sum()
        print(f'  Escapees:  {escapees}  (target: 0)')

    save_path = os.path.join(FIGURES, 'unseen_predictions.png')
    visualise(flags, details, y_true, save_path)

    # JSON summary
    summary = {
        'n_chips': int(len(flags)),
        'skip_count': int((flags == 0).sum()),
        'run_count': int((flags == 1).sum()),
        'skip_rate': round(float(skip_rate), 4),
    }
    if y_true is not None:
        summary['escapees'] = int(((flags == 0) & (y_true == 1)).sum())
        summary['over_rejects'] = int(((flags == 1) & (y_true == 0)).sum())
    summary_path = os.path.join(ARTIFACTS, 'unseen_predictions.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved summary → {summary_path}')


if __name__ == '__main__':
    main()
