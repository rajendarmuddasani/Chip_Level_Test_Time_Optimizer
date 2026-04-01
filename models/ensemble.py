"""
Ensemble Model combining Classification, Anomaly Detection, and Statistical Rules

This module implements the hybrid ensemble that combines predictions from:
1. Neural Network Classifier
2. VAE Anomaly Detector
3. Sigma Rules

Uses conservative OR logic: TEST if ANY method flags the chip.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import joblib


class HybridEnsemble:
    """
    Ensemble model combining DL and statistical approaches
    
    Decision Logic (Conservative OR):
        TEST = Classifier_flag OR VAE_flag OR Sigma_flag
        SKIP = All methods agree chip is normal
    
    This conservative approach prioritizes safety (zero escapees) over
    aggressive test skipping.
    
    Attributes:
        classifier: Neural network classifier model
        vae: Variational autoencoder for anomaly detection
        sigma: Statistical sigma rules
        classifier_threshold: Decision threshold for classifier (default 0.2)
        vae_threshold: Reconstruction error threshold for VAE
    
    Example:
        >>> ensemble = HybridEnsemble(classifier, vae, sigma)
        >>> flags = ensemble.predict(X_test)
        >>> print(f"Skip rate: {(flags == 0).mean():.2%}")
    """
    
    def __init__(self, 
                 classifier=None, 
                 vae=None, 
                 sigma=None,
                 classifier_threshold: float = 0.2,
                 vae_threshold: float = None):
        """
        Initialize ensemble with sub-models
        
        Args:
            classifier: Trained PyTorch classifier
            vae: Trained VAE model
            sigma: Fitted sigma rules
            classifier_threshold: Threshold for classifier (lower = more conservative)
            vae_threshold: Threshold for VAE reconstruction error
        """
        self.classifier = classifier
        self.vae = vae
        self.sigma = sigma
        self.classifier_threshold = classifier_threshold
        self.vae_threshold = vae_threshold
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.classifier is not None:
            self.classifier.to(self.device)
            self.classifier.eval()
        
        if self.vae is not None:
            self.vae.to(self.device)
            self.vae.eval()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions with OR logic
        
        Args:
            X: Input features DataFrame
        
        Returns:
            Binary flags: 0 = SKIP test, 1 = RUN test
        """
        n_samples = len(X)
        flags = np.zeros(n_samples, dtype=int)
        
        # Convert to tensor for DL models
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        
        # 1. Neural Network Classifier
        if self.classifier is not None:
            with torch.no_grad():
                probs = self.classifier.predict_proba(X_tensor)
                classifier_flags = (probs[:, 1] >= self.classifier_threshold).cpu().numpy()
                flags = flags | classifier_flags
        
        # 2. VAE Anomaly Detection
        if self.vae is not None and self.vae_threshold is not None:
            with torch.no_grad():
                vae_flags = self.vae.detect_anomalies(X_tensor, self.vae_threshold).cpu().numpy()
                flags = flags | vae_flags
        
        # 3. Statistical Sigma Rules
        if self.sigma is not None:
            sigma_flags = self.sigma.predict(X).values
            flags = flags | sigma_flags
        
        return flags
    
    def predict_with_details(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate predictions with per-model breakdown
        
        Useful for debugging and understanding which models contribute to flags.
        
        Args:
            X: Input features DataFrame
        
        Returns:
            Dictionary containing:
                - 'ensemble': Final ensemble flags
                - 'classifier': Classifier-only flags
                - 'vae': VAE-only flags
                - 'sigma': Sigma-only flags
                - 'classifier_probs': Raw classifier probabilities
                - 'vae_errors': Raw VAE reconstruction errors
        """
        n_samples = len(X)
        results = {
            'classifier': np.zeros(n_samples, dtype=int),
            'vae': np.zeros(n_samples, dtype=int),
            'sigma': np.zeros(n_samples, dtype=int)
        }
        
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        
        # Get predictions from each model
        if self.classifier is not None:
            with torch.no_grad():
                probs = self.classifier.predict_proba(X_tensor)
                results['classifier'] = (probs[:, 1] >= self.classifier_threshold).cpu().numpy()
                results['classifier_probs'] = probs[:, 1].cpu().numpy()
        
        if self.vae is not None and self.vae_threshold is not None:
            with torch.no_grad():
                errors = self.vae.get_reconstruction_error(X_tensor)
                results['vae'] = (errors > self.vae_threshold).cpu().numpy()
                results['vae_errors'] = errors.cpu().numpy()
        
        if self.sigma is not None:
            results['sigma'] = self.sigma.predict(X).values
        
        # Ensemble with OR logic
        results['ensemble'] = (
            results['classifier'] | 
            results['vae'] | 
            results['sigma']
        )
        
        return results
    
    def calculate_metrics(self, X: pd.DataFrame, y_true: np.ndarray) -> Dict[str, Dict]:
        """
        Calculate metrics for ensemble and each sub-model
        
        Args:
            X: Input features
            y_true: True labels (0=pass, 1=fail)
        
        Returns:
            Dictionary of metrics for each model and ensemble
        """
        from sklearn.metrics import confusion_matrix, classification_report
        
        predictions = self.predict_with_details(X)
        metrics = {}
        
        for model_name in ['classifier', 'vae', 'sigma', 'ensemble']:
            if model_name not in predictions:
                continue
            
            y_pred = predictions[model_name]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            metrics[model_name] = {
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'skip_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'overreject_rate': fp / (tn + fp) if (tn + fp) > 0 else 0,
                'escapee_rate': fn / (tp + fn) if (tp + fn) > 0 else 0,
                'flag_count': int(y_pred.sum()),
                'flag_rate': float(y_pred.mean())
            }
        
        return metrics
    
    def save(self, filepath: str):
        """
        Save ensemble to disk
        
        Saves all sub-models and configuration to a single file.
        
        Args:
            filepath: Path to save file
        """
        state = {
            'classifier_state': self.classifier.state_dict() if self.classifier else None,
            'vae_state': self.vae.state_dict() if self.vae else None,
            'sigma': self.sigma,
            'classifier_threshold': self.classifier_threshold,
            'vae_threshold': self.vae_threshold,
            'classifier_config': {
                'input_size': self.classifier.input_size if self.classifier else None,
                'hidden_size': self.classifier.hidden_size if self.classifier else None,
                'dropout': self.classifier.dropout_rate if self.classifier else None,
            },
            'vae_config': {
                'input_size': self.vae.input_size if self.vae else None,
                'latent_size': self.vae.latent_size if self.vae else None,
                'num_layers': self.vae.num_layers if self.vae else None,
                'beta': self.vae.beta if self.vae else None,
            }
        }
        
        torch.save(state, filepath)
        print(f"Ensemble saved to {filepath}")
    
    @staticmethod
    def load(filepath: str, classifier_class=None, vae_class=None):
        """
        Load ensemble from disk
        
        Args:
            filepath: Path to saved file
            classifier_class: Class for classifier (needed for reconstruction)
            vae_class: Class for VAE (needed for reconstruction)
        
        Returns:
            Loaded HybridEnsemble
        """
        state = torch.load(filepath)
        
        # Reconstruct models
        classifier = None
        if state['classifier_state'] and classifier_class:
            classifier = classifier_class(**state['classifier_config'])
            classifier.load_state_dict(state['classifier_state'])
        
        vae = None
        if state['vae_state'] and vae_class:
            vae = vae_class(**state['vae_config'])
            vae.load_state_dict(state['vae_state'])
        
        ensemble = HybridEnsemble(
            classifier=classifier,
            vae=vae,
            sigma=state['sigma'],
            classifier_threshold=state['classifier_threshold'],
            vae_threshold=state['vae_threshold']
        )
        
        print(f"Ensemble loaded from {filepath}")
        return ensemble


def calculate_time_savings(flags: np.ndarray, test_time_per_chip: float = 30.0) -> Dict[str, float]:
    """
    Calculate test time savings from skip flags
    
    Args:
        flags: Binary flags (0=skip, 1=test)
        test_time_per_chip: Average test time per chip in minutes
    
    Returns:
        Dictionary with time savings metrics
    """
    n_total = len(flags)
    n_skipped = (flags == 0).sum()
    n_tested = (flags == 1).sum()
    
    skip_rate = n_skipped / n_total
    total_time_original = n_total * test_time_per_chip
    total_time_saved = n_skipped * test_time_per_chip
    total_time_actual = n_tested * test_time_per_chip
    
    return {
        'skip_rate': skip_rate,
        'n_skipped': n_skipped,
        'n_tested': n_tested,
        'time_saved_minutes': total_time_saved,
        'time_saved_hours': total_time_saved / 60,
        'time_reduction_percent': skip_rate * 100,
        'original_time_minutes': total_time_original,
        'actual_time_minutes': total_time_actual
    }


if __name__ == "__main__":
    print("Testing Hybrid Ensemble...")
    
    # This is a demonstration - in practice, you would load trained models
    print("\n⚠️  This is a structural test only.")
    print("Real usage requires trained models from classifier_model.py, vae_model.py, and sigma_rules.py")
    
    # Create dummy data
    n_samples = 1000
    n_features = 50
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = np.random.randint(0, 2, n_samples)
    
    # Initialize empty ensemble (no models loaded)
    ensemble = HybridEnsemble()
    
    print(f"\nCreated ensemble with:")
    print(f"  - Classifier: {ensemble.classifier is not None}")
    print(f"  - VAE: {ensemble.vae is not None}")
    print(f"  - Sigma: {ensemble.sigma is not None}")
    
    print("\n✓ Ensemble structure test passed!")
    print("\nTo use this ensemble in production:")
    print("1. Train classifier: python models/classification/classifier_model.py")
    print("2. Train VAE: python models/anomaly_detection/vae_model.py")
    print("3. Fit sigma rules: python models/statistical/sigma_rules.py")
    print("4. Load trained models into HybridEnsemble")
    print("5. Call ensemble.predict(X) to generate flags")
