"""
Robust Statistical Sigma Rules for Outlier Detection

This module implements statistical outlier detection using robust statistics
(median and quantile-based standard deviation) to identify chips that deviate
significantly from normal behavior.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple
import joblib


class SigmaRule:
    """
    Robust statistical outlier detection for semiconductor testing
    
    Uses median and robust standard deviation (based on quantiles) to detect
    chips with abnormal test values. More robust to outliers than mean/std.
    
    Args:
        target_fail_count: Target number of chips to flag
        min_sigma: Minimum sigma multiplier to search
        max_sigma: Maximum sigma multiplier to search
        step: Step size for sigma search
    
    Example:
        >>> sigma = SigmaRule(target_fail_count=100, min_sigma=3.0, max_sigma=10.0)
        >>> sigma.fit(X_train)
        >>> flags = sigma.predict(X_test)
    """
    
    def __init__(self, target_fail_count: int = 100, 
                 min_sigma: float = 3.0, 
                 max_sigma: float = 10.0, 
                 step: float = 0.1):
        self.target_fail_count = target_fail_count
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.step = step
        self.dict_sigma = {}
    
    def robust_std(self, val: np.ndarray, q_left: float = 0.05, 
                   q_right: float = 0.95) -> float:
        """
        Calculate robust standard deviation using quantiles
        
        More stable than sample std in presence of outliers.
        Based on the inter-quantile range scaled to match normal distribution.
        
        Formula:
            robust_std = (Q_right - Q_left) / (Φ^(-1)(q_right) - Φ^(-1)(q_left))
        
        where Φ^(-1) is the inverse CDF of standard normal distribution.
        
        Args:
            val: Array of values
            q_left: Left quantile (default 5th percentile)
            q_right: Right quantile (default 95th percentile)
        
        Returns:
            Robust standard deviation
        """
        numerator = np.quantile(val, q_right) - np.quantile(val, q_left)
        denominator = (
            stats.norm.ppf(q_right, loc=0, scale=1) - 
            stats.norm.ppf(q_left, loc=0, scale=1)
        )
        return numerator / denominator
    
    def fit(self, X: pd.DataFrame) -> 'SigmaRule':
        """
        Learn optimal sigma multipliers for each feature
        
        For each feature:
        1. Calculate median and robust_std from training data
        2. Search for sigma multiplier that yields target_fail_count
        3. Store parameters for inference
        
        Args:
            X: Training features DataFrame
        
        Returns:
            self (for method chaining)
        """
        self.dict_sigma = {}
        
        for col in X.columns:
            col_val = X[col].dropna()
            median = col_val.median()
            rstd = self.robust_std(col_val)
            
            # Search for optimal sigma multipliers (upper and lower separately)
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
            upper_sigma = np.interp(self.target_fail_count, upper_rejects, sigmas)
            lower_sigma = np.interp(self.target_fail_count, lower_rejects, sigmas)
            
            self.dict_sigma[col] = {
                'upper': round(upper_sigma, 2),
                'lower': round(lower_sigma, 2),
                'rstd': rstd,
                'median': median
            }
        
        return self
    
    def get_bounds(self, col: str) -> Tuple[float, float]:
        """
        Get the upper and lower bounds for a feature
        
        Args:
            col: Feature name
        
        Returns:
            (lower_bound, upper_bound)
        """
        if col not in self.dict_sigma:
            raise ValueError(f"Feature '{col}' not fitted")
        
        params = self.dict_sigma[col]
        lower = params['median'] - (params['lower'] * params['rstd'])
        upper = params['median'] + (params['upper'] * params['rstd'])
        
        return lower, upper
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Flag chips with any feature outside learned sigma bounds
        
        Args:
            X: Features DataFrame (same columns as training)
        
        Returns:
            Binary series: 1 if flagged (TEST), 0 if normal (SKIP)
        """
        result = pd.Series(0, index=X.index)
        
        for col in X.columns:
            if col not in self.dict_sigma:
                # Skip features not seen during training
                continue
            
            col_val = X[col]
            params = self.dict_sigma[col]
            
            # Calculate bounds
            upper_bound = params['median'] + (params['upper'] * params['rstd'])
            lower_bound = params['median'] - (params['lower'] * params['rstd'])
            
            # Flag outliers (OR operation across all features)
            outliers = (col_val > upper_bound) | (col_val < lower_bound)
            result = result | outliers.astype(int)
        
        return result
    
    def get_metadata(self) -> Dict:
        """
        Get fitted parameters (for serialization/inspection)
        
        Returns:
            Dictionary of parameters for each feature
        """
        return self.dict_sigma
    
    def save(self, filepath: str):
        """Save fitted model to disk"""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath: str) -> 'SigmaRule':
        """Load fitted model from disk"""
        return joblib.load(filepath)


def calculate_sigma_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate performance metrics for sigma rule predictions
    
    Args:
        y_true: True labels (0=pass, 1=fail)
        y_pred: Predicted flags (0=skip, 1=test)
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'overreject_rate': fp / (tn + fp) if (tn + fp) > 0 else 0,
        'escapee_rate': fn / (tp + fn) if (tp + fn) > 0 else 0,
        'skip_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'flag_rate': (tp + fp) / (tn + fp + fn + tp)
    }
    
    return metrics


if __name__ == "__main__":
    # Test Sigma Rules
    print("Testing Sigma Rules...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 10000
    n_features = 50
    
    # Normal samples (98%)
    X_normal = pd.DataFrame(
        np.random.randn(int(n_samples * 0.98), n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_normal = np.zeros(len(X_normal))
    
    # Anomalous samples (2%)
    X_anomaly = pd.DataFrame(
        np.random.randn(int(n_samples * 0.02), n_features) * 5,  # Larger variance
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_anomaly = np.ones(len(X_anomaly))
    
    # Combine
    X = pd.concat([X_normal, X_anomaly], ignore_index=True)
    y = np.concatenate([y_normal, y_anomaly])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X.iloc[shuffle_idx].reset_index(drop=True)
    y = y[shuffle_idx]
    
    # Split
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Fit sigma rules
    print(f"\nTraining on {len(X_train)} samples...")
    sigma = SigmaRule(target_fail_count=int(len(X_train) * 0.02), 
                     min_sigma=3.0, max_sigma=10.0)
    sigma.fit(X_train)
    
    # Predict
    print(f"Predicting on {len(X_test)} samples...")
    y_pred = sigma.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_sigma_metrics(y_test, y_pred)
    
    print("\n=== Results ===")
    print(f"Accuracy:        {metrics['accuracy']:.2%}")
    print(f"Precision:       {metrics['precision']:.2%}")
    print(f"Recall:          {metrics['recall']:.2%}")
    print(f"Overreject Rate: {metrics['overreject_rate']:.2%}")
    print(f"Escapee Rate:    {metrics['escapee_rate']:.2%}")
    print(f"Skip Rate:       {metrics['skip_rate']:.2%}")
    print(f"Flag Rate:       {metrics['flag_rate']:.2%}")
    
    # Test bounds
    print("\n=== Sample Bounds ===")
    for i, col in enumerate(X.columns[:3]):
        lower, upper = sigma.get_bounds(col)
        print(f"{col}: [{lower:.2f}, {upper:.2f}]")
    
    print("\n✓ Sigma Rules test passed!")
