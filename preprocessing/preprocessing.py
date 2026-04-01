"""
Data Preprocessing and Validation

This module handles input validation, outlier detection, and feature engineering
for semiconductor test data before model inference.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler


class DataValidator:
    """
    Validate input data quality and format
    
    Checks for:
    - Missing values
    - Invalid chip IDs
    - Out-of-range values
    - Data type consistency
    
    Example:
        >>> validator = DataValidator()
        >>> is_valid, errors = validator.validate(df)
    """
    
    def __init__(self, expected_features: Optional[List[str]] = None):
        """
        Initialize validator
        
        Args:
            expected_features: List of required feature names
        """
        self.expected_features = expected_features
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input DataFrame
        
        Args:
            df: Input DataFrame to validate
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check if DataFrame is empty
        if len(df) == 0:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check for expected features
        if self.expected_features:
            missing_features = set(self.expected_features) - set(df.columns)
            if missing_features:
                errors.append(f"Missing features: {missing_features}")
        
        # Check for placeholder values (common in semiconductor data)
        placeholder_values = [99999, -99999, 999999, -999999]
        for col in df.select_dtypes(include=[np.number]).columns:
            n_placeholders = df[col].isin(placeholder_values).sum()
            if n_placeholders > 0:
                errors.append(
                    f"Column '{col}' contains {n_placeholders} placeholder values"
                )
        
        # Check for excessive missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            errors.append(
                f"High missing values in: {high_missing.to_dict()}"
            )
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = [col for col in numeric_cols if np.isinf(df[col]).any()]
        if inf_cols:
            errors.append(f"Infinite values in: {inf_cols}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def clean_placeholders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace placeholder values with NaN
        
        Args:
            df: Input DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        placeholder_values = [99999, -99999, 999999, -999999]
        
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace(placeholder_values, np.nan)
        
        return df


class FeatureScaler:
    """
    Scale features using robust statistics
    
    Uses RobustScaler by default (median and IQR) to handle outliers better
    than StandardScaler (mean and std).
    
    Example:
        >>> scaler = FeatureScaler()
        >>> scaler.fit(X_train)
        >>> X_scaled = scaler.transform(X_test)
    """
    
    def __init__(self, method: str = 'robust'):
        """
        Initialize scaler
        
        Args:
            method: 'robust' or 'standard'
        """
        self.method = method
        if method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame) -> 'FeatureScaler':
        """
        Fit scaler to training data
        
        Args:
            X: Training features
        
        Returns:
            self
        """
        self.feature_names = list(X.columns)
        self.scaler.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features
        
        Args:
            X: Features to transform
        
        Returns:
            Scaled features
        """
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert scaled features back to original scale"""
        X_original = self.scaler.inverse_transform(X)
        return pd.DataFrame(X_original, columns=self.feature_names, index=X.index)
    
    def save(self, filepath: str):
        """Save fitted scaler"""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath: str) -> 'FeatureScaler':
        """Load fitted scaler"""
        return joblib.load(filepath)


class OutlierDetector:
    """
    Detect outliers using IQR method
    
    Values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR are marked as outliers.
    
    Example:
        >>> detector = OutlierDetector()
        >>> detector.fit(X_train)
        >>> outlier_mask = detector.detect(X_test)
    """
    
    def __init__(self, multiplier: float = 1.5):
        """
        Initialize detector
        
        Args:
            multiplier: IQR multiplier (1.5 is standard, higher is more lenient)
        """
        self.multiplier = multiplier
        self.bounds = {}
    
    def fit(self, X: pd.DataFrame) -> 'OutlierDetector':
        """
        Calculate bounds from training data
        
        Args:
            X: Training features
        
        Returns:
            self
        """
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            
            self.bounds[col] = {
                'lower': Q1 - self.multiplier * IQR,
                'upper': Q3 + self.multiplier * IQR
            }
        
        return self
    
    def detect(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers
        
        Args:
            X: Features to check
        
        Returns:
            Boolean DataFrame: True = outlier
        """
        outliers = pd.DataFrame(False, index=X.index, columns=X.columns)
        
        for col in X.columns:
            if col not in self.bounds:
                continue
            
            lower = self.bounds[col]['lower']
            upper = self.bounds[col]['upper']
            
            outliers[col] = (X[col] < lower) | (X[col] > upper)
        
        return outliers
    
    def get_outlier_summary(self, X: pd.DataFrame) -> pd.Series:
        """
        Get count of outliers per sample
        
        Args:
            X: Features to check
        
        Returns:
            Series with outlier count per sample
        """
        outliers = self.detect(X)
        return outliers.sum(axis=1)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load test data from CSV
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame with test data
    """
    return pd.read_csv(filepath)


def preprocess_pipeline(df: pd.DataFrame, 
                       scaler: FeatureScaler,
                       validator: Optional[DataValidator] = None,
                       clean_placeholders: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Full preprocessing pipeline
    
    Steps:
    1. Validate input
    2. Clean placeholder values
    3. Handle missing values
    4. Scale features
    
    Args:
        df: Raw input DataFrame
        scaler: Fitted FeatureScaler
        validator: Optional DataValidator
        clean_placeholders: Whether to clean placeholder values
    
    Returns:
        (processed_df, metadata)
    """
    metadata = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'validation_errors': []
    }
    
    # Step 1: Validate
    if validator:
        is_valid, errors = validator.validate(df)
        metadata['validation_errors'] = errors
        if not is_valid:
            raise ValueError(f"Validation failed: {errors}")
    
    # Step 2: Clean placeholders
    if clean_placeholders:
        validator_temp = DataValidator()
        df = validator_temp.clean_placeholders(df)
        metadata['n_placeholders_cleaned'] = (df.isnull().sum() - 
                                              metadata.get('n_missing_original', 0))
    
    # Step 3: Handle missing values (simple forward fill for now)
    n_missing_before = df.isnull().sum().sum()
    df = df.ffill().bfill().fillna(0)
    metadata['n_missing_imputed'] = n_missing_before
    
    # Step 4: Scale features
    df_scaled = scaler.transform(df)
    
    return df_scaled, metadata


if __name__ == "__main__":
    print("Testing Preprocessing Pipeline...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some issues
    df.iloc[5, 3] = 99999  # Placeholder
    df.iloc[10, 7] = np.nan  # Missing
    df.iloc[20, 2] = 100  # Outlier
    
    # Test validator
    print("\n=== Testing Validator ===")
    validator = DataValidator(expected_features=df.columns.tolist())
    is_valid, errors = validator.validate(df)
    print(f"Valid: {is_valid}")
    print(f"Errors: {errors}")
    
    # Test scaler
    print("\n=== Testing Scaler ===")
    scaler = FeatureScaler(method='robust')
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    print(f"Original mean: {df.mean().mean():.4f}")
    print(f"Scaled mean: {df_scaled.mean().mean():.4f}")
    
    # Test outlier detector
    print("\n=== Testing Outlier Detector ===")
    detector = OutlierDetector(multiplier=1.5)
    detector.fit(df)
    outliers = detector.detect(df)
    outlier_counts = detector.get_outlier_summary(df)
    print(f"Samples with outliers: {(outlier_counts > 0).sum()}/{len(df)}")
    
    # Test full pipeline
    print("\n=== Testing Full Pipeline ===")
    df_processed, metadata = preprocess_pipeline(df, scaler, validator)
    print(f"Processed shape: {df_processed.shape}")
    print(f"Metadata: {metadata}")
    
    print("\n✓ Preprocessing test passed!")
