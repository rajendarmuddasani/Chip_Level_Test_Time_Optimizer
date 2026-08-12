"""
Model Evaluation and Metrics Calculation

This module provides comprehensive evaluation metrics for test-skip models,
including business metrics (time savings, cost) and ML metrics (accuracy, recall).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTimeEvaluator:
    """
    Comprehensive evaluator for chip test-time optimization models
    
    Calculates both ML metrics and business metrics:
    - ML: Accuracy, Precision, Recall, F1, ROC-AUC
    - Business: Skip rate, Escapee rate, Overreject rate, Time savings
    
    Example:
        >>> evaluator = TestTimeEvaluator()
        >>> metrics = evaluator.evaluate(y_true, y_pred, y_proba)
        >>> print(f"Skip rate: {metrics['skip_rate']:.2%}")
    """

    __test__ = False
    
    def __init__(self):
        self.results = {}
    
    def evaluate(self, y_true: np.ndarray, 
                y_pred: np.ndarray,
                y_proba: np.ndarray = None) -> Dict:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels (0=pass, 1=fail)
            y_pred: Predicted flags (0=skip, 1=test)
            y_proba: Predicted probabilities (optional, for AUC)
        
        Returns:
            Dictionary of metrics
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # ML Metrics
        ml_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            ml_metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        # Business Metrics
        business_metrics = {
            'skip_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'test_rate': (tp + fp) / len(y_true),
            'escapee_rate': fn / (tp + fn) if (tp + fn) > 0 else 0,
            'overreject_rate': fp / (tn + fp) if (tn + fp) > 0 else 0,
        }
        
        # Confusion Matrix Components
        cm_metrics = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_samples': len(y_true)
        }
        
        # Combine all metrics
        all_metrics = {
            **ml_metrics,
            **business_metrics,
            **cm_metrics
        }
        
        self.results = all_metrics
        return all_metrics
    
    def calculate_time_savings(
        self,
        y_pred: np.ndarray,
        early_stage_units: float = 85.0,
        optional_stage_units: float = 15.0,
    ) -> Dict:
        """Calculate simulated optional-stage reduction for the observed flags."""
        if early_stage_units < 0 or optional_stage_units <= 0:
            raise ValueError(
                "Require early_stage_units >= 0 and optional_stage_units > 0"
            )
        if len(y_pred) == 0:
            raise ValueError("At least one prediction is required")
        if not set(np.unique(y_pred)).issubset({0, 1}):
            raise ValueError("Predictions must be binary")
        n_skipped = (y_pred == 0).sum()
        skip_rate = n_skipped / len(y_pred)
        full_flow_units = early_stage_units + optional_stage_units
        baseline_units = len(y_pred) * full_flow_units
        saved_units = n_skipped * optional_stage_units

        return {
            'skip_rate': skip_rate,
            'chips_skipped': int(n_skipped),
            'saved_units': float(saved_units),
            'time_reduction_percent': float(saved_units / baseline_units * 100.0),
            'baseline_units': float(baseline_units),
            'actual_units': float(baseline_units - saved_units),
            'early_stage_units': early_stage_units,
            'optional_stage_units': optional_stage_units,
        }
    
    def print_report(self):
        """Print formatted evaluation report"""
        if not self.results:
            logger.warning("No results to print. Run evaluate() first.")
            return
        
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        print("\n📊 ML METRICS:")
        print(f"  Accuracy:         {self.results['accuracy']:.4f}")
        print(f"  Precision:        {self.results['precision']:.4f}")
        print(f"  Recall:           {self.results['recall']:.4f}")
        print(f"  F1 Score:         {self.results['f1']:.4f}")
        if 'roc_auc' in self.results:
            print(f"  ROC-AUC:          {self.results['roc_auc']:.4f}")
        
        print("\n💼 BUSINESS METRICS:")
        print(f"  Skip Rate:        {self.results['skip_rate']:.2%}")
        print(f"  Escapee Rate:     {self.results['escapee_rate']:.2%}")
        print(f"  Overreject Rate:  {self.results['overreject_rate']:.2%}")
        
        print("\n📈 CONFUSION MATRIX:")
        print(f"  True Negatives:   {self.results['true_negatives']}")
        print(f"  False Positives:  {self.results['false_positives']}")
        print(f"  False Negatives:  {self.results['false_negatives']}")
        print(f"  True Positives:   {self.results['true_positives']}")
        
        print("\n" + "="*60)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save figure
        """
        from sklearn.metrics import ConfusionMatrixDisplay
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, 
            display_labels=['PASS (Skip)', 'FAIL (Test)'],
            cmap='Blues',
            ax=ax
        )
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      save_path: str = None):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Optional path to save figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models side-by-side
        
        Args:
            results_dict: Dictionary mapping model names to their metrics
        
        Returns:
            DataFrame with model comparison
        """
        comparison = pd.DataFrame(results_dict).T
        
        # Sort by key metrics
        if 'skip_rate' in comparison.columns:
            comparison = comparison.sort_values('skip_rate', ascending=False)
        
        return comparison


def calculate_production_metrics(flags_df: pd.DataFrame,
                                 test_suite: str,
                                 early_stage_minutes: float = 85.0,
                                 optional_stage_minutes: float = 15.0) -> Dict:
    """
    Calculate production deployment metrics
    
    Args:
        flags_df: DataFrame with columns [LOT_ID, CHIP_ID, FLAG]
        test_suite: Name of test suite
        early_stage_minutes: Mandatory time per chip
        optional_stage_minutes: Skippable time per chip
    
    Returns:
        Production metrics dictionary
    """
    total_chips = len(flags_df)
    skipped_chips = (flags_df['FLAG'] == 0).sum()
    tested_chips = (flags_df['FLAG'] == 1).sum()
    
    skip_rate = skipped_chips / total_chips
    if early_stage_minutes < 0 or optional_stage_minutes <= 0:
        raise ValueError(
            "Require early_stage_minutes >= 0 and optional_stage_minutes > 0"
        )
    time_saved_hours = skipped_chips * optional_stage_minutes / 60
    baseline_minutes = total_chips * (early_stage_minutes + optional_stage_minutes)
    time_reduction_percent = (
        skipped_chips * optional_stage_minutes / baseline_minutes * 100.0
    )
    
    metrics = {
        'test_suite': test_suite,
        'total_chips': total_chips,
        'skipped_chips': skipped_chips,
        'tested_chips': tested_chips,
        'skip_rate': skip_rate,
        'time_saved_hours': time_saved_hours,
        'time_reduction_percent': time_reduction_percent,
        'early_stage_minutes': early_stage_minutes,
        'optional_stage_minutes': optional_stage_minutes,
    }
    
    # Aggregate by lot
    if 'LOT_ID' in flags_df.columns:
        lot_metrics = flags_df.groupby('LOT_ID')['FLAG'].agg([
            ('skip_rate', lambda x: (x == 0).mean()),
            ('count', 'size')
        ])
        metrics['lots_processed'] = len(lot_metrics)
        metrics['avg_skip_rate_per_lot'] = lot_metrics['skip_rate'].mean()
        metrics['std_skip_rate_per_lot'] = lot_metrics['skip_rate'].std()
    
    return metrics
