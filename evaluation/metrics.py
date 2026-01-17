"""
Model Evaluation and Metrics Calculation

This module provides comprehensive evaluation metrics for DTFS models,
including business metrics (time savings, cost) and ML metrics (accuracy, recall).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DTFSEvaluator:
    """
    Comprehensive evaluator for DTFS models
    
    Calculates both ML metrics and business metrics:
    - ML: Accuracy, Precision, Recall, F1, ROC-AUC
    - Business: Skip rate, Escapee rate, Overreject rate, Time savings
    
    Example:
        >>> evaluator = DTFSEvaluator()
        >>> metrics = evaluator.evaluate(y_true, y_pred, y_proba)
        >>> print(f"Skip rate: {metrics['skip_rate']:.2%}")
    """
    
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
    
    def calculate_time_savings(self, y_pred: np.ndarray,
                              test_time_per_chip: float = 30.0,
                              cost_per_hour: float = 100.0,
                              n_lots: int = 1,
                              chips_per_lot: int = 13000) -> Dict:
        """
        Calculate time and cost savings from test skipping
        
        Args:
            y_pred: Predicted flags (0=skip, 1=test)
            test_time_per_chip: Minutes per chip
            cost_per_hour: Tester cost per hour (€)
            n_lots: Number of production lots
            chips_per_lot: Average chips per lot
        
        Returns:
            Dictionary with savings calculations
        """
        n_skipped = (y_pred == 0).sum()
        skip_rate = n_skipped / len(y_pred)
        
        # Time calculations
        time_saved_minutes = n_skipped * test_time_per_chip
        time_saved_hours = time_saved_minutes / 60
        
        # Scale to production
        total_chips = n_lots * chips_per_lot
        total_time_saved_hours = skip_rate * total_chips * test_time_per_chip / 60
        
        # Cost calculations
        cost_saved = total_time_saved_hours * cost_per_hour
        
        return {
            'skip_rate': skip_rate,
            'chips_skipped': n_skipped,
            'time_saved_hours_sample': time_saved_hours,
            'time_reduction_percent': skip_rate * 100,
            'total_lots': n_lots,
            'total_chips': total_chips,
            'total_time_saved_hours': total_time_saved_hours,
            'estimated_cost_saved_eur': cost_saved,
            'cost_saved_millions': cost_saved / 1e6
        }
    
    def print_report(self):
        """Print formatted evaluation report"""
        if not self.results:
            logger.warning("No results to print. Run evaluate() first.")
            return
        
        print("\n" + "="*60)
        print("DTFS MODEL EVALUATION REPORT")
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
        plt.title('DTFS Confusion Matrix')
        
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
                                 test_time_minutes: float = 30.0) -> Dict:
    """
    Calculate production deployment metrics
    
    Args:
        flags_df: DataFrame with columns [LOT_ID, CHIP_ID, FLAG]
        test_suite: Name of test suite
        test_time_minutes: Test time per chip
    
    Returns:
        Production metrics dictionary
    """
    total_chips = len(flags_df)
    skipped_chips = (flags_df['FLAG'] == 0).sum()
    tested_chips = (flags_df['FLAG'] == 1).sum()
    
    skip_rate = skipped_chips / total_chips
    time_saved_hours = skipped_chips * test_time_minutes / 60
    
    metrics = {
        'test_suite': test_suite,
        'total_chips': total_chips,
        'skipped_chips': skipped_chips,
        'tested_chips': tested_chips,
        'skip_rate': skip_rate,
        'time_saved_hours': time_saved_hours,
        'time_reduction_percent': skip_rate * 100
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


if __name__ == "__main__":
    print("=== DTFS Evaluator Example ===\n")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 10000
    
    # Simulate 98% pass rate
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    
    # Simulate predictions with some errors
    y_pred = y_true.copy()
    # Add 0.5% overreject (FP)
    overreject_idx = np.random.choice(
        np.where(y_true == 0)[0], 
        size=int(0.005 * len(y_true)), 
        replace=False
    )
    y_pred[overreject_idx] = 1
    
    # Simulate probabilities
    y_proba = np.where(y_pred == 1, 
                       np.random.uniform(0.7, 1.0, n_samples),
                       np.random.uniform(0.0, 0.3, n_samples))
    
    # Evaluate
    evaluator = DTFSEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, y_proba)
    evaluator.print_report()
    
    # Calculate savings
    print("\n💰 TIME & COST SAVINGS:")
    savings = evaluator.calculate_time_savings(
        y_pred, 
        test_time_per_chip=30.0,
        cost_per_hour=100.0,
        n_lots=400,
        chips_per_lot=13000
    )
    print(f"  Skip Rate:         {savings['skip_rate']:.2%}")
    print(f"  Time Saved:        {savings['total_time_saved_hours']:.0f} hours")
    print(f"  Cost Saved:        €{savings['estimated_cost_saved_eur']:,.0f}")
    print(f"  Cost Saved:        €{savings['cost_saved_millions']:.2f}M")
    
    print("\n✓ Evaluation complete!")
