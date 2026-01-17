"""
Flag Generation for Production Test Flow

This script generates binary test flags for production lots using the trained ensemble.
Each chip receives a flag per test suite: 0 = SKIP test, 1 = RUN test.

Output Format: SORTFILE (text file with chip IDs and binary flags)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import sys
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.ensemble import HybridEnsemble
from preprocessing.preprocessing import preprocess_pipeline, FeatureScaler, DataValidator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FlagGenerator:
    """
    Generate test skip flags for production lots
    
    Workflow:
    1. Load test data (chip IDs + test measurements)
    2. Preprocess features
    3. Run ensemble inference
    4. Generate sortfile (chip ID + flags)
    5. Save results
    
    Example:
        >>> generator = FlagGenerator(ensemble, scaler)
        >>> flags_df = generator.generate_flags(data_df)
        >>> generator.save_sortfile(flags_df, 'output.txt')
    """
    
    def __init__(self, ensemble: HybridEnsemble, scaler: FeatureScaler):
        """
        Initialize flag generator
        
        Args:
            ensemble: Trained HybridEnsemble model
            scaler: Fitted FeatureScaler
        """
        self.ensemble = ensemble
        self.scaler = scaler
        self.validator = DataValidator()
    
    def generate_flags(self, data: pd.DataFrame, 
                      chip_id_col: str = 'CHIP_ID') -> pd.DataFrame:
        """
        Generate test flags for a production lot
        
        Args:
            data: DataFrame with chip IDs and test features
            chip_id_col: Name of chip ID column
        
        Returns:
            DataFrame with columns: [CHIP_ID, FLAG]
            where FLAG is 0 (SKIP) or 1 (TEST)
        """
        logger.info(f"Generating flags for {len(data)} chips...")
        
        # Separate chip IDs from features
        chip_ids = data[chip_id_col].copy()
        features = data.drop(columns=[chip_id_col])
        
        # Preprocess
        logger.info("Preprocessing features...")
        features_scaled, metadata = preprocess_pipeline(
            features, 
            self.scaler,
            self.validator
        )
        
        # Generate flags
        logger.info("Running ensemble inference...")
        flags = self.ensemble.predict(features_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({
            chip_id_col: chip_ids,
            'FLAG': flags
        })
        
        # Log summary statistics
        skip_rate = (flags == 0).mean()
        logger.info(f"Flag generation complete:")
        logger.info(f"  - Total chips: {len(flags)}")
        logger.info(f"  - SKIP (0): {(flags == 0).sum()} ({skip_rate:.2%})")
        logger.info(f"  - TEST (1): {(flags == 1).sum()} ({1-skip_rate:.2%})")
        
        return results
    
    def generate_flags_with_details(self, data: pd.DataFrame,
                                    chip_id_col: str = 'CHIP_ID') -> pd.DataFrame:
        """
        Generate flags with per-model breakdown
        
        Args:
            data: DataFrame with chip IDs and test features
            chip_id_col: Name of chip ID column
        
        Returns:
            DataFrame with columns: [CHIP_ID, FLAG, CLASSIFIER, VAE, SIGMA]
        """
        logger.info(f"Generating detailed flags for {len(data)} chips...")
        
        chip_ids = data[chip_id_col].copy()
        features = data.drop(columns=[chip_id_col])
        
        # Preprocess
        features_scaled, _ = preprocess_pipeline(features, self.scaler, self.validator)
        
        # Get detailed predictions
        predictions = self.ensemble.predict_with_details(features_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({
            chip_id_col: chip_ids,
            'FLAG': predictions['ensemble'],
            'CLASSIFIER': predictions.get('classifier', 0),
            'VAE': predictions.get('vae', 0),
            'SIGMA': predictions.get('sigma', 0)
        })
        
        # Log per-model statistics
        for model in ['CLASSIFIER', 'VAE', 'SIGMA']:
            if model in results.columns:
                flag_rate = results[model].mean()
                logger.info(f"  - {model}: {flag_rate:.2%} flagged")
        
        return results
    
    def save_sortfile(self, flags_df: pd.DataFrame, filepath: str,
                     chip_id_col: str = 'CHIP_ID', flag_col: str = 'FLAG'):
        """
        Save flags in SORTFILE format for test equipment
        
        Format: Each line is "CHIP_ID FLAG"
        Example:
            CHIP_00001 1
            CHIP_00002 0
            CHIP_00003 1
        
        Args:
            flags_df: DataFrame with chip IDs and flags
            filepath: Output file path
            chip_id_col: Name of chip ID column
            flag_col: Name of flag column
        """
        logger.info(f"Saving sortfile to {filepath}...")
        
        with open(filepath, 'w') as f:
            for _, row in flags_df.iterrows():
                f.write(f"{row[chip_id_col]} {row[flag_col]}\n")
        
        logger.info(f"Sortfile saved: {len(flags_df)} entries written")
    
    def calculate_savings(self, flags: np.ndarray, 
                         test_time_per_chip: float = 30.0) -> Dict:
        """
        Calculate test time and cost savings
        
        Args:
            flags: Binary flags array
            test_time_per_chip: Test time per chip in minutes
        
        Returns:
            Dictionary with savings metrics
        """
        n_total = len(flags)
        n_skipped = (flags == 0).sum()
        skip_rate = n_skipped / n_total
        
        time_saved_minutes = n_skipped * test_time_per_chip
        time_saved_hours = time_saved_minutes / 60
        
        # Cost estimation (example: €100 per tester-hour)
        cost_per_hour = 100
        cost_saved = time_saved_hours * cost_per_hour
        
        return {
            'skip_rate': skip_rate,
            'chips_skipped': n_skipped,
            'chips_tested': n_total - n_skipped,
            'time_saved_minutes': time_saved_minutes,
            'time_saved_hours': time_saved_hours,
            'time_reduction_percent': skip_rate * 100,
            'estimated_cost_saved_eur': cost_saved
        }


def main():
    """
    Main execution function for CLI usage
    
    Example usage:
        python generate_flags.py --input data.csv --output sortfile.txt
    """
    parser = argparse.ArgumentParser(description='Generate test skip flags')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file with test data')
    parser.add_argument('--output', type=str, required=True,
                       help='Output sortfile path')
    parser.add_argument('--ensemble-path', type=str, default='models/ensemble.pth',
                       help='Path to trained ensemble model')
    parser.add_argument('--scaler-path', type=str, default='models/scaler.pkl',
                       help='Path to fitted scaler')
    parser.add_argument('--chip-id-col', type=str, default='CHIP_ID',
                       help='Name of chip ID column')
    parser.add_argument('--detailed', action='store_true',
                       help='Include per-model breakdown in output')
    
    args = parser.parse_args()
    
    # Load models (placeholder - adjust based on your model loading)
    logger.info("Loading models...")
    # ensemble = HybridEnsemble.load(args.ensemble_path)
    # scaler = FeatureScaler.load(args.scaler_path)
    
    logger.warning("⚠️  Model loading not implemented in example code")
    logger.warning("Please implement model loading based on your setup")
    
    # Load data
    logger.info(f"Loading data from {args.input}...")
    data = pd.read_csv(args.input)
    logger.info(f"Loaded {len(data)} chips with {len(data.columns)} features")
    
    # Generate flags (placeholder)
    logger.info("Flag generation would happen here")
    logger.info(f"Output would be saved to {args.output}")
    
    # Example of what the actual implementation would look like:
    """
    generator = FlagGenerator(ensemble, scaler)
    
    if args.detailed:
        flags_df = generator.generate_flags_with_details(data, args.chip_id_col)
    else:
        flags_df = generator.generate_flags(data, args.chip_id_col)
    
    generator.save_sortfile(flags_df, args.output, args.chip_id_col)
    
    # Calculate and log savings
    savings = generator.calculate_savings(flags_df['FLAG'].values)
    logger.info(f"Time savings: {savings['time_saved_hours']:.1f} hours")
    logger.info(f"Skip rate: {savings['skip_rate']:.2%}")
    """


if __name__ == "__main__":
    # For demonstration, create a simple example
    print("=== Flag Generation Example ===\n")
    
    print("This script would:")
    print("1. Load trained ensemble model and scaler")
    print("2. Read production lot data (CSV with chip IDs + test measurements)")
    print("3. Preprocess features")
    print("4. Generate binary flags: 0=SKIP, 1=TEST")
    print("5. Save sortfile for test equipment")
    print("6. Calculate time/cost savings")
    
    print("\nExample output format (sortfile.txt):")
    print("-" * 40)
    print("CHIP_00001 1")
    print("CHIP_00002 0")
    print("CHIP_00003 0")
    print("CHIP_00004 1")
    print("...")
    print("-" * 40)
    
    print("\nTo use in production:")
    print("python generate_flags.py --input lot_M4287Z_P10.csv --output sortfile.txt")
