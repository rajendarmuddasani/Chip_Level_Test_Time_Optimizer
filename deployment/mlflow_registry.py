"""
MLflow Model Registry Integration

This module handles model versioning, registration, and deployment using MLflow.
Supports A/B testing and model rollback capabilities.
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pathlib import Path
import logging
from typing import Dict, Optional
import torch
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manage model lifecycle with MLflow
    
    Features:
    - Model versioning
    - Artifact storage (models, scalers, config)
    - Model promotion (Staging → Production)
    - A/B testing support
    
    Example:
        >>> registry = ModelRegistry(tracking_uri='http://localhost:5000')
        >>> registry.register_model(model, 'TF_ADC_1', metadata)
        >>> model = registry.load_production_model('TF_ADC_1')
    """
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = 'chip_test_optimizer'):
        """
        Initialize MLflow registry
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of MLflow experiment
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
        self.client = MlflowClient()
        logger.info(f"MLflow registry initialized: {experiment_name}")
    
    def register_model(self, 
                      model: torch.nn.Module,
                      model_name: str,
                      scaler: Optional[object] = None,
                      metadata: Optional[Dict] = None,
                      tags: Optional[Dict] = None) -> str:
        """
        Register a new model version
        
        Args:
            model: Trained PyTorch model
            model_name: Name for model registry (e.g., 'TF_ADC_1')
            scaler: Fitted feature scaler
            metadata: Additional metadata (metrics, config, etc.)
            tags: MLflow tags
        
        Returns:
            Model version string
        """
        with mlflow.start_run() as run:
            # Log model
            mlflow.pytorch.log_model(model, "model")
            
            # Log scaler as artifact
            if scaler is not None:
                scaler_path = "scaler.pkl"
                joblib.dump(scaler, scaler_path)
                mlflow.log_artifact(scaler_path)
            
            # Log metadata
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(key, value)
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            mlflow.log_param(f"{key}.{subkey}", subvalue)
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, model_name)
            
            logger.info(f"Registered {model_name} version {mv.version}")
            return mv.version
    
    def promote_to_production(self, model_name: str, version: int):
        """
        Promote a model version to Production stage
        
        Args:
            model_name: Name of registered model
            version: Version number to promote
        """
        # Archive current production version
        current_prod = self.client.get_latest_versions(
            model_name, stages=["Production"]
        )
        for mv in current_prod:
            self.client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Archived"
            )
        
        # Promote new version
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        logger.info(f"Promoted {model_name} v{version} to Production")
    
    def load_model(self, model_name: str, 
                   version: Optional[int] = None,
                   stage: str = "Production") -> Dict:
        """
        Load a registered model
        
        Args:
            model_name: Name of registered model
            version: Specific version (or None for latest in stage)
            stage: Model stage ('Production', 'Staging', etc.)
        
        Returns:
            Dictionary with 'model', 'scaler', and 'metadata'
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{stage}"
        
        # Load model
        model = mlflow.pytorch.load_model(model_uri)
        
        # Load artifacts from the same run
        run_id = self.client.get_model_version(model_name, version).run_id
        artifact_path = self.client.download_artifacts(run_id, "")
        
        # Load scaler if exists
        scaler = None
        scaler_path = Path(artifact_path) / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        
        # Load run metadata
        run = self.client.get_run(run_id)
        metadata = {
            'params': run.data.params,
            'metrics': run.data.metrics,
            'tags': run.data.tags
        }
        
        logger.info(f"Loaded {model_name} from {stage} stage")
        
        return {
            'model': model,
            'scaler': scaler,
            'metadata': metadata
        }
    
    def compare_models(self, model_name: str, 
                      version_a: int, 
                      version_b: int) -> Dict:
        """
        Compare two model versions
        
        Args:
            model_name: Name of model
            version_a: First version to compare
            version_b: Second version to compare
        
        Returns:
            Dictionary with comparison metrics
        """
        # Get run IDs
        mv_a = self.client.get_model_version(model_name, version_a)
        mv_b = self.client.get_model_version(model_name, version_b)
        
        # Get metrics
        run_a = self.client.get_run(mv_a.run_id)
        run_b = self.client.get_run(mv_b.run_id)
        
        comparison = {
            'version_a': version_a,
            'version_b': version_b,
            'metrics_a': run_a.data.metrics,
            'metrics_b': run_b.data.metrics,
        }
        
        # Calculate differences
        common_metrics = set(run_a.data.metrics.keys()) & set(run_b.data.metrics.keys())
        differences = {}
        for metric in common_metrics:
            diff = run_b.data.metrics[metric] - run_a.data.metrics[metric]
            differences[metric] = {
                'a': run_a.data.metrics[metric],
                'b': run_b.data.metrics[metric],
                'diff': diff,
                'percent_change': (diff / run_a.data.metrics[metric] * 100) 
                                 if run_a.data.metrics[metric] != 0 else 0
            }
        
        comparison['differences'] = differences
        
        return comparison
    
    def list_models(self, name_filter: Optional[str] = None) -> list:
        """
        List all registered models
        
        Args:
            name_filter: Optional filter for model names
        
        Returns:
            List of model information
        """
        models = []
        for rm in self.client.search_registered_models():
            if name_filter and name_filter not in rm.name:
                continue
            
            latest_versions = self.client.get_latest_versions(rm.name)
            models.append({
                'name': rm.name,
                'description': rm.description,
                'versions': [mv.version for mv in latest_versions],
                'latest_version': max([mv.version for mv in latest_versions])
            })
        
        return models


def example_registration():
    """
    Example: Register a model with MLflow
    """
    print("=== MLflow Model Registration Example ===\n")
    
    # Initialize registry (Note: requires MLflow server running)
    print("1. Initialize registry:")
    print("   registry = ModelRegistry(tracking_uri='http://localhost:5000')")
    
    print("\n2. Train your model:")
    print("   model = ChipTestClassifier(input_size=202, hidden_size=4)")
    print("   # ... training code ...")
    
    print("\n3. Register model:")
    print("   metadata = {")
    print("       'accuracy': 0.95,")
    print("       'skip_rate': 0.15,")
    print("       'escapee_rate': 0.0")
    print("   }")
    print("   version = registry.register_model(model, 'TF_ADC_1', scaler, metadata)")
    
    print("\n4. Promote to production:")
    print("   registry.promote_to_production('TF_ADC_1', version)")
    
    print("\n5. Load in production:")
    print("   loaded = registry.load_model('TF_ADC_1', stage='Production')")
    print("   model = loaded['model']")
    print("   scaler = loaded['scaler']")
    
    print("\n6. A/B test two versions:")
    print("   comparison = registry.compare_models('TF_ADC_1', version_a=1, version_b=2)")
    print("   print(comparison['differences'])")


if __name__ == "__main__":
    example_registration()
    
    print("\n" + "="*60)
    print("Note: This example requires MLflow server to be running.")
    print("Start server with: mlflow server --host 0.0.0.0 --port 5000")
    print("="*60)
