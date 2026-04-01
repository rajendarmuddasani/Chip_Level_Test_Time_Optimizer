"""
Example Training Script for Chip Test Classifier

This script demonstrates the training workflow for the neural network classifier.
In production, this would be adapted to your specific data pipeline and infrastructure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.classification.classifier_model import ChipTestClassifier, CustomDataset
from preprocessing.preprocessing import FeatureScaler, preprocess_pipeline
from evaluation.metrics import TestTimeEvaluator


def train_classifier(config: dict):
    """
    Train chip test classifier
    
    Args:
        config: Training configuration dictionary
    """
    print("="*60)
    print("Chip Test Classifier Training")
    print("="*60)
    
    # 1. Load Data
    print("\n[1/7] Loading data...")
    # In production: data = load_from_database(config['query'])
    # For demo:
    data = pd.read_csv(config.get('data_path', 'data/training_data.csv'))
    print(f"  Loaded {len(data)} samples with {len(data.columns)} features")
    
    # Separate features and labels
    X = data.drop(columns=['label'])
    y = data['label'].values
    
    # 2. Preprocessing
    print("\n[2/7] Preprocessing...")
    scaler = FeatureScaler(method='robust')
    X_scaled = scaler.fit_transform(X)
    print(f"  Features scaled using {scaler.method} method")
    
    # 3. Train/Test Split
    print("\n[3/7] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Class distribution: {np.bincount(y_train)}")
    
    # 4. Create DataLoaders
    print("\n[4/7] Creating dataloaders...")
    train_dataset = CustomDataset(X_train.values, y_train)
    test_dataset = CustomDataset(X_test.values, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    print(f"  Batch size: {config['batch_size']}")
    
    # 5. Initialize Model
    print("\n[5/7] Initializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    model = ChipTestClassifier(
        input_size=X_train.shape[1],
        hidden_size=config['hidden_size'],
        dropout=config['dropout']
    ).to(device)
    
    print(f"  Architecture: {X_train.shape[1]} → {config['hidden_size']} → 2")
    
    # 6. Training
    print("\n[6/7] Training...")
    
    # Loss function with class weights (handle imbalance)
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor([1.0, class_counts[0] / class_counts[1]]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    best_loss = float('inf')
    patience = config.get('patience', 10)
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{config['epochs']}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config['output_path'])
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print(f"  Training complete. Best val loss: {best_loss:.4f}")
    
    # 7. Evaluation
    print("\n[7/7] Evaluating...")
    model.load_state_dict(torch.load(config['output_path']))
    model.eval()
    
    # Get predictions
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            probs = model.predict_proba(batch_X)
            preds = model.predict(batch_X, threshold=config.get('threshold', 0.2))
            
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Evaluate
    evaluator = TestTimeEvaluator()
    metrics = evaluator.evaluate(y_test, all_preds, all_probs)
    evaluator.print_report()
    
    # Save scaler
    scaler_path = config['output_path'].replace('.pth', '_scaler.pkl')
    scaler.save(scaler_path)
    print(f"\n✓ Model saved to: {config['output_path']}")
    print(f"✓ Scaler saved to: {scaler_path}")
    
    return model, scaler, metrics


if __name__ == "__main__":
    # Example configuration
    config = {
        'data_path': 'data/training_data.csv',
        'output_path': 'models/classifier_trained.pth',
        'batch_size': 4096,
        'hidden_size': 4,
        'dropout': 0.5,
        'learning_rate': 0.001,
        'epochs': 111,
        'patience': 15,
        'threshold': 0.2  # Lower threshold = more conservative
    }
    
    print("\n⚠️  This is an example training script")
    print("In production, adapt this to your data pipeline\n")
    
    # If you have training data, uncomment to run:
    # model, scaler, metrics = train_classifier(config)
    
    print("To train:")
    print("1. Prepare training data as CSV with features + 'label' column")
    print("2. Update config['data_path']")
    print("3. Run: python examples/train_example.py")
