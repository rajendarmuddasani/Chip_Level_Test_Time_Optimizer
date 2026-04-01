"""
Neural Network Classifier for Test Skip Prediction

This module implements a simple feedforward neural network for binary classification
of chips into PASS/FAIL categories to determine which tests can be safely skipped.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ChipTestClassifier(nn.Module):
    """
    Binary classifier for test skip prediction
    
    Architecture:
        - Input layer: Accepts test features
        - Hidden layer: Small hidden layer with ReLU activation
        - Dropout: Regularization to prevent overfitting
        - Output layer: 2 classes (PASS/FAIL)
    
    Args:
        input_size: Number of input features
        hidden_size: Number of neurons in hidden layer
        dropout: Dropout probability for regularization
    
    Example:
        >>> model = ChipTestClassifier(input_size=202, hidden_size=4, dropout=0.5)
        >>> x = torch.randn(32, 202)  # Batch of 32 chips
        >>> logits = model(x)
        >>> probs = model.predict_proba(x)
    """
    
    def __init__(self, input_size: int = 202, hidden_size: int = 4, dropout: float = 0.5):
        super(ChipTestClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        
        # Layer 1: Input → Hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2: Hidden → Output
        self.fc2 = nn.Linear(hidden_size, 2)  # Binary classification
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input features of shape (batch_size, input_size)
        
        Returns:
            Logits of shape (batch_size, 2)
        """
        # Hidden layer with ReLU activation
        out = F.relu(self.fc1(x))
        
        # Dropout (active during training)
        out = self.dropout1(out)
        
        # Output layer (logits, no activation)
        out = self.fc2(out)
        
        return out
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions
        
        Args:
            x: Input features of shape (batch_size, input_size)
        
        Returns:
            Probabilities of shape (batch_size, 2) where:
            - [:, 0] = P(PASS)
            - [:, 1] = P(FAIL)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions
        
        Args:
            x: Input features
            threshold: Decision threshold for classification
        
        Returns:
            Binary predictions: 0 = PASS (SKIP test), 1 = FAIL (RUN test)
        """
        probs = self.predict_proba(x)
        return (probs[:, 1] >= threshold).long()


class CustomDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for chip test data
    
    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Target array of shape (n_samples,)
    
    Example:
        >>> dataset = CustomDataset(X_train, y_train)
        >>> loader = torch.utils.data.DataLoader(dataset, batch_size=4096)
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Convert to float32 for PyTorch
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.len = self.X.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]
    
    def __len__(self) -> int:
        return self.len


def create_model(input_size: int, config: dict) -> ChipTestClassifier:
    """
    Factory function to create model with configuration
    
    Args:
        input_size: Number of input features
        config: Dictionary containing model hyperparameters
    
    Returns:
        Initialized model
    
    Example:
        >>> config = {'hidden_size': 4, 'dropout': 0.5}
        >>> model = create_model(input_size=202, config=config)
    """
    return ChipTestClassifier(
        input_size=input_size,
        hidden_size=config.get('hidden_size', 4),
        dropout=config.get('dropout', 0.5)
    )


if __name__ == "__main__":
    # Test the model
    import numpy as np
    
    # Create dummy data
    X = np.random.randn(100, 202)
    y = np.random.randint(0, 2, 100)
    
    # Create model
    model = ChipTestClassifier(input_size=202, hidden_size=4)
    
    # Create dataset and loader
    dataset = CustomDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # Test forward pass
    for batch_X, batch_y in loader:
        logits = model(batch_X)
        probs = model.predict_proba(batch_X)
        preds = model.predict(batch_X, threshold=0.2)
        
        print(f"Batch shape: {batch_X.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Probabilities shape: {probs.shape}")
        print(f"Predictions shape: {preds.shape}")
        break
    
    print("\n✓ Model test passed!")
