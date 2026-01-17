"""
Variational Autoencoder (VAE) for Anomaly Detection

This module implements a β-VAE for detecting anomalous chip behavior.
Chips with high reconstruction error are flagged for testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


class VariationalAutoEncoder(nn.Module):
    """
    β-VAE for anomaly detection in semiconductor testing
    
    The model learns to compress normal chip behavior into a low-dimensional
    latent space. Chips that cannot be well-reconstructed are considered
    anomalies and are flagged for testing.
    
    Args:
        input_size: Number of input features
        latent_size: Dimension of latent space (typically 2 for visualization)
        num_layers: Number of layers in encoder/decoder
        beta: β parameter for β-VAE (controls disentanglement)
    
    Example:
        >>> vae = VariationalAutoEncoder(input_size=202, latent_size=2, beta=1.5)
        >>> x = torch.randn(32, 202)
        >>> recon, mu, logvar = vae(x)
        >>> loss = vae.loss_function(recon, x, mu, logvar)
    """
    
    def __init__(self, input_size: int, latent_size: int = 2, 
                 num_layers: int = 2, beta: float = 1.5):
        super(VariationalAutoEncoder, self).__init__()
        
        self.input_size = input_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.beta = beta
        
        # Generate layer sizes (progressive compression)
        layer_sizes = self._generate_layer_sizes()
        
        # Build Encoder: Input → Latent parameters
        self.encoder = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.encoder.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.encoder.append(nn.ReLU())
        
        # Latent space parameters (mean and log-variance)
        self.fc_mu = nn.Linear(layer_sizes[-1], self.latent_size)
        self.fc_logvar = nn.Linear(layer_sizes[-1], self.latent_size)
        
        # Build Decoder: Latent → Reconstruction
        self.decoder = nn.ModuleList()
        for i in range(len(layer_sizes) - 1, 0, -1):
            self.decoder.append(nn.Linear(layer_sizes[i], layer_sizes[i-1]))
            if i > 1:
                self.decoder.append(nn.ReLU())
    
    def _generate_layer_sizes(self) -> List[int]:
        """
        Generate progressive layer sizes for encoder/decoder
        
        Uses exponential decay from input_size to latent_size
        
        Returns:
            List of layer sizes
        """
        sizes = [self.input_size]
        for i in range(1, self.num_layers):
            # Exponential interpolation
            ratio = (self.latent_size / self.input_size) ** (i / (self.num_layers - 1))
            size = int(self.input_size * ratio)
            sizes.append(max(self.latent_size, size))
        return sizes
    
    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for backpropagation through sampling
        
        z = μ + σ * ε, where ε ~ N(0, 1)
        
        Args:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent parameters
        
        Args:
            x: Input features of shape (batch_size, input_size)
        
        Returns:
            mu: Mean of shape (batch_size, latent_size)
            logvar: Log-variance of shape (batch_size, latent_size)
        """
        for layer in self.encoder:
            x = layer(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction
        
        Args:
            z: Latent vector of shape (batch_size, latent_size)
        
        Returns:
            Reconstruction of shape (batch_size, input_size)
        """
        for layer in self.decoder:
            z = layer(z)
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: Encode → Sample → Decode
        
        Args:
            x: Input features
        
        Returns:
            recon: Reconstructed input
            mu: Latent mean
            logvar: Latent log-variance
        """
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        β-VAE loss: Reconstruction + β * KL Divergence
        
        Loss = MSE(x, recon_x) + β * KL(q(z|x) || p(z))
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Latent mean
            logvar: Latent log-variance
        
        Returns:
            Total loss (scalar)
        """
        # Reconstruction loss (Mean Squared Error)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL Divergence: KL(q(z|x) || N(0,1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Normalize by batch size
        
        # Total β-VAE loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate per-sample reconstruction error for anomaly detection
        
        Higher error → More anomalous → Flag for testing
        
        Args:
            x: Input features of shape (batch_size, input_size)
        
        Returns:
            Reconstruction errors of shape (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            # Calculate MSE per sample
            error = F.mse_loss(recon, x, reduction='none').mean(dim=1)
        return error
    
    def detect_anomalies(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Flag anomalous samples based on reconstruction error
        
        Args:
            x: Input features
            threshold: Error threshold for anomaly detection
        
        Returns:
            Binary flags: 0 = normal (SKIP), 1 = anomaly (TEST)
        """
        errors = self.get_reconstruction_error(x)
        return (errors > threshold).long()


def train_vae(model: VariationalAutoEncoder, 
              dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              device: str = 'cpu') -> float:
    """
    Train VAE for one epoch
    
    Args:
        model: VAE model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        
        # Forward pass
        recon, mu, logvar = model(data)
        loss = model.loss_function(recon, data, mu, logvar)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


if __name__ == "__main__":
    # Test the VAE
    print("Testing Variational Autoencoder...")
    
    # Create model
    vae = VariationalAutoEncoder(input_size=202, latent_size=2, num_layers=2, beta=1.5)
    
    # Create dummy data
    x = torch.randn(32, 202)
    
    # Test forward pass
    recon, mu, logvar = vae(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent mean shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Test loss
    loss = vae.loss_function(recon, x, mu, logvar)
    print(f"Loss: {loss.item():.4f}")
    
    # Test anomaly detection
    errors = vae.get_reconstruction_error(x)
    print(f"Reconstruction errors shape: {errors.shape}")
    print(f"Mean error: {errors.mean().item():.4f}")
    
    # Test anomaly flagging
    flags = vae.detect_anomalies(x, threshold=0.5)
    print(f"Anomaly flags: {flags.sum().item()}/{len(flags)} flagged")
    
    print("\n✓ VAE test passed!")
