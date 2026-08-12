"""Tests for deterministic VAE anomaly scoring."""

import torch

from models.anomaly_detection.vae_model import VariationalAutoEncoder


def test_reconstruction_error_is_repeatable_in_evaluation():
    torch.manual_seed(42)
    model = VariationalAutoEncoder(input_size=8, latent_size=2, num_layers=3)
    features = torch.randn(16, 8)

    first = model.get_reconstruction_error(features)
    second = model.get_reconstruction_error(features)

    assert torch.equal(first, second)