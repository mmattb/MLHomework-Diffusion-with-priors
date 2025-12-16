"""
Time embedding module for diffusion models.

Sinusoidal position embeddings adapted for timesteps.
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embeddings for diffusion timesteps.

    This is similar to positional encodings in transformers, but adapted
    for timestep information in diffusion models.
    """

    def __init__(self, embedding_dim: int, max_period: int = 10000):
        """
        Args:
            embedding_dim: Dimension of the output embedding
            max_period: Maximum period for sinusoidal encoding
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for timesteps.

        Args:
            timesteps: Tensor of shape (batch_size,) containing timestep indices

        Returns:
            Embeddings of shape (batch_size, embedding_dim)
        """
        # Ensure timesteps is 1D
        assert (
            timesteps.ndim == 1
        ), f"Expected 1D timesteps, got shape {timesteps.shape}"

        half_dim = self.embedding_dim // 2

        # Create frequency components
        # emb = exp(log(max_period) * -i / (half_dim - 1))
        # where i ranges from 0 to half_dim-1
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
        )

        # Apply to timesteps
        # Shape: (batch_size, 1) * (1, half_dim) -> (batch_size, half_dim)
        emb = timesteps.float()[:, None] * emb[None, :]

        # Concatenate sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # If embedding_dim is odd, add extra dimension
        if self.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb


class TimeEmbeddingMLP(nn.Module):
    """
    MLP to process time embeddings before injecting into the model.

    This takes the sinusoidal embeddings and projects them to the desired
    dimension through a small MLP.
    """

    def __init__(
        self, time_dim: int, output_dim: int, hidden_dim: Optional[int] = None
    ):
        """
        Args:
            time_dim: Dimension of input time embeddings
            output_dim: Dimension of output
            hidden_dim: Hidden dimension (default: 4 * time_dim)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * time_dim

        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Project time embeddings.

        Args:
            time_emb: Time embeddings of shape (batch_size, time_dim)

        Returns:
            Processed embeddings of shape (batch_size, output_dim)
        """
        return self.mlp(time_emb)


from typing import Optional

if __name__ == "__main__":
    # Test time embeddings
    batch_size = 4
    timesteps = torch.randint(0, 1000, (batch_size,))

    # Sinusoidal embedding
    time_embed = SinusoidalTimeEmbedding(embedding_dim=128)
    emb = time_embed(timesteps)
    print(f"Timesteps: {timesteps}")
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding values (first sample, first 10 dims): {emb[0, :10]}")

    # MLP projection
    mlp = TimeEmbeddingMLP(time_dim=128, output_dim=256)
    projected = mlp(emb)
    print(f"Projected shape: {projected.shape}")
