"""
Latent Prior Network for Model B (hierarchical diffusion).

★★★ TODO: YOU NEED TO IMPLEMENT THE forward() METHOD ★★★

This model learns p(z1 | z2) using diffusion in the latent space.
"""

import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbedding


class LatentPrior(nn.Module):
    """
    Diffusion prior in latent space.

    This network learns to denoise latent embeddings z1, conditioned on z2.
    It operates in the 2-dimensional latent space (normalized embeddings).

    Architecture:
        1. Time embedding: Convert timestep to sinusoidal embedding
        2. Condition embedding: Project z2 conditioning
        3. MLP: Process concatenated [z1_t, time_emb, cond_emb]
        4. Output: Predicted noise in latent space
    """

    def __init__(
        self,
        latent_dim: int = 2,
        condition_dim: int = 2,  # z2 one-hot encoding
        time_embed_dim: int = 32,
        hidden_dims: tuple = (128, 128),
    ):
        """
        Initialize the latent prior.

        Args:
            latent_dim: Dimension of z1 embeddings (2, L2-normalized)
            condition_dim: Dimension of z2 conditioning (2 for one-hot)
            time_embed_dim: Dimension of time embeddings
            hidden_dims: Hidden dimensions for MLP layers
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(embedding_dim=time_embed_dim)

        # Condition embedding (project z2 to embedding space)
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # MLP to process concatenated inputs
        # Input: [z1_t, time_emb, cond_emb]
        input_dim = latent_dim + time_embed_dim + time_embed_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.SiLU(),
                ]
            )
            prev_dim = hidden_dim

        # Output layer: predict noise in latent space
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, z1_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise in latent space.

        ★★★ TODO: IMPLEMENT THIS METHOD ★★★

        Args:
            z1_t: Noisy latent embeddings at timestep t
                Shape: (batch_size, 2)
                Values: These are noisy versions of the normalized z1 embeddings

            t: Diffusion timesteps
                Shape: (batch_size,)
                Values: Integers in [0, num_timesteps-1], e.g., [0, 999]

            condition: Conditioning information (z2)
                Shape: (batch_size, 2)
                Values: One-hot encoding of z2 category

        Returns:
            noise_pred: Predicted noise in latent space
                Shape: (batch_size, 2)
                Should match the shape of z1_t

        Implementation steps:
            1. Get time embeddings using self.time_embed(t)
               - Input: t with shape (batch_size,)
               - Output: time_emb with shape (batch_size, time_embed_dim)

            2. Get condition embeddings using self.condition_embed(condition)
               - Input: condition with shape (batch_size, condition_dim)
               - Output: cond_emb with shape (batch_size, time_embed_dim)

            3. Concatenate all inputs: [z1_t, time_emb, cond_emb]
               - Use: torch.cat([z1_t, time_emb, cond_emb], dim=1)
               - Result shape: (batch_size, latent_dim + 2 * time_embed_dim)

            4. Pass through MLP: self.mlp(concatenated)
               - Input: concatenated with shape (batch_size, input_dim)
               - Output: noise_pred with shape (batch_size, latent_dim)

            5. Return the predicted noise

        Notes:
            - Unlike the image denoiser, this works in latent space (2-dim)
            - The diffusion process in latent space is the same as in image space
            - After denoising, you may want to re-normalize the embeddings

        Debugging tips:
            - Print shapes at each step
            - Verify that output shape is (batch_size, 2)
            - Check that concatenation includes all three components
        """
        # ============================================================
        # YOUR CODE HERE
        # ============================================================

        raise NotImplementedError(
            "You need to implement the forward() method of LatentPrior. "
            "See the docstring above for detailed instructions."
        )

        # ============================================================
        # END YOUR CODE
        # ============================================================


if __name__ == "__main__":
    """Test the LatentPrior (will fail until you implement forward())."""

    print("Testing LatentPrior...")
    print("=" * 60)

    # Create model
    model = LatentPrior(
        latent_dim=2, condition_dim=2, time_embed_dim=128, hidden_dims=(128, 128)
    )

    print(f"Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create dummy inputs
    batch_size = 4
    z1_t = torch.randn(batch_size, 2)
    # Normalize to unit length (like our real z1 embeddings)
    z1_t = z1_t / torch.norm(z1_t, dim=1, keepdim=True)
    t = torch.randint(0, 1000, (batch_size,))
    condition = torch.randn(batch_size, 2)  # z2 one-hot

    print("Input shapes:")
    print(f"  z1_t: {z1_t.shape}")
    print(f"  z1_t norm: {torch.norm(z1_t, dim=1)}")  # Should be ~1.0
    print(f"  t: {t.shape}")
    print(f"  condition: {condition.shape}")
    print()

    try:
        # Forward pass
        noise_pred = model(z1_t, t, condition)

        print("✓ Forward pass successful!")
        print(f"  Output shape: {noise_pred.shape}")
        print()

        # Verify output shape
        assert (
            noise_pred.shape == z1_t.shape
        ), f"Output shape {noise_pred.shape} doesn't match input shape {z1_t.shape}"

        print("✓ All tests passed!")

    except NotImplementedError as e:
        print("✗ Forward pass not implemented yet.")
        print(f"  {e}")
        print()
        print(
            "Please implement the forward() method following the instructions in the docstring."
        )
