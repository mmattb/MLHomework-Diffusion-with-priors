"""
Image Denoiser Network for Model A and Model B.

★★★ TODO: YOU NEED TO IMPLEMENT THE forward() METHOD ★★★

This model predicts noise given a noisy image, timestep, and conditioning.
"""

import torch
import torch.nn as nn
from .time_embedding import SinusoidalTimeEmbedding
from .unet import SimpleUNet


class ImageDenoiser(nn.Module):
    """
    Image denoising model for diffusion.

    This network takes a noisy image x_t, a timestep t, and conditioning information,
    and predicts the noise that was added to create x_t.

    Architecture:
        1. Time embedding: Convert timestep to sinusoidal embedding
        2. Condition embedding: Project conditioning to embedding space
        3. Combine embeddings: Concatenate time and condition embeddings
        4. UNet backbone: Process noisy image with combined embeddings
        5. Output: Predicted noise
    """

    def __init__(
        self,
        image_channels: int = 1,
        condition_dim: int = 2,  # For z2 one-hot encoding
        time_embed_dim: int = 32,
        model_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4),
    ):
        """
        Initialize the image denoiser.

        Args:
            image_channels: Number of image channels (1 for grayscale)
            condition_dim: Dimension of conditioning input
                          For Model A: 2 (z2 one-hot)
                          For Model B: 2 (z1 embedding)
            time_embed_dim: Dimension of time embeddings
            model_channels: Base number of channels in UNet
            channel_multipliers: Channel multipliers for each UNet level
        """
        super().__init__()

        self.image_channels = image_channels
        self.condition_dim = condition_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(embedding_dim=time_embed_dim)

        # Condition embedding (project to same dimension as time embedding)
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Combined embedding dimension (time + condition)
        combined_dim = time_embed_dim * 2

        # UNet backbone (provided - you don't need to modify this)
        self.unet = SimpleUNet(
            in_channels=image_channels,
            out_channels=image_channels,
            embed_dim=combined_dim,
            base_channels=model_channels,
            channel_multipliers=channel_multipliers,
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise from noisy image.

        ★★★ TODO: IMPLEMENT THIS METHOD ★★★

        Args:
            x_t: Noisy images at timestep t
                Shape: (batch_size, 1, 64, 64)
                Values: Normalized to [-1, 1]

            t: Diffusion timesteps
                Shape: (batch_size,)
                Values: Integers in [0, num_timesteps-1], e.g., [0, 999]

            condition: Conditioning information
                Shape: (batch_size, condition_dim)
                For Model A: z2 one-hot encoding, shape (batch_size, 2)
                For Model B: z1 embedding, shape (batch_size, 2)

        Returns:
            noise_pred: Predicted noise
                Shape: (batch_size, 1, 64, 64)

        Implementation steps:
            1. Get time embeddings using self.time_embed(t)
               - Input: t with shape (batch_size,)
               - Output: time_emb with shape (batch_size, time_embed_dim)

            2. Get condition embeddings using self.condition_embed(condition)
               - Input: condition with shape (batch_size, condition_dim)
               - Output: cond_emb with shape (batch_size, time_embed_dim)

            3. Combine time and condition embeddings
               - Concatenate: combined = torch.cat([time_emb, cond_emb], dim=1)
               - Result shape: (batch_size, time_embed_dim * 2)

            4. Pass through UNet: self.unet(x_t, combined)
               - Input: x_t with shape (batch_size, 1, 64, 64)
                       combined with shape (batch_size, time_embed_dim * 2)
               - Output: noise_pred with shape (batch_size, 1, 64, 64)

            5. Return the predicted noise

        Debugging tips:
            - Print shapes at each step to verify correctness
            - Check that output shape matches input image shape
            - Verify that embeddings are properly combined
        """
        # ============================================================
        # YOUR CODE HERE
        # ============================================================

        raise NotImplementedError(
            "You need to implement the forward() method of ImageDenoiser. "
            "See the docstring above for detailed instructions."
        )

        # ============================================================
        # END YOUR CODE
        # ============================================================


if __name__ == "__main__":
    """Test the ImageDenoiser (will fail until you implement forward())."""

    print("Testing ImageDenoiser...")
    print("=" * 60)

    # Create model
    model = ImageDenoiser(
        image_channels=1,
        condition_dim=2,  # Model A: z2 one-hot
        time_embed_dim=32,
        model_channels=64,
    )

    print(f"Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create dummy inputs
    batch_size = 4
    x_t = torch.randn(batch_size, 1, 64, 64)
    t = torch.randint(0, 1000, (batch_size,))
    condition = torch.randn(batch_size, 2)  # z2 one-hot

    print("Input shapes:")
    print(f"  x_t: {x_t.shape}")
    print(f"  t: {t.shape}")
    print(f"  condition: {condition.shape}")
    print()

    try:
        # Forward pass
        noise_pred = model(x_t, t, condition)

        print("✓ Forward pass successful!")
        print(f"  Output shape: {noise_pred.shape}")
        print()

        # Verify output shape
        assert (
            noise_pred.shape == x_t.shape
        ), f"Output shape {noise_pred.shape} doesn't match input shape {x_t.shape}"

        print("✓ All tests passed!")

    except NotImplementedError as e:
        print("✗ Forward pass not implemented yet.")
        print(f"  {e}")
        print()
        print(
            "Please implement the forward() method following the instructions in the docstring."
        )
