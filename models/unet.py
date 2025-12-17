"""
Simple UNet backbone for image generation.

This is a minimal UNet implementation for the denoising model.
You don't need to modify this - it's provided as the architecture backbone.
"""

import torch
import torch.nn as nn
from typing import Optional


class ResidualBlock(nn.Module):
    """Residual block with time and condition embedding injection."""

    def __init__(self, in_channels: int, out_channels: int, embed_dim: int):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

        self.embedding_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dim, out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, in_channels, H, W)
            embed: Embedding tensor of shape (batch, embed_dim)
        Returns:
            Output tensor of shape (batch, out_channels, H, W)
        """
        h = self.conv1(x)

        # Add embedding (broadcast over spatial dimensions)
        emb = self.embedding_proj(embed)[:, :, None, None]
        h = h + emb

        h = self.conv2(h)

        return h + self.skip(x)


class DownBlock(nn.Module):
    """Downsampling block."""

    def __init__(self, in_channels: int, out_channels: int, embed_dim: int):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, embed_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> tuple:
        h = self.res_block(x, embed)
        return self.downsample(h), h


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, embed_dim: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, 4, stride=2, padding=1
        )
        self.res_block = ResidualBlock(
            in_channels + skip_channels, out_channels, embed_dim
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, embed: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.res_block(x, embed)


class SimpleUNet(nn.Module):
    """
    Simple UNet for image denoising.

    This is a minimal UNet with downsampling and upsampling paths.
    It takes an image and embeddings (time + condition) and outputs
    the predicted noise.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 256,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4),
    ):
        """
        Args:
            in_channels: Number of input image channels (1 for grayscale)
            out_channels: Number of output channels (1 for grayscale noise)
            embed_dim: Dimension of combined embeddings
            base_channels: Base number of channels
            channel_multipliers: Channel multiplier at each resolution
        """
        super().__init__()

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = [base_channels * m for m in channel_multipliers]
        in_ch = base_channels
        for out_ch in channels:
            self.down_blocks.append(DownBlock(in_ch, out_ch, embed_dim))
            in_ch = out_ch

        # Middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(channels[-1], channels[-1], embed_dim),
            ResidualBlock(channels[-1], channels[-1], embed_dim),
        )

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        channels_reversed = list(reversed(channels))
        for i in range(len(channels_reversed)):
            in_ch = channels_reversed[i]  # Channels from below
            skip_ch = channels_reversed[i]  # Channels from skip connection
            out_ch = channels_reversed[i + 1] if i + 1 < len(channels_reversed) else base_channels
            # UpBlock: upsample in_ch, concatenate with skip_ch, output out_ch
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch, embed_dim))

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through UNet.

        Args:
            x: Input image of shape (batch, in_channels, H, W)
            embed: Combined embeddings of shape (batch, embed_dim)

        Returns:
            Output of shape (batch, out_channels, H, W)
        """
        # Initial conv
        h = self.init_conv(x)

        # Downsampling with skip connections
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, embed)
            skips.append(skip)

        # Middle
        for mid_block in self.middle:
            h = mid_block(h, embed)

        # Upsampling with skip connections
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            h = up_block(h, skip, embed)

        # Final conv
        return self.final_conv(h)


if __name__ == "__main__":
    # Test UNet
    batch_size = 2
    unet = SimpleUNet(in_channels=1, out_channels=1, embed_dim=256, base_channels=64)

    x = torch.randn(batch_size, 1, 64, 64)
    embed = torch.randn(batch_size, 256)

    out = unet(x, embed)
    print(f"Input shape: {x.shape}")
    print(f"Embedding shape: {embed.shape}")
    print(f"Output shape: {out.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in unet.parameters())
    print(f"Number of parameters: {num_params:,}")
