"""
Training utilities for diffusion models.

This module provides complete training loops for both Model A (flat) and Model B (hierarchical).
You don't need to modify this - it's provided as scaffolding.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

from models import ImageDenoiser, LatentPrior
from diffusion import DiffusionProcess


class TrainerModelA:
    """
    Trainer for Model A: Flat conditional diffusion.

    This model directly generates images conditioned on z2 only.
    Expected to show mode collapse.
    """

    def __init__(
        self,
        model: ImageDenoiser,
        diffusion: DiffusionProcess,
        optimizer: optim.Optimizer,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.device = device
        self.step = 0

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            # Get data
            images = batch["image"].to(self.device)  # (B, 3, 64, 64)
            z2 = batch["z2"].to(self.device)  # (B,)

            # Create z2 one-hot encoding for conditioning
            batch_size = images.shape[0]
            z2_onehot = torch.zeros(batch_size, 2, device=self.device)
            z2_onehot.scatter_(1, z2.unsqueeze(1), 1.0)

            # Sample random timesteps
            t = torch.randint(
                0, self.diffusion.num_timesteps, (batch_size,), device=self.device
            )

            # Forward diffusion: add noise
            noisy_images, noise = self.diffusion.q_sample(images, t)

            # Predict noise
            noise_pred = self.model(noisy_images, t, z2_onehot)

            # Compute loss (simple MSE)
            loss = nn.functional.mse_loss(noise_pred, noise)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            self.step += 1

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self.step,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        return checkpoint["epoch"]


class TrainerModelB:
    """
    Trainer for Model B: Hierarchical diffusion with latent prior.

    This model has two components:
    1. Latent prior: p(z1 | z2) in latent space
    2. Image decoder: p(image | z1) in image space

    Training alternates between or jointly trains both components.
    """

    def __init__(
        self,
        prior: LatentPrior,
        decoder: ImageDenoiser,
        prior_diffusion: DiffusionProcess,
        decoder_diffusion: DiffusionProcess,
        prior_optimizer: optim.Optimizer,
        decoder_optimizer: optim.Optimizer,
        device: str = "cuda",
    ):
        self.prior = prior.to(device)
        self.decoder = decoder.to(device)
        self.prior_diffusion = prior_diffusion
        self.decoder_diffusion = decoder_diffusion
        self.prior_optimizer = prior_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.device = device
        self.step = 0

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Trains both prior and decoder jointly.

        Returns:
            Dictionary with training metrics
        """
        self.prior.train()
        self.decoder.train()

        total_prior_loss = 0.0
        total_decoder_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            batch_size = batch["image"].shape[0]

            # Get data
            images = batch["image"].to(self.device)  # (B, 3, 64, 64)
            z2 = batch["z2"].to(self.device)  # (B,)
            z1_embedding = batch["z1_embedding"].to(self.device)  # (B, 32)

            # Create z2 one-hot encoding
            z2_onehot = torch.zeros(batch_size, 2, device=self.device)
            z2_onehot.scatter_(1, z2.unsqueeze(1), 1.0)

            # ===== Train Prior: p(z1 | z2) =====
            # Sample random timesteps for prior
            t_prior = torch.randint(
                0, self.prior_diffusion.num_timesteps, (batch_size,), device=self.device
            )

            # Forward diffusion in latent space
            noisy_z1, noise_z1 = self.prior_diffusion.q_sample(z1_embedding, t_prior)

            # Predict noise in latent space
            noise_pred_prior = self.prior(noisy_z1, t_prior, z2_onehot)

            # Compute prior loss
            prior_loss = nn.functional.mse_loss(noise_pred_prior, noise_z1)

            # Update prior
            self.prior_optimizer.zero_grad()
            prior_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.prior.parameters(), 1.0)
            self.prior_optimizer.step()

            # ===== Train Decoder: p(image | z1) =====
            # Sample random timesteps for decoder
            t_decoder = torch.randint(
                0,
                self.decoder_diffusion.num_timesteps,
                (batch_size,),
                device=self.device,
            )

            # Forward diffusion in image space
            noisy_images, noise_images = self.decoder_diffusion.q_sample(
                images, t_decoder
            )

            # Predict noise in image space, conditioned on ground truth z1
            noise_pred_decoder = self.decoder(noisy_images, t_decoder, z1_embedding)

            # Compute decoder loss
            decoder_loss = nn.functional.mse_loss(noise_pred_decoder, noise_images)

            # Update decoder
            self.decoder_optimizer.zero_grad()
            decoder_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
            self.decoder_optimizer.step()

            # Track metrics
            total_prior_loss += prior_loss.item()
            total_decoder_loss += decoder_loss.item()
            num_batches += 1
            self.step += 1

            pbar.set_postfix(
                {"prior_loss": prior_loss.item(), "decoder_loss": decoder_loss.item()}
            )

        avg_prior_loss = total_prior_loss / num_batches
        avg_decoder_loss = total_decoder_loss / num_batches

        return {
            "prior_loss": avg_prior_loss,
            "decoder_loss": avg_decoder_loss,
            "total_loss": avg_prior_loss + avg_decoder_loss,
        }

    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "prior_state_dict": self.prior.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "prior_optimizer_state_dict": self.prior_optimizer.state_dict(),
                "decoder_optimizer_state_dict": self.decoder_optimizer.state_dict(),
                "step": self.step,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.prior.load_state_dict(checkpoint["prior_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.prior_optimizer.load_state_dict(checkpoint["prior_optimizer_state_dict"])
        self.decoder_optimizer.load_state_dict(
            checkpoint["decoder_optimizer_state_dict"]
        )
        self.step = checkpoint["step"]
        return checkpoint["epoch"]


def visualize_samples(
    samples: torch.Tensor, save_path: str, nrow: int = 4, title: Optional[str] = None
):
    """
    Visualize generated samples in a grid.

    Args:
        samples: Tensor of shape (N, 3, 64, 64) with values in [-1, 1]
        save_path: Path to save the image
        nrow: Number of images per row
        title: Optional title for the plot
    """
    # Convert to numpy and denormalize
    samples = samples.cpu().detach()
    samples = (samples + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    samples = samples.clamp(0, 1)

    n_samples = samples.shape[0]
    ncol = (n_samples + nrow - 1) // nrow

    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
    if ncol == 1:
        axes = axes[None, :]

    axes = axes.flatten()

    for i in range(n_samples):
        img = samples[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis("off")

    # Hide extra subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis("off")

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved samples to {save_path}")


def plot_training_curves(losses: Dict[str, list], save_path: str):
    """
    Plot training loss curves.

    Args:
        losses: Dictionary mapping loss names to lists of values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    for name, values in losses.items():
        plt.plot(values, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")
