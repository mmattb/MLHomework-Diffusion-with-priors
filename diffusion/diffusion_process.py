"""
Core diffusion process utilities.

Implements the forward and reverse diffusion processes for DDPM.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def get_beta_schedule(schedule_name: str, num_timesteps: int) -> torch.Tensor:
    """
    Get beta schedule for diffusion process.

    Args:
        schedule_name: 'linear' or 'cosine'
        num_timesteps: Number of diffusion timesteps

    Returns:
        Beta schedule tensor of shape (num_timesteps,)
    """
    if schedule_name == "linear":
        # Linear schedule from DDPM paper
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, num_timesteps)

    elif schedule_name == "cosine":
        # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")


class DiffusionProcess:
    """
    Handles forward and reverse diffusion processes.

    This class manages the noise schedule and provides methods for:
    - Adding noise to data (forward process)
    - Predicting clean data from noisy data (reverse process)
    - Sampling from the model
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        device: str = "cuda",
    ):
        """
        Args:
            num_timesteps: Number of diffusion steps
            beta_schedule: 'linear' or 'cosine'
            device: 'cuda' or 'cpu'
        """
        self.num_timesteps = num_timesteps
        self.device = device

        # Get beta schedule
        betas = get_beta_schedule(beta_schedule, num_timesteps)

        # Pre-compute useful quantities
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # Store as buffers (will be moved to device)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        ).to(device)
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).to(device)

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0).

        Sample from q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)

        Args:
            x_0: Clean data, shape (batch_size, ...)
            t: Timestep indices, shape (batch_size,)
            noise: Optional pre-sampled noise

        Returns:
            Tuple of (noisy data x_t, noise used)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Extract coefficients for the given timesteps
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # Apply noise: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

        return x_t, noise

    def p_sample_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: p(x_{t-1} | x_t).

        Args:
            model: Denoising model that predicts noise
            x_t: Noisy data at timestep t, shape (batch_size, ...)
            t: Current timestep, shape (batch_size,)
            condition: Conditioning information
            clip_denoised: Whether to clip predictions to [-1, 1]

        Returns:
            x_{t-1}: Less noisy data
        """
        batch_size = x_t.shape[0]

        # Predict noise
        with torch.no_grad():
            predicted_noise = model(x_t, t, condition)

        # Predict x_0 from x_t and predicted noise
        x_0_pred = self._predict_x0_from_noise(x_t, t, predicted_noise)

        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # Compute mean of q(x_{t-1} | x_t, x_0)
        model_mean = self._q_posterior_mean(x_0_pred, x_t, t)

        if t[0] == 0:
            # No noise at final step
            return model_mean
        else:
            # Add noise
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def p_sample(
        self,
        model: nn.Module,
        shape: Tuple,
        condition: torch.Tensor,
        clip_denoised: bool = True,
        return_all_timesteps: bool = False,
    ) -> torch.Tensor:
        """
        Full reverse diffusion sampling: p(x_0).

        Start from random noise and iteratively denoise.

        Args:
            model: Denoising model
            shape: Shape of samples to generate
            condition: Conditioning information for all samples
            clip_denoised: Whether to clip intermediate predictions
            return_all_timesteps: If True, return all intermediate steps

        Returns:
            Generated samples x_0, or list of all timesteps if return_all_timesteps=True
        """
        device = self.device
        batch_size = shape[0]

        # Start from random noise
        x_t = torch.randn(shape, device=device)

        all_timesteps = [x_t] if return_all_timesteps else None

        # Reverse diffusion
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x_t = self.p_sample_step(model, x_t, t, condition, clip_denoised)

            if return_all_timesteps:
                all_timesteps.append(x_t)

        if return_all_timesteps:
            return all_timesteps
        return x_t

    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple,
        condition: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        DDIM sampling for faster generation.

        Args:
            model: Denoising model
            shape: Shape of samples to generate
            condition: Conditioning information
            num_steps: Number of sampling steps (< num_timesteps for speedup)
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
            clip_denoised: Whether to clip predictions

        Returns:
            Generated samples x_0
        """
        device = self.device
        batch_size = shape[0]

        # Create subsequence of timesteps
        step_size = self.num_timesteps // num_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device)
        timesteps = torch.flip(timesteps, dims=[0])

        # Start from noise
        x_t = torch.randn(shape, device=device)

        for i, t_curr in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t_curr, device=device, dtype=torch.long)

            # Predict noise
            with torch.no_grad():
                predicted_noise = model(x_t, t_batch, condition)

            # Predict x_0
            x_0_pred = self._predict_x0_from_noise(x_t, t_batch, predicted_noise)

            if clip_denoised:
                x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                t_next_batch = torch.full(
                    (batch_size,), t_next, device=device, dtype=torch.long
                )

                # DDIM update
                alpha_t = self._extract(self.alphas_cumprod, t_batch, x_t.shape)
                alpha_t_next = self._extract(
                    self.alphas_cumprod, t_next_batch, x_t.shape
                )

                sigma_t = eta * torch.sqrt(
                    (1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next)
                )

                mean = (
                    torch.sqrt(alpha_t_next) * x_0_pred
                    + torch.sqrt(1 - alpha_t_next - sigma_t**2) * predicted_noise
                )

                noise = torch.randn_like(x_t) if eta > 0 else 0
                x_t = mean + sigma_t * noise
            else:
                x_t = x_0_pred

        return x_t

    def _predict_x0_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        # x_0 = (x_t - sqrt(1 - alpha_cumprod_t) * noise) / sqrt(alpha_cumprod_t)
        return (x_t - sqrt_one_minus_alpha_cumprod_t * noise) / sqrt_alpha_cumprod_t

    def _q_posterior_mean(
        self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean of q(x_{t-1} | x_t, x_0)."""
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        return coef1 * x_0 + coef2 * x_t

    def _extract(
        self, arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: Tuple
    ) -> torch.Tensor:
        """
        Extract values from arr at indices timesteps and reshape for broadcasting.

        Args:
            arr: Array to extract from
            timesteps: Indices to extract
            broadcast_shape: Shape to broadcast to

        Returns:
            Extracted values reshaped for broadcasting
        """
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res
