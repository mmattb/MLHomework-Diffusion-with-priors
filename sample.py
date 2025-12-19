"""
Sample generation script.

Usage:
    # Generate samples from Model A
    python sample.py --model flat --checkpoint outputs/model_a_final.pt --z2 animal --num_samples 16

    # Generate samples from Model B
    python sample.py --model hierarchical --checkpoint outputs/model_b_final.pt --z2 vehicle --num_samples 16
"""

import argparse
import torch
from pathlib import Path

from models import ImageDenoiser, LatentPrior, CategoricalPrior
from diffusion import DiffusionProcess
from training import visualize_samples


def sample_model_a(args):
    """Generate samples from Model A."""
    print("=" * 60)
    print(f"Generating samples from Model A for z2={args.z2}")
    print("=" * 60)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    print("Creating model...")
    model = ImageDenoiser(
        image_channels=1,
        condition_dim=2,
        time_embed_dim=32,
        model_channels=args.model_channels,
    )
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print()

    # Create diffusion process
    diffusion = DiffusionProcess(
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        device=device,
    )

    # Create conditioning
    z2_categories = ["animal", "vehicle"]
    z2_idx = z2_categories.index(args.z2)
    z2_onehot = torch.zeros(args.num_samples, 2, device=device)
    z2_onehot[:, z2_idx] = 1.0

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    with torch.no_grad():
        if args.use_ddim:
            samples = diffusion.ddim_sample(
                model,
                shape=(args.num_samples, 1, 64, 64),
                condition=z2_onehot,
                num_steps=args.ddim_steps,
                eta=args.ddim_eta,
            )
        else:
            samples = diffusion.p_sample(
                model, shape=(args.num_samples, 1, 64, 64), condition=z2_onehot
            )

    # Visualize
    output_path = Path(args.output_dir) / f"samples_a_{args.z2}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_samples(
        samples, str(output_path), nrow=4, title=f"Model A - {args.z2.capitalize()}"
    )

    print("Done!")


def sample_model_b(args):
    """Generate samples from Model B."""
    print("=" * 60)
    print(f"Generating samples from Model B for z2={args.z2}")
    print("=" * 60)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load checkpoint to detect prior type
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    prior_type = checkpoint.get("prior_type", "diffusion")
    print(f"Detected prior type: {prior_type}")
    print()

    # Create models based on prior type
    print("Creating models...")
    from data import get_dataloader
    _, dataset = get_dataloader(num_samples=100, batch_size=32, shuffle=False, num_workers=0)
    
    if prior_type == "categorical":
        prior = CategoricalPrior(
            z1_embeddings=dataset.z1_embeddings,
            z1_subtypes=dataset.z1_subtypes,
            condition_dim=2,
            hidden_dim=256,
        )
    else:
        prior = LatentPrior(
            latent_dim=2, condition_dim=2, time_embed_dim=32, hidden_dims=(256, 256, 256)
        )
    
    decoder = ImageDenoiser(
        image_channels=1,
        condition_dim=2,
        time_embed_dim=32,
        model_channels=args.model_channels,
    )
    prior = prior.to(device)
    decoder = decoder.to(device)

    # Load state dicts
    prior.load_state_dict(checkpoint["prior_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    prior.eval()
    decoder.eval()
    print()

    # Create diffusion processes
    prior_diffusion = DiffusionProcess(
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        device=device,
    )
    decoder_diffusion = DiffusionProcess(
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        device=device,
    )

    # Create conditioning
    z2_categories = ["animal", "vehicle"]
    z2_idx = z2_categories.index(args.z2)
    z2_onehot = torch.zeros(args.num_samples, 2, device=device)
    z2_onehot[:, z2_idx] = 1.0

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    with torch.no_grad():
        # Sample z1 from prior
        print("  Sampling z1 from prior...")
        if prior_type == "categorical":
            z1_samples = prior(z2_onehot)
        else:
            if args.use_ddim:
                z1_samples = prior_diffusion.ddim_sample(
                    prior,
                    shape=(args.num_samples, 2),
                    condition=z2_onehot,
                    num_steps=args.ddim_steps,
                    eta=args.ddim_eta,
                )
            else:
                z1_samples = prior_diffusion.p_sample(
                    prior, shape=(args.num_samples, 2), condition=z2_onehot
                )

            # Normalize z1 samples
            z1_samples = z1_samples / torch.norm(z1_samples, dim=1, keepdim=True)

        # Sample images from decoder
        print("  Sampling images from decoder...")
        if args.use_ddim:
            samples = decoder_diffusion.ddim_sample(
                decoder,
                shape=(args.num_samples, 1, 64, 64),
                condition=z1_samples,
                num_steps=args.ddim_steps,
                eta=args.ddim_eta,
            )
        else:
            samples = decoder_diffusion.p_sample(
                decoder, shape=(args.num_samples, 1, 64, 64), condition=z1_samples
            )

    # Visualize
    output_path = Path(args.output_dir) / f"samples_b_{args.z2}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_samples(
        samples, str(output_path), nrow=4, title=f"Model B - {args.z2.capitalize()}"
    )

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained models")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["flat", "hierarchical"],
        help="Model type",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    # Sampling parameters
    parser.add_argument(
        "--z2",
        type=str,
        required=True,
        choices=["animal", "vehicle"],
        help="Category to condition on",
    )
    parser.add_argument(
        "--num_samples", type=int, default=16, help="Number of samples to generate"
    )
    parser.add_argument(
        "--use_ddim", action="store_true", help="Use DDIM sampling (faster)"
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=50, help="Number of DDIM steps"
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="DDIM eta parameter (0=deterministic)",
    )

    # Model parameters (must match training)
    parser.add_argument(
        "--model_channels", type=int, default=64, help="Base number of channels in UNet"
    )
    parser.add_argument(
        "--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps"
    )
    parser.add_argument(
        "--beta_schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Noise schedule",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/samples",
        help="Output directory for samples",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Generate samples
    if args.model == "flat":
        sample_model_a(args)
    else:
        sample_model_b(args)


if __name__ == "__main__":
    main()
