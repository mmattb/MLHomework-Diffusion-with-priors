"""
Main training script for hierarchical diffusion models.

Usage:
    # Train Model A (flat baseline)
    python train.py --model flat --epochs 50 --batch_size 32

    # Train Model B (hierarchical)
    python train.py --model hierarchical --epochs 50 --batch_size 32
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path

from data import get_dataloader
from models import ImageDenoiser, LatentPrior
from diffusion import DiffusionProcess
from training import (
    TrainerModelA,
    TrainerModelB,
    visualize_samples,
    plot_training_curves,
)


def train_model_a(args):
    """Train Model A: Flat conditional diffusion."""
    print("=" * 60)
    print("Training Model A: Flat Conditional Diffusion")
    print("=" * 60)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataset
    print("Creating dataset...")
    dataloader, dataset = get_dataloader(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    print(f"Dataset size: {len(dataset)}")
    print()

    # Create model
    print("Creating model...")
    model = ImageDenoiser(
        image_channels=1,
        condition_dim=2,  # z2 one-hot
        time_embed_dim=32,
        model_channels=args.model_channels,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    print()

    # Create diffusion process
    diffusion = DiffusionProcess(
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        device=device,
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Create trainer
    trainer = TrainerModelA(model, diffusion, optimizer, device)

    # Training loop
    losses = []

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        metrics = trainer.train_epoch(dataloader)
        losses.append(metrics["loss"])

        print(f"  Loss: {metrics['loss']:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = Path(args.output_dir) / f"model_a_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(str(checkpoint_path), epoch + 1)

        # Generate samples for visualization
        if (epoch + 1) % args.visualize_every == 0:
            print("  Generating samples...")
            model.eval()
            with torch.no_grad():
                # Generate samples for "animal" category
                z2_onehot = torch.zeros(16, 2, device=device)
                z2_onehot[:, 0] = 1.0  # animal
                samples = diffusion.p_sample(
                    model, shape=(16, 1, 64, 64), condition=z2_onehot
                )
                visualize_samples(
                    samples,
                    str(
                        Path(args.output_dir)
                        / f"samples_a_epoch_{epoch + 1}_animal.png"
                    ),
                    title=f"Model A - Epoch {epoch + 1} - Animal",
                )

                # Generate samples for "vehicle" category
                z2_onehot = torch.zeros(16, 2, device=device)
                z2_onehot[:, 1] = 1.0  # vehicle
                samples = diffusion.p_sample(
                    model, shape=(16, 1, 64, 64), condition=z2_onehot
                )
                visualize_samples(
                    samples,
                    str(
                        Path(args.output_dir)
                        / f"samples_a_epoch_{epoch + 1}_vehicle.png"
                    ),
                    title=f"Model A - Epoch {epoch + 1} - Vehicle",
                )
        print()

    # Save final model
    final_path = Path(args.output_dir) / "model_a_final.pt"
    trainer.save_checkpoint(str(final_path), args.epochs)

    # Plot training curves
    plot_training_curves(
        {"loss": losses}, str(Path(args.output_dir) / "training_curves_a.png")
    )

    print("Training complete!")


def train_model_b(args):
    """Train Model B: Hierarchical diffusion with prior."""
    print("=" * 60)
    print("Training Model B: Hierarchical Diffusion")
    print("=" * 60)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataset
    print("Creating dataset...")
    dataloader, dataset = get_dataloader(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    print(f"Dataset size: {len(dataset)}")
    print()

    # Create models
    print("Creating models...")

    # Create prior based on type
    if args.prior_type == "categorical":
        from models import CategoricalPrior

        prior = CategoricalPrior(
            z1_embeddings=dataset.z1_embeddings,
            z1_subtypes=dataset.z1_subtypes,
            condition_dim=2,
        )
        prior_diffusion = None  # Not used for categorical prior
        print(f"Using CategoricalPrior (discrete z1 sampling)")
    else:  # diffusion
        from models import LatentPrior

        prior = LatentPrior(
            latent_dim=args.latent_dim,
            condition_dim=2,  # z2 one-hot
            time_embed_dim=32,
            hidden_dims=(256, 256, 256),
        )
        prior_diffusion = DiffusionProcess(
            num_timesteps=args.num_timesteps,
            beta_schedule=args.beta_schedule,
            device=device,
        )
        print(f"Using LatentPrior (diffusion-based z1 sampling)")

    decoder = ImageDenoiser(
        image_channels=1,
        condition_dim=args.latent_dim,  # z1 embedding
        time_embed_dim=32,
        model_channels=args.model_channels,
    )

    num_params_prior = sum(p.numel() for p in prior.parameters())
    num_params_decoder = sum(p.numel() for p in decoder.parameters())
    print(f"Prior parameters: {num_params_prior:,}")
    print(f"Decoder parameters: {num_params_decoder:,}")
    print(f"Total parameters: {num_params_prior + num_params_decoder:,}")
    print()

    # Create diffusion process for decoder
    decoder_diffusion = DiffusionProcess(
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        device=device,
    )

    # Create optimizers
    # Use a higher learning rate for the categorical prior since it's a very small model
    prior_lr = (
        args.learning_rate * 10
        if args.prior_type == "categorical"
        else args.learning_rate
    )

    prior_optimizer = optim.AdamW(
        prior.parameters(), lr=prior_lr, weight_decay=args.weight_decay
    )
    decoder_optimizer = optim.AdamW(
        decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Create trainer
    trainer = TrainerModelB(
        prior,
        decoder,
        prior_diffusion,
        decoder_diffusion,
        prior_optimizer,
        decoder_optimizer,
        device,
    )

    # Training loop
    prior_losses = []
    decoder_losses = []

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        metrics = trainer.train_epoch(dataloader)
        prior_losses.append(metrics["prior_loss"])
        decoder_losses.append(metrics["decoder_loss"])

        print(f"  Prior Loss: {metrics['prior_loss']:.4f}")
        print(f"  Decoder Loss: {metrics['decoder_loss']:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = Path(args.output_dir) / f"model_b_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(str(checkpoint_path), epoch + 1)

        # Generate samples for visualization
        if (epoch + 1) % args.visualize_every == 0:
            print("  Generating samples...")
            prior.eval()
            decoder.eval()
            with torch.no_grad():
                # Generate samples for "animal" category
                z2_onehot = torch.zeros(16, 2, device=device)
                z2_onehot[:, 0] = 1.0  # animal
                # Sample z1 from prior
                if args.prior_type == "categorical":
                    z1_samples = prior(z2_onehot)
                else:
                    z1_samples = prior_diffusion.p_sample(
                        prior, shape=(16, args.latent_dim), condition=z2_onehot
                    )
                    z1_samples = z1_samples / torch.norm(
                        z1_samples, dim=1, keepdim=True
                    )
                # Sample images from decoder
                samples = decoder_diffusion.p_sample(
                    decoder, shape=(16, 1, 64, 64), condition=z1_samples
                )
                visualize_samples(
                    samples,
                    str(
                        Path(args.output_dir)
                        / f"samples_b_epoch_{epoch + 1}_animal.png"
                    ),
                    title=f"Model B - Epoch {epoch + 1} - Animal",
                )

                # Generate samples for "vehicle" category
                z2_onehot = torch.zeros(16, 2, device=device)
                z2_onehot[:, 1] = 1.0  # vehicle
                if args.prior_type == "categorical":
                    z1_samples = prior(z2_onehot)
                else:
                    z1_samples = prior_diffusion.p_sample(
                        prior, shape=(16, args.latent_dim), condition=z2_onehot
                    )
                    z1_samples = z1_samples / torch.norm(
                        z1_samples, dim=1, keepdim=True
                    )
                samples = decoder_diffusion.p_sample(
                    decoder, shape=(16, 1, 64, 64), condition=z1_samples
                )
                visualize_samples(
                    samples,
                    str(
                        Path(args.output_dir)
                        / f"samples_b_epoch_{epoch + 1}_vehicle.png"
                    ),
                    title=f"Model B - Epoch {epoch + 1} - Vehicle",
                )
        print()

    # Save final model
    final_path = Path(args.output_dir) / "model_b_final.pt"
    trainer.save_checkpoint(str(final_path), args.epochs, args.prior_type)

    # Plot training curves
    plot_training_curves(
        {"prior_loss": prior_losses, "decoder_loss": decoder_losses},
        str(Path(args.output_dir) / "training_curves_b.png"),
    )

    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical diffusion models")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["flat", "hierarchical"],
        help="Model type: flat (Model A) or hierarchical (Model B)",
    )
    parser.add_argument(
        "--prior_type",
        type=str,
        default="categorical",
        choices=["diffusion", "categorical"],
        help="Type of prior for Model B: 'diffusion' or 'categorical' (default: categorical)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")

    # Data parameters
    parser.add_argument(
        "--num_samples", type=int, default=10000, help="Number of samples in dataset"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )

    # Model parameters
    parser.add_argument(
        "--model_channels", type=int, default=64, help="Base number of channels in UNet"
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=2,
        help="Dimension of z1 latent embeddings (use 2 for visualization)",
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
        default="outputs",
        help="Output directory for checkpoints and samples",
    )
    parser.add_argument(
        "--save_every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--visualize_every", type=int, default=5, help="Generate samples every N epochs"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Train appropriate model
    if args.model == "flat":
        train_model_a(args)
    else:
        train_model_b(args)


if __name__ == "__main__":
    main()
