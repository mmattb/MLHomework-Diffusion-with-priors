"""
Evaluation script for trained models.

Usage:
    # Evaluate Model A
    python evaluate.py --model flat --checkpoint outputs/model_a_final.pt

    # Evaluate Model B
    python evaluate.py --model hierarchical --checkpoint outputs/model_b_final.pt
"""

import argparse
import torch
from pathlib import Path
import json

from data import get_dataloader
from models import ImageDenoiser, LatentPrior, CategoricalPrior
from diffusion import DiffusionProcess
from evaluation import evaluate_model
from training import visualize_samples


def evaluate_model_a(args):
    """Evaluate Model A."""
    print("=" * 60)
    print("Evaluating Model A: Flat Conditional Diffusion")
    print("=" * 60)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataset
    print("Loading dataset...")
    _, dataset = get_dataloader(
        num_samples=1000,  # Small for evaluation
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    print()

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

    # Evaluate
    print("Running evaluation...")
    results = evaluate_model(
        model=model,
        diffusion=diffusion,
        dataset=dataset,
        num_samples=args.num_eval_samples,
        device=device,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for z2_name, metrics in results.items():
        print(f"\nCategory: {z2_name}")
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                print(f"  {metric_name}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {metric_name}: {value:.4f}")

    # Save results
    output_path = Path(args.output_dir) / "eval_results_a.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Generate sample visualization
    print("\nGenerating sample visualizations...")
    for z2_idx, z2_name in enumerate(dataset.z2_categories):
        z2_onehot = torch.zeros(16, 2, device=device)
        z2_onehot[:, z2_idx] = 1.0

        with torch.no_grad():
            samples = diffusion.p_sample(
                model, shape=(16, 1, 64, 64), condition=z2_onehot
            )

        visualize_samples(
            samples,
            str(Path(args.output_dir) / f"eval_samples_a_{z2_name}.png"),
            title=f"Model A - {z2_name.capitalize()}",
        )


def evaluate_model_b(args):
    """Evaluate Model B."""
    print("=" * 60)
    print("Evaluating Model B: Hierarchical Diffusion")
    print("=" * 60)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataset
    print("Loading dataset...")
    _, dataset = get_dataloader(
        num_samples=1000, batch_size=32, shuffle=False, num_workers=0
    )
    print()

    # Load checkpoint to detect prior type
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    prior_type = checkpoint.get("prior_type", "diffusion")
    print(f"Detected prior type: {prior_type}")
    print()

    # Create models based on prior type
    print("Creating models...")
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

    # Evaluate
    print("Running evaluation...")
    results = evaluate_model(
        model=(prior, decoder),
        diffusion=(prior_diffusion, decoder_diffusion),
        dataset=dataset,
        num_samples=args.num_eval_samples,
        device=device,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for z2_name, metrics in results.items():
        print(f"\nCategory: {z2_name}")
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                print(f"  {metric_name}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {metric_name}: {value:.4f}")

    # Save results
    output_path = Path(args.output_dir) / "eval_results_b.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Generate sample visualization
    print("\nGenerating sample visualizations...")
    for z2_idx, z2_name in enumerate(dataset.z2_categories):
        z2_onehot = torch.zeros(16, 2, device=device)
        z2_onehot[:, z2_idx] = 1.0

        with torch.no_grad():
            # Sample z1 from prior
            if prior_type == "categorical":
                z1_samples = prior(z2_onehot)
            else:
                z1_samples = prior_diffusion.p_sample(
                    prior, shape=(16, 2), condition=z2_onehot
                )
                z1_samples = z1_samples / torch.norm(z1_samples, dim=1, keepdim=True)

            # Sample images from decoder
            samples = decoder_diffusion.p_sample(
                decoder, shape=(16, 1, 64, 64), condition=z1_samples
            )

        visualize_samples(
            samples,
            str(Path(args.output_dir) / f"eval_samples_b_{z2_name}.png"),
            title=f"Model B - {z2_name.capitalize()}",
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["flat", "hierarchical"],
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    # Evaluation parameters
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=100,
        help="Number of samples to generate for evaluation",
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
        default="outputs/evaluation",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Evaluate appropriate model
    if args.model == "flat":
        evaluate_model_a(args)
    else:
        evaluate_model_b(args)


if __name__ == "__main__":
    main()
