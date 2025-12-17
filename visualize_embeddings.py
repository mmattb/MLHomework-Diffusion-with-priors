"""
Visualize z1 embeddings in 2D space.

This script shows the latent embeddings learned by Model B's prior.
Since z1 has 2 dimensions and is L2-normalized, all embeddings lie on the unit circle.

You should see 4 distinct clusters representing: dog, cat, car, truck

Usage:
    python visualize_embeddings.py --checkpoint outputs/hierarchical/final.pt --z2 animal --num_samples 500
"""

import argparse
import torch
import numpy as np

# Set matplotlib to non-interactive backend before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from models import LatentPrior
from diffusion import DiffusionProcess
from data import SyntheticHierarchicalDataset


def visualize_z1_embeddings(args):
    """Visualize sampled z1 embeddings in 2D space."""
    print("=" * 60)
    print(f"Visualizing z1 embeddings for z2={args.z2}")
    print("=" * 60)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataset to get ground truth embeddings
    dataset = SyntheticHierarchicalDataset(num_samples=1000)

    # Create prior model
    print("Creating prior model...")
    prior = LatentPrior(
        latent_dim=2,  # Changed to 2 for visualization
        condition_dim=2,
        time_embed_dim=32,
        hidden_dims=(256, 256, 256),  # Match training config
    )
    prior = prior.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    prior.load_state_dict(checkpoint["prior_state_dict"])
    prior.eval()
    print()

    # Create diffusion process
    diffusion = DiffusionProcess(
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        device=device,
    )

    # Sample z1 embeddings for the specified z2
    z2_categories = ["animal", "vehicle"]
    z2_idx = z2_categories.index(args.z2)
    z2_onehot = torch.zeros(args.num_samples, 2, device=device)
    z2_onehot[:, z2_idx] = 1.0

    print(f"Sampling {args.num_samples} z1 embeddings from prior...")
    with torch.no_grad():
        z1_samples = diffusion.p_sample(
            prior, shape=(args.num_samples, 2), condition=z2_onehot
        )
        # Normalize to unit circle (z1 embeddings are L2-normalized)
        z1_samples = z1_samples / torch.norm(z1_samples, dim=1, keepdim=True)

    z1_samples_np = z1_samples.cpu().numpy()

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Ground truth embeddings
    ax1 = axes[0]
    z1_subtypes = dataset.z1_subtypes[args.z2]
    colors = ["red", "blue"]
    markers = ["o", "s"]

    for i, z1_name in enumerate(z1_subtypes):
        emb = dataset.z1_embeddings[z1_name]
        ax1.scatter(
            emb[0],
            emb[1],
            s=300,
            c=colors[i],
            marker=markers[i],
            label=f"{z1_name} (ground truth)",
            alpha=0.7,
            edgecolors="black",
            linewidths=2,
        )

    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, linewidth=1)

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Dimension 1", fontsize=12)
    ax1.set_ylabel("Dimension 2", fontsize=12)
    ax1.set_title(
        f"Ground Truth z1 Embeddings\n(z2={args.z2})", fontsize=14, fontweight="bold"
    )
    ax1.legend(fontsize=10)

    # Right plot: Sampled embeddings
    ax2 = axes[1]
    ax2.scatter(
        z1_samples_np[:, 0],
        z1_samples_np[:, 1],
        s=20,
        alpha=0.5,
        c="purple",
        label="Sampled z1",
    )

    # Overlay ground truth for reference
    for i, z1_name in enumerate(z1_subtypes):
        emb = dataset.z1_embeddings[z1_name]
        ax2.scatter(
            emb[0],
            emb[1],
            s=300,
            c=colors[i],
            marker=markers[i],
            alpha=0.7,
            edgecolors="black",
            linewidths=2,
        )

    # Draw unit circle
    ax2.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, linewidth=1)

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Dimension 1", fontsize=12)
    ax2.set_ylabel("Dimension 2", fontsize=12)
    ax2.set_title(
        f"Model B: Sampled z1 Embeddings\n(z2={args.z2}, n={args.num_samples})",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend(fontsize=10)

    plt.tight_layout()

    # Save figure
    output_path = Path(args.output_dir) / f"z1_embeddings_{args.z2}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")

    # Also create a histogram of cluster assignments
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Assign each sample to nearest ground truth embedding
    assignments = []
    for sample in z1_samples_np:
        distances = []
        for z1_name in z1_subtypes:
            emb = dataset.z1_embeddings[z1_name]
            dist = np.linalg.norm(sample - emb)
            distances.append(dist)
        assignments.append(z1_subtypes[np.argmin(distances)])

    # Count assignments
    from collections import Counter

    counts = Counter(assignments)

    # Plot histogram
    labels = z1_subtypes
    values = [counts[label] for label in labels]
    bars = ax.bar(
        labels, values, color=["red", "blue"], alpha=0.7, edgecolor="black", linewidth=2
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Add expected count line
    expected_count = args.num_samples / len(z1_subtypes)
    ax.axhline(
        expected_count,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Expected (uniform): {expected_count:.0f}",
    )

    ax.set_xlabel("z1 Category", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Mode Coverage Histogram\n(z2={args.z2}, n={args.num_samples})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    hist_path = Path(args.output_dir) / f"z1_histogram_{args.z2}.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    print(f"Saved histogram to {hist_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Total samples: {args.num_samples}")
    print(f"Expected per mode (uniform): {expected_count:.0f}")
    print("\nActual counts:")
    for label in labels:
        count = counts[label]
        percentage = (count / args.num_samples) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    print("\nMode coverage ratio:", len(counts) / len(z1_subtypes))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize z1 embeddings")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--z2",
        type=str,
        required=True,
        choices=["animal", "vehicle"],
        help="Which z2 category to sample",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of z1 samples to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps",
    )
    parser.add_argument(
        "--beta_schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Beta schedule type",
    )

    args = parser.parse_args()
    visualize_z1_embeddings(args)
