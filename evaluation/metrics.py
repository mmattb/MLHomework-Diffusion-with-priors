"""
Evaluation metrics for hierarchical diffusion models.

★★★ TODO: YOU NEED TO IMPLEMENT THREE METRIC FUNCTIONS ★★★

These metrics measure mode collapse and diversity in the generated samples.
"""

import torch
import numpy as np
from typing import Dict, Optional
from sklearn.cluster import KMeans


def compute_mode_coverage(
    samples: torch.Tensor, z2_condition: str, dataset, method: str = "nearest_neighbor"
) -> Dict[str, float]:
    """
    Compute mode coverage metric.

    ★★★ TODO: IMPLEMENT THIS FUNCTION ★★★

    Measures how many distinct z1 categories are recovered when sampling
    conditioned on z2.

    Args:
        samples: Generated images
            Shape: (num_samples, 1, 64, 64)
            Values: Normalized to [-1, 1]

        z2_condition: The z2 category used for conditioning
            Value: String, either "animal" or "vehicle"

        dataset: The SyntheticHierarchicalDataset object
            Provides ground truth information and reference images

        method: Method for classification
            'nearest_neighbor': Match to closest reference image
            'feature_matching': Use simple feature matching

    Returns:
        Dictionary containing:
            - 'num_modes_found': Number of distinct z1 modes identified
            - 'total_possible_modes': Total number of z1 modes for this z2
            - 'coverage_ratio': Fraction of modes covered (in [0, 1])
            - 'mode_counts': Dictionary mapping z1 names to counts

    Implementation approach:
        1. Get the possible z1 subtypes for the given z2
           - Use: dataset.z1_subtypes[z2_condition]
           - Example: For "animal", this gives ["dog", "cat"]

        2. For each generated sample, determine which z1 it matches best
           Option A: Nearest neighbor to reference images
               - Get reference images: dataset.get_reference_images(z1_name)
               - Compute distance (e.g., MSE) to each reference set
               - Assign to closest z1

           Option B: Simple feature matching
               - Extract features (e.g., shape descriptors, color histograms)
               - Match to ground truth feature distributions

        3. Count how many unique z1 categories were found
           - Use a set or counter to track unique assignments

        4. Compute coverage ratio
           - coverage_ratio = num_modes_found / total_possible_modes

        5. Return the statistics

    Example return value:
        {
            'num_modes_found': 2,
            'total_possible_modes': 2,
            'coverage_ratio': 1.0,
            'mode_counts': {'dog': 8, 'cat': 8}
        }

    Tips:
        - Visualize some samples to debug classification
        - MSE in pixel space works reasonably for this synthetic dataset
        - You can average over multiple reference images per z1
        - Consider normalizing distances before comparing
    """
    # ============================================================
    # YOUR CODE HERE
    # ============================================================

    raise NotImplementedError(
        "You need to implement compute_mode_coverage(). "
        "See the docstring for detailed instructions."
    )

    # ============================================================
    # END YOUR CODE
    # ============================================================


def compute_conditional_entropy(
    z1_samples: torch.Tensor,
    z2_condition: str,
    dataset,
    num_clusters: Optional[int] = None,
) -> float:
    """
    Compute conditional entropy H(z1 | z2).

    ★★★ TODO: IMPLEMENT THIS FUNCTION ★★★

    Measures the diversity/uncertainty in the latent space conditioned on z2.
    Higher entropy = more diversity.

    Args:
        z1_samples: Sampled latent embeddings from the prior
            Shape: (num_samples, 2)
            Values: L2-normalized embeddings

        z2_condition: The z2 category used for conditioning
            Value: String, either "animal" or "vehicle"

        dataset: The SyntheticHierarchicalDataset object
            Provides ground truth embeddings for reference

        num_clusters: Number of clusters (default: number of z1 categories)

    Returns:
        Conditional entropy in nats (or bits if you prefer)

    Implementation approach:
        1. Determine the number of z1 categories for this z2
           - If num_clusters is None, use len(dataset.z1_subtypes[z2_condition])

        2. Cluster the z1_samples into discrete categories
           Option A (Recommended): Nearest neighbor to ground truth embeddings
               - Get ground truth z1 embeddings from dataset.z1_embeddings
               - Assign each sample to nearest ground truth embedding
               - This is more robust than K-means for this synthetic dataset

           Option B: K-means clustering
               - Use sklearn.cluster.KMeans
               - Fit to z1_samples
               - Get cluster assignments

        3. Estimate the probability distribution p(z1 | z2)
           - Count samples in each cluster
           - Normalize to get probabilities: p_i = count_i / total_count

        4. Compute entropy
           - H = -∑ p_i * log(p_i)
           - Handle p_i = 0 by skipping those terms
           - Use natural log (nats) or log2 (bits)

        5. Return the entropy value

    Example:
        - If all samples are in one cluster: H ≈ 0 (no diversity)
        - If samples are uniformly distributed: H = log(num_clusters)
        - For 2 modes uniform: H = log(2) ≈ 0.693 nats ≈ 1.0 bits

    Tips:
        - Add small epsilon (1e-10) to avoid log(0)
        - Visualize cluster assignments to verify correctness
        - Compare to theoretical maximum: log(num_clusters)
    """
    # ============================================================
    # YOUR CODE HERE
    # ============================================================

    raise NotImplementedError(
        "You need to implement compute_conditional_entropy(). "
        "See the docstring for detailed instructions."
    )

    # ============================================================
    # END YOUR CODE
    # ============================================================


def compute_kl_divergence(
    z1_samples: torch.Tensor, z2_condition: str, dataset
) -> float:
    """
    Compute KL divergence between model and data distributions.

    ★★★ TODO: IMPLEMENT THIS FUNCTION ★★★

    Measures KL(p_data(z1 | z2) || p_model(z1 | z2)).
    Lower KL = better match to ground truth distribution.

    Args:
        z1_samples: Sampled latent embeddings from the model
            Shape: (num_samples, 2)
            Values: L2-normalized embeddings

        z2_condition: The z2 category used for conditioning
            Value: String, either "animal" or "vehicle"

        dataset: The SyntheticHierarchicalDataset object
            Provides ground truth distribution p_data(z1 | z2)

    Returns:
        KL divergence in nats (or bits if you prefer)

    Implementation approach:
        1. Get the ground truth distribution p_data(z1 | z2)
           - Use: dataset.get_ground_truth_distribution(z2_condition)
           - Example: For "animal", this gives {"dog": 0.5, "cat": 0.5}

        2. Estimate the model distribution p_model(z1 | z2) from samples
           - Classify each sample to a z1 category (similar to mode coverage)
           - Count samples per category
           - Normalize to get probabilities

        3. Compute KL divergence
           - KL = ∑ p_data(z1|z2) * log(p_data(z1|z2) / p_model(z1|z2))
           - Iterate over all z1 categories
           - Handle cases where p_model is 0 (model didn't generate that mode)

        4. Return the KL divergence

    Example:
        Ground truth: {"dog": 0.5, "cat": 0.5}
        Model generates: {"dog": 0.5, "cat": 0.5} -> KL ≈ 0 (perfect match)
        Model generates: {"dog": 1.0, "cat": 0.0} -> KL = ∞ (mode collapse)

    Tips:
        - Add small epsilon to p_model to avoid division by zero
        - If p_model(z1) = 0 but p_data(z1) > 0, KL = ∞ (or very large)
        - You can clip KL to a maximum value for numerical stability
        - Lower KL is better

    Numerical stability:
        When p_model is 0 for a mode that p_data assigns probability to,
        the KL divergence is technically infinite. In practice, you can:
        - Add a small epsilon (e.g., 1e-10) to all probabilities
        - Or clip the KL contribution to a large but finite value
    """
    # ============================================================
    # YOUR CODE HERE
    # ============================================================

    raise NotImplementedError(
        "You need to implement compute_kl_divergence(). "
        "See the docstring for detailed instructions."
    )

    # ============================================================
    # END YOUR CODE
    # ============================================================


def evaluate_model(
    model, diffusion, dataset, num_samples: int = 100, device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive evaluation of a model.

    This function is complete - it calls your implemented metrics.

    Args:
        model: The generative model (either Model A or tuple of (prior, decoder))
        diffusion: Diffusion process (or tuple for hierarchical model)
        dataset: The dataset object
        num_samples: Number of samples to generate per category
        device: Device to run on

    Returns:
        Dictionary with metrics for each z2 category
    """
    model_is_hierarchical = isinstance(model, tuple)

    results = {}

    for z2_name in dataset.z2_categories:
        print(f"\nEvaluating on z2={z2_name}...")

        # Get z2 conditioning
        z2_idx = dataset.z2_categories.index(z2_name)
        z2_onehot = torch.zeros(num_samples, 2, device=device)
        z2_onehot[:, z2_idx] = 1.0

        if model_is_hierarchical:
            # Model B: Sample from prior then decoder
            prior, decoder = model
            prior_diffusion, decoder_diffusion = diffusion

            # Sample z1 from prior
            with torch.no_grad():
                z1_samples = prior_diffusion.p_sample(
                    prior, shape=(num_samples, 2), condition=z2_onehot
                )
                # Normalize z1 samples
                z1_samples = z1_samples / torch.norm(z1_samples, dim=1, keepdim=True)

                # Sample images from decoder
                samples = decoder_diffusion.p_sample(
                    decoder, shape=(num_samples, 1, 64, 64), condition=z1_samples
                )
        else:
            # Model A: Sample directly from model
            with torch.no_grad():
                samples = diffusion.p_sample(
                    model, shape=(num_samples, 1, 64, 64), condition=z2_onehot
                )
            z1_samples = None  # Not applicable for Model A

        # Compute metrics
        metrics = {}

        # Mode coverage
        try:
            coverage = compute_mode_coverage(samples, z2_name, dataset)
            metrics.update(coverage)
        except NotImplementedError:
            print("  ⚠ Mode coverage not implemented yet")

        # Conditional entropy (only for hierarchical model)
        if model_is_hierarchical and z1_samples is not None:
            try:
                entropy = compute_conditional_entropy(z1_samples, z2_name, dataset)
                metrics["conditional_entropy"] = entropy
            except NotImplementedError:
                print("  ⚠ Conditional entropy not implemented yet")

        # KL divergence (only for hierarchical model)
        if model_is_hierarchical and z1_samples is not None:
            try:
                kl_div = compute_kl_divergence(z1_samples, z2_name, dataset)
                metrics["kl_divergence"] = kl_div
            except NotImplementedError:
                print("  ⚠ KL divergence not implemented yet")

        results[z2_name] = metrics

    return results


if __name__ == "__main__":
    """Test metric functions (will fail until you implement them)."""

    print("Testing evaluation metrics...")
    print("=" * 60)
    print()
    print("These tests will fail until you implement the metric functions.")
    print("Follow the instructions in ASSIGNMENT.md to implement them.")
    print()

    # Create dummy data
    num_samples = 16
    samples = torch.randn(num_samples, 1, 64, 64)
    z1_samples = torch.randn(num_samples, 2)
    z1_samples = z1_samples / torch.norm(z1_samples, dim=1, keepdim=True)

    # This would need a real dataset
    print("To test metrics, you need to:")
    print("1. Load a dataset")
    print("2. Generate samples from a trained model")
    print("3. Call the metric functions with real data")
    print()
    print("See evaluate.py for a complete evaluation script.")
