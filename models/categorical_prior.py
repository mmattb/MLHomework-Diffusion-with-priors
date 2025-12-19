"""
Categorical Prior for Model B (hierarchical model with discrete z1).

This is an alternative to the diffusion-based LatentPrior that directly
models p(z1 | z2) as a categorical distribution over discrete z1 modes.

This is more appropriate when z1 has a small number of discrete values
(like our 4 categories: dog, cat, car, truck).

Key insight: Hierarchical priors can use ANY generative model!
- Could be diffusion (like DALL-E 2)
- Could be categorical (like this)
- Could be autoregressive, VAE, flow, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalPrior(nn.Module):
    """
    Learns p(z1 | z2) as a categorical distribution.

    Given z2 (one-hot encoded), outputs logits over possible z1 categories.
    Then samples a discrete z1 and looks up its embedding.

    Architecture:
        z2_onehot → MLP → logits → sample z1_idx → lookup z1_embedding

    This is MUCH simpler and more appropriate than diffusion for discrete modes!
    """

    def __init__(
        self,
        z1_embeddings: dict,  # {"dog": tensor, "cat": tensor, ...}
        z1_subtypes: dict,  # {"animal": ["dog", "cat"], "vehicle": ["car", "truck"]}
        condition_dim: int = 2,  # z2 dimension (one-hot)
    ):
        """
        Args:
            z1_embeddings: Dictionary mapping z1 names to their embeddings
            z1_subtypes: Dictionary mapping z2 names to lists of z1 names
            condition_dim: Dimension of z2 (2 for binary one-hot)
        """
        super().__init__()

        self.z1_embeddings = z1_embeddings
        self.z1_subtypes = z1_subtypes
        self.z1_names = list(z1_embeddings.keys())  # ["dog", "cat", "car", "truck"]
        self.num_z1_modes = len(self.z1_names)

        # Create embedding lookup table
        embedding_list = [z1_embeddings[name] for name in self.z1_names]
        self.embedding_table = nn.Parameter(
            torch.stack(
                [torch.tensor(emb, dtype=torch.float32) for emb in embedding_list]
            ),
            requires_grad=False,  # Fixed embeddings
        )

        # Logits table: z2_onehot → logits over z1 categories
        # Since z2 is one-hot, a single Linear layer (no bias) is exactly a lookup table.
        # This is much more appropriate for discrete categories than a deep MLP.
        self.logits_table = nn.Linear(condition_dim, self.num_z1_modes, bias=False)

    def forward(self, z2_onehot: torch.Tensor) -> torch.Tensor:
        """
        Sample z1 embeddings given z2.

        Args:
            z2_onehot: Shape (batch_size, 2), one-hot encoded z2

        Returns:
            z1_embeddings: Shape (batch_size, 2), sampled z1 embeddings
        """
        batch_size = z2_onehot.shape[0]

        # Get logits for each z1 category
        logits = self.logits_table(z2_onehot)  # (batch_size, 4)

        # Sample from categorical distribution
        probs = F.softmax(logits, dim=-1)
        z1_indices = torch.multinomial(probs, num_samples=1).squeeze(
            -1
        )  # (batch_size,)

        # Look up embeddings
        z1_embeddings = self.embedding_table[z1_indices]  # (batch_size, 2)

        return z1_embeddings

    def get_log_probs(
        self, z2_onehot: torch.Tensor, z1_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log p(z1 | z2) for training.

        Args:
            z2_onehot: Shape (batch_size, 2)
            z1_embeddings: Shape (batch_size, 2), ground truth z1

        Returns:
            log_probs: Shape (batch_size,), log probability of each z1
        """
        batch_size = z2_onehot.shape[0]

        # Get logits
        logits = self.logits_table(z2_onehot)  # (batch_size, 4)
        log_probs_all = F.log_softmax(logits, dim=-1)  # (batch_size, 4)

        # Find which z1 index each embedding corresponds to
        # Vectorized match to nearest embedding in the table
        # self.embedding_table: (4, 2), z1_embeddings: (batch_size, 2)
        # Compute pairwise distances: (batch_size, 4)
        dists = torch.cdist(z1_embeddings, self.embedding_table)
        z1_indices = torch.argmin(dists, dim=-1)  # (batch_size,)

        # Compute log probabilities
        batch_log_probs = log_probs_all[torch.arange(batch_size), z1_indices]

        return batch_log_probs

        # Gather the log probabilities for the ground truth z1
        log_probs = log_probs_all.gather(1, z1_indices.unsqueeze(-1)).squeeze(-1)

        return log_probs


if __name__ == "__main__":
    print("Testing CategoricalPrior...")

    # Create dummy embeddings (on unit circle)
    import numpy as np

    z1_embeddings = {
        "dog": np.array([0.5, 0.866]),
        "cat": np.array([-0.5, 0.866]),
        "car": np.array([-0.5, -0.866]),
        "truck": np.array([0.5, -0.866]),
    }

    z1_subtypes = {
        "animal": ["dog", "cat"],
        "vehicle": ["car", "truck"],
    }

    # Create model
    model = CategoricalPrior(z1_embeddings, z1_subtypes)

    # Test forward pass
    print("\nTest: Sample z1 given z2='animal'")
    z2_onehot = torch.zeros(8, 2)
    z2_onehot[:, 0] = 1.0  # animal
    z1_samples = model(z2_onehot)
    print(f"Sampled z1 embeddings shape: {z1_samples.shape}")
    print(f"First 3 samples:\n{z1_samples[:3]}")

    # Test log probability
    print("\nTest: Compute log p(z1 | z2)")
    z1_gt = torch.stack(
        [
            torch.tensor(z1_embeddings["dog"], dtype=torch.float32),
            torch.tensor(z1_embeddings["cat"], dtype=torch.float32),
        ]
        * 4
    )
    log_probs = model.get_log_probs(z2_onehot, z1_gt)
    print(f"Log probabilities: {log_probs}")
    print(f"Mean log prob: {log_probs.mean():.3f}")

    print("\n✓ CategoricalPrior tests passed!")
