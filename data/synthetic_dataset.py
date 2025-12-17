"""
Synthetic hierarchical dataset for demonstrating mode collapse.

The dataset generates images from a known 3-level hierarchy:
    z2 (high-level category) -> z1 (subtype) -> z0 (style) -> image

This allows us to study mode collapse when the model only sees z2 but must
generate diverse z1 options.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import cv2


class SyntheticHierarchicalDataset(Dataset):
    """
    Generate synthetic images with known hierarchical structure.

    Hierarchy:
        - z2: Category (2 options: "animal", "vehicle")
        - z1: Subtype (2 per category: dog/cat, car/truck)
        - z0: Style (continuous: hue, brightness, position jitter)

    Images are 64x64 RGB with simple geometric shapes.
    """

    def __init__(self, num_samples: int = 10000, image_size: int = 64, seed: int = 42):
        """
        Args:
            num_samples: Number of samples in the dataset
            image_size: Size of square images (default 64x64)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed

        # Define hierarchy
        self.z2_categories = ["animal", "vehicle"]
        self.z1_subtypes = {"animal": ["dog", "cat"], "vehicle": ["car", "truck"]}

        # Ground truth conditional probabilities p(z1 | z2)
        # For this experiment, we'll use uniform distribution
        self.p_z1_given_z2 = {
            "animal": {"dog": 0.5, "cat": 0.5},
            "vehicle": {"car": 0.5, "truck": 0.5},
        }

        # Create fixed embeddings for each z1 subtype (2-dim, L2-normalized)
        # IMPORTANT: Group embeddings by z2 category so the prior can learn the structure!
        # Animals get top half of circle (0° to 180°), vehicles get bottom half (180° to 360°)
        np.random.seed(seed)
        self.z1_embeddings = {}
        
        # Assign distinct angular regions for each z2 category
        z2_angle_ranges = {
            "animal": (0, np.pi),      # 0° to 180° (top semicircle)
            "vehicle": (np.pi, 2*np.pi)  # 180° to 360° (bottom semicircle)
        }
        
        for z2 in self.z2_categories:
            angle_min, angle_max = z2_angle_ranges[z2]
            z1_list = self.z1_subtypes[z2]
            
            # Evenly space subtypes within this category's angular range
            for i, z1 in enumerate(z1_list):
                # Place subtypes evenly within the range, with some spacing from boundaries
                angle = angle_min + (angle_max - angle_min) * (i + 1) / (len(z1_list) + 1)
                vec = np.array([np.cos(angle), np.sin(angle)])
                self.z1_embeddings[z1] = vec

        # Pre-generate all samples for consistency
        self._generate_samples()

    def _generate_samples(self):
        """Pre-generate all dataset samples."""
        np.random.seed(self.seed)

        self.samples = []

        for idx in range(self.num_samples):
            # Sample z2 (category) uniformly
            z2 = np.random.choice(self.z2_categories)

            # Sample z1 (subtype) according to p(z1 | z2)
            z1_options = self.z1_subtypes[z2]
            z1_probs = [self.p_z1_given_z2[z2][z1] for z1 in z1_options]
            z1 = np.random.choice(z1_options, p=z1_probs)

            # Sample z0 (style parameters)
            z0 = {
                "hue": np.random.uniform(0, 180),  # HSV hue
                "brightness": np.random.uniform(0.5, 1.0),
                "jitter_x": np.random.uniform(-5, 5),
                "jitter_y": np.random.uniform(-5, 5),
                "scale": np.random.uniform(0.8, 1.2),
                "nose_offset_y": np.random.uniform(-3, 3) if z1 == "dog" else 0,  # Dogs have varying nose position
                "nose_size": np.random.uniform(2, 5) if z1 == "dog" else 3,  # Dogs have varying nose size
            }

            self.samples.append(
                {
                    "z2": z2,
                    "z1": z1,
                    "z0": z0,
                    "z1_embedding": self.z1_embeddings[z1].copy(),
                }
            )

    def _render_image(self, z1: str, z0: Dict) -> np.ndarray:
        """
        Render a 64x64 grayscale image based on z1 and z0.

        Shape encoding:
            - dog: Circle
            - cat: Circle with triangular ears
            - car: Rectangle (horizontal)
            - truck: Rectangle (vertical/taller)

        Args:
            z1: Subtype string
            z0: Style parameters dict

        Returns:
            Grayscale image as numpy array, shape (64, 64), dtype uint8
        """
        # Create blank canvas
        img = np.ones((self.image_size, self.image_size), dtype=np.uint8) * 255

        # Base position (center + jitter)
        center_x = int(self.image_size // 2 + z0["jitter_x"])
        center_y = int(self.image_size // 2 + z0["jitter_y"])

        # Base size with scale
        base_size = int(20 * z0["scale"])

        # Create shape based on z1
        if z1 == "dog":
            # Big floppy ears (like springer spaniels) - draw first so face overlaps
            ear_width = 8
            ear_height = 25
            # Left ear
            left_ear_pts = np.array(
                [
                    [center_x - base_size, center_y - 5],
                    [center_x - base_size - ear_width, center_y],
                    [center_x - base_size - ear_width, center_y + ear_height],
                    [center_x - base_size, center_y + ear_height],
                ],
                np.int32,
            )
            cv2.fillPoly(img, [left_ear_pts], (0, 0, 0))
            # Right ear
            right_ear_pts = np.array(
                [
                    [center_x + base_size, center_y - 5],
                    [center_x + base_size + ear_width, center_y],
                    [center_x + base_size + ear_width, center_y + ear_height],
                    [center_x + base_size, center_y + ear_height],
                ],
                np.int32,
            )
            cv2.fillPoly(img, [right_ear_pts], (0, 0, 0))
            # Circle (dog face)
            radius = base_size
            cv2.circle(img, (center_x, center_y), radius, (0, 0, 0), -1)
            # Eyes
            cv2.circle(img, (center_x - 8, center_y - 5), 3, (255, 255, 255), -1)
            cv2.circle(img, (center_x + 8, center_y - 5), 3, (255, 255, 255), -1)
            # White nose (varies in position and size)
            nose_y = int(center_y + 5 + z0["nose_offset_y"])
            nose_radius = int(z0["nose_size"])
            cv2.circle(img, (center_x, nose_y), nose_radius, (255, 255, 255), -1)

        elif z1 == "cat":
            # Circle with triangular ears
            radius = base_size
            cv2.circle(img, (center_x, center_y), radius, (0, 0, 0), -1)
            # Triangular ears
            ear_pts1 = np.array(
                [
                    [center_x - 15, center_y - 15],
                    [center_x - 8, center_y - 25],
                    [center_x - 5, center_y - 15],
                ],
                np.int32,
            )
            ear_pts2 = np.array(
                [
                    [center_x + 15, center_y - 15],
                    [center_x + 8, center_y - 25],
                    [center_x + 5, center_y - 15],
                ],
                np.int32,
            )
            cv2.fillPoly(img, [ear_pts1], (0, 0, 0))
            cv2.fillPoly(img, [ear_pts2], (0, 0, 0))
            # Eyes
            cv2.circle(img, (center_x - 8, center_y - 5), 3, (255, 255, 255), -1)
            cv2.circle(img, (center_x + 8, center_y - 5), 3, (255, 255, 255), -1)
            # White nose (fixed position and size)
            cv2.circle(img, (center_x, center_y + 5), 3, (255, 255, 255), -1)

        elif z1 == "car":
            # Horizontal rectangle
            width = int(base_size * 1.5)
            height = int(base_size * 0.8)
            top_left = (center_x - width, center_y - height)
            bottom_right = (center_x + width, center_y + height)
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), -1)
            # Windows
            cv2.rectangle(
                img,
                (center_x - width + 5, center_y - height + 5),
                (center_x, center_y),
                (255, 255, 255),
                -1,
            )
            cv2.rectangle(
                img,
                (center_x + 5, center_y - height + 5),
                (center_x + width - 5, center_y),
                (255, 255, 255),
                -1,
            )

        elif z1 == "truck":
            # Vertical rectangle (taller)
            width = int(base_size * 0.8)
            height = int(base_size * 1.5)
            top_left = (center_x - width, center_y - height)
            bottom_right = (center_x + width, center_y + height)
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), -1)
            # Front window
            cv2.rectangle(
                img,
                (center_x - width + 5, center_y - height + 5),
                (center_x + width - 5, center_y - height + 15),
                (255, 255, 255),
                -1,
            )

        # Apply brightness to grayscale image
        img = img.astype(np.float32)
        mask = img < 250  # Non-background pixels
        img[mask] = img[mask] * z0["brightness"]
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - 'image': Tensor of shape (1, 64, 64), normalized to [-1, 1]
                - 'z2': Category index (0 or 1)
                - 'z2_name': Category name string
                - 'z1': Subtype index (0 or 1 within category)
                - 'z1_name': Subtype name string
                - 'z1_embedding': Ground truth z1 embedding, shape (32,)
        """
        sample = self.samples[idx]

        # Render image
        img = self._render_image(sample["z1"], sample["z0"])

        # Convert to torch tensor and normalize to [-1, 1]
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
        img_tensor = (img_tensor / 127.5) - 1.0  # [0, 255] -> [-1, 1]

        # Convert categorical variables to indices
        z2_idx = self.z2_categories.index(sample["z2"])
        z1_options = self.z1_subtypes[sample["z2"]]
        z1_idx = z1_options.index(sample["z1"])

        return {
            "image": img_tensor,
            "z2": torch.tensor(z2_idx, dtype=torch.long),
            "z2_name": sample["z2"],
            "z1": torch.tensor(z1_idx, dtype=torch.long),
            "z1_name": sample["z1"],
            "z1_embedding": torch.from_numpy(sample["z1_embedding"]).float(),
        }

    def get_ground_truth_distribution(self, z2: str) -> Dict[str, float]:
        """Get the ground truth p(z1 | z2) distribution."""
        return self.p_z1_given_z2[z2].copy()

    def get_reference_images(self, z1: str, num_refs: int = 5) -> torch.Tensor:
        """
        Get reference images for a specific z1 subtype.

        Useful for evaluation (e.g., nearest neighbor matching).

        Args:
            z1: Subtype name
            num_refs: Number of reference images

        Returns:
            Tensor of shape (num_refs, 1, 64, 64)
        """
        refs = []
        for sample in self.samples:
            if sample["z1"] == z1:
                img = self._render_image(sample["z1"], sample["z0"])
                img_tensor = torch.from_numpy(img).float()
                img_tensor = img_tensor.unsqueeze(0)
                img_tensor = (img_tensor / 127.5) - 1.0
                refs.append(img_tensor)
                if len(refs) >= num_refs:
                    break

        return torch.stack(refs)

    def get_z2_embedding(self, z2: str) -> torch.Tensor:
        """
        Get embedding for z2 category (simple one-hot encoding).

        Args:
            z2: Category name or index

        Returns:
            One-hot tensor of shape (2,)
        """
        if isinstance(z2, str):
            z2_idx = self.z2_categories.index(z2)
        else:
            z2_idx = z2

        embedding = torch.zeros(len(self.z2_categories))
        embedding[z2_idx] = 1.0
        return embedding


def visualize_dataset_samples(
    dataset: SyntheticHierarchicalDataset, num_samples: int = 100
):
    """
    Visualize samples from the dataset.

    Args:
        dataset: The dataset to visualize
        num_samples: Number of samples to show
    """
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend to avoid Qt/display issues
    import matplotlib.pyplot as plt

    # Calculate grid size dynamically
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(grid_size * 1.5, grid_size * 1.5)
    )
    axes = axes.flatten()

    for i in range(min(num_samples, grid_size * grid_size)):
        sample = dataset[i]
        img = sample["image"].squeeze(0).numpy()  # Remove channel dimension
        img = (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"{sample['z2_name']}: {sample['z1_name']}", fontsize=6)
        axes[i].axis("off")

    # Hide any unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("dataset_samples.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to dataset_samples.png")


if __name__ == "__main__":
    # Test dataset generation
    print("Creating synthetic dataset...")
    dataset = SyntheticHierarchicalDataset(num_samples=1000)

    print(f"Dataset size: {len(dataset)}")
    print(f"Categories (z2): {dataset.z2_categories}")
    print(f"Subtypes (z1): {dataset.z1_subtypes}")

    # Check a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  z2: {sample['z2_name']} (index {sample['z2']})")
    print(f"  z1: {sample['z1_name']} (index {sample['z1']})")
    print(f"  z1 embedding shape: {sample['z1_embedding'].shape}")
    print(f"  z1 embedding norm: {torch.norm(sample['z1_embedding']):.4f}")

    # Visualize
    print("\nGenerating visualization...")
    visualize_dataset_samples(dataset)
