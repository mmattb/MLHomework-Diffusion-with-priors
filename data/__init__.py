"""Data loading utilities."""

from torch.utils.data import DataLoader
from .synthetic_dataset import SyntheticHierarchicalDataset


def get_dataloader(
    num_samples: int = 10000,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader for the synthetic dataset.

    Args:
        num_samples: Number of samples in dataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        seed: Random seed

    Returns:
        PyTorch DataLoader
    """
    dataset = SyntheticHierarchicalDataset(
        num_samples=num_samples, image_size=64, seed=seed
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader, dataset
