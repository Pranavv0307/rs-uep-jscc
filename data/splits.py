"""The 40k / 10k / 10k CIFAR-10 protocol (roadmap Rule 1), in one place.

* 40,000 images of the official training split: model training, quantizer
  calibration, training-derived oracle ranking, sensitivity-head labels.
* 10,000 remaining training images: validation (checkpoint selection, UEP
  scheme selection, early stopping).
* The 10,000-image official test split: final tables only.

The split is a deterministic permutation of the 50,000 training indices drawn
from ``split_seed``. It is deliberately separate from the *model* seed, so the
multi-seed experiment (models trained from seeds 42 / 100 / 2024) uses the same
train/validation partition every time.
"""
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEFAULT_SPLIT_SEED = 42
DEFAULT_VAL_SIZE = 10_000


def split_indices(num_train: int, split_seed: int = DEFAULT_SPLIT_SEED,
                  val_size: int = DEFAULT_VAL_SIZE) -> Tuple[list, list]:
    """``(train_indices, val_indices)``; identical to the original train-split script."""
    if not 0 < val_size < num_train:
        raise ValueError(f"val_size must be between 1 and {num_train - 1}")
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(num_train, generator=generator).tolist()
    return permutation[:-val_size], permutation[-val_size:]


def cifar10_subsets(data_dir: str, split_seed: int = DEFAULT_SPLIT_SEED,
                    val_size: int = DEFAULT_VAL_SIZE):
    """``(train_subset, val_subset, test_dataset)`` with ``ToTensor`` transforms."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    train_indices, val_indices = split_indices(len(train_dataset), split_seed, val_size)
    return Subset(train_dataset, train_indices), Subset(train_dataset, val_indices), test_dataset


def make_loaders(data_dir: str, batch_size: int, num_workers: int, seed: int, val_size: int,
                 loader_seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Train / validation / test loaders.

    ``seed`` is the *split* seed. ``loader_seed`` seeds the training shuffle and
    defaults to ``seed + 1`` (the original behaviour).
    """
    from experiments.reproducibility import seed_worker

    train_subset, val_subset, test_dataset = cifar10_subsets(data_dir, seed, val_size)
    loader_generator = torch.Generator().manual_seed(seed + 1 if loader_seed is None else loader_seed)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, generator=loader_generator,
        num_workers=num_workers, pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, worker_init_fn=seed_worker,
    )
    return train_loader, val_loader, test_loader
