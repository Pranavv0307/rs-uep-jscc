"""
CIFAR-10 data loading for DeepJSCC.
Images are returned as float tensors in [0, 1], shape (3, 32, 32).
No mean/std normalization is applied — DeepJSCC reconstructs pixel values
directly, so ImageNet-style normalization would break the [0,1] target.

data_dir defaults to a path that works on both Colab and Kaggle if you
pass the right value at call time (see notebooks in §5/§6) — this module
itself is platform-agnostic.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loaders(data_dir: str = "./cifar10_data",
                         batch_size: int = 64,
                         num_workers: int = 2):
    """
    Returns (train_loader, test_loader).
    Downloads CIFAR-10 into data_dir on first call if not already present.
    On Colab: point data_dir at /content/cifar10_data (ephemeral, fine —
        redownloads in ~20s each fresh session) or a Drive path to cache it.
    On Kaggle: point data_dir at /kaggle/working/cifar10_data, and make
        sure "Internet" is turned ON in the notebook's settings panel,
        or attach the "cifar10" Kaggle dataset instead (see §6.2).
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def get_overfit_subset_loader(data_dir: str = "./cifar10_data",
                                n_images: int = 10,
                                batch_size: int = 10):
    """Returns a DataLoader over exactly n_images for the Week 1 overfit test."""
    transform = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    subset = torch.utils.data.Subset(full_train, list(range(n_images)))
    return DataLoader(subset, batch_size=batch_size, shuffle=False)
