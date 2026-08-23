import torch

from data.cifar10 import get_cifar10_loaders


def test_cifar10_loader():
    train_loader, test_loader = get_cifar10_loaders(
        data_dir="./cifar10_data",
        batch_size=4,
        num_workers=0
    )

    images, labels = next(iter(train_loader))

    print("Image shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Minimum pixel value:", images.min().item())
    print("Maximum pixel value:", images.max().item())

    assert images.shape == (4, 3, 32, 32)
    assert labels.shape == (4,)
    assert images.dtype == torch.float32

    assert images.min() >= 0.0
    assert images.max() <= 1.0

    assert len(train_loader.dataset) == 50000
    assert len(test_loader.dataset) == 10000