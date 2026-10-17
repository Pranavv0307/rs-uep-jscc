"""Image sources for every experiment: 32x32 sets, Kodak, and high-res training crops.

* ``load_image_tensor`` returns the first ``max_images`` images of a named
  dataset/split as one in-memory ``(N, 3, H, W)`` float tensor in [0, 1]. The
  order is deterministic, so batch ``i`` always contains the same images and the
  seeded packet-loss masks line up between runs and between methods.
* CIFAR-100 and STL-10 (resized to 32x32) are the *transfer* sets of Step A.1:
  models are never trained or tuned on them.
* ``KodakDataset`` reads the 24 lossless 768x512 Kodak PhotoCD images.
* ``FolderImageDataset`` serves random crops from any folder of PNG/JPG images
  (for the optional high-resolution training of Step C.1).
"""
import glob
import os
import urllib.request
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from data.splits import DEFAULT_SPLIT_SEED, DEFAULT_VAL_SIZE, cifar10_subsets

KODAK_URL = "http://r0k.us/graphics/kodak/kodak/kodim{index:02d}.png"
DATASETS_32 = ("cifar10", "cifar100", "stl10")


def _collect(dataset, max_images: Optional[int]) -> torch.Tensor:
    count = len(dataset) if max_images is None else min(max_images, len(dataset))
    return torch.stack([dataset[i][0] for i in range(count)])


def load_image_tensor(name: str, split: str, data_dir: str, max_images: Optional[int] = None,
                      split_seed: int = DEFAULT_SPLIT_SEED, val_size: int = DEFAULT_VAL_SIZE) -> torch.Tensor:
    """First ``max_images`` images of ``name``/``split`` as ``(N, 3, 32, 32)``.

    * cifar10: ``train`` (the 40k training subset), ``val`` (10k), ``test`` (official 10k)
    * cifar100: ``train`` / ``test`` (official)
    * stl10: ``train`` / ``test`` (official), downsampled 96 -> 32 with antialiasing
    """
    to_tensor = transforms.Compose([transforms.ToTensor()])
    if name == "cifar10":
        train, val, test = cifar10_subsets(data_dir, split_seed, val_size)
        dataset = {"train": train, "val": val, "test": test}[split]
    elif name == "cifar100":
        dataset = datasets.CIFAR100(root=data_dir, train=(split == "train"), download=True, transform=to_tensor)
    elif name == "stl10":
        resize = transforms.Compose([transforms.Resize(32, antialias=True), transforms.ToTensor()])
        dataset = datasets.STL10(root=data_dir, split=split, download=True, transform=resize)
    else:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS_32}")
    return _collect(dataset, max_images)


def is_official_test(name: str, split: str) -> bool:
    return name == "cifar10" and split == "test"


# --------------------------------------------------------------------------
# Kodak
# --------------------------------------------------------------------------
def download_kodak(root: str) -> List[str]:
    """Fetch kodim01-24.png into ``root`` (skips files that already exist)."""
    os.makedirs(root, exist_ok=True)
    paths = []
    for index in range(1, 25):
        path = os.path.join(root, f"kodim{index:02d}.png")
        if not os.path.exists(path):
            urllib.request.urlretrieve(KODAK_URL.format(index=index), path)
        paths.append(path)
    return paths


class KodakDataset(Dataset):
    """The Kodak images at native resolution, as ``(tensor, filename)``."""

    def __init__(self, root: str):
        self.paths = sorted(glob.glob(os.path.join(root, "kodim*.png")))
        if not self.paths:
            raise FileNotFoundError(f"no kodim*.png in {root}; run experiments/download_data.py --kodak")
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        from PIL import Image

        image = Image.open(self.paths[index]).convert("RGB")
        return self.to_tensor(image), os.path.basename(self.paths[index])


# --------------------------------------------------------------------------
# High-resolution training crops
# --------------------------------------------------------------------------
class FolderImageDataset(Dataset):
    """Random ``crop x crop`` patches from every image under ``root`` (recursive)."""

    EXTENSIONS = ("png", "jpg", "jpeg", "bmp", "webp")

    def __init__(self, root: str, crop: int = 256, train: bool = True, seed: int = 0):
        self.paths = sorted(
            path for ext in self.EXTENSIONS
            for path in glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
        )
        if not self.paths:
            raise FileNotFoundError(f"no images found under {root}")
        self.crop, self.train, self.seed = crop, train, seed
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        from PIL import Image

        image = self.to_tensor(Image.open(self.paths[index]).convert("RGB"))
        _, height, width = image.shape
        if height < self.crop or width < self.crop:
            image = torch.nn.functional.interpolate(
                image[None], size=(max(height, self.crop), max(width, self.crop)),
                mode="bilinear", align_corners=False, antialias=True,
            )[0]
            _, height, width = image.shape
        if self.train:
            top = int(torch.randint(0, height - self.crop + 1, (1,)))
            left = int(torch.randint(0, width - self.crop + 1, (1,)))
        else:  # fixed centre crop: evaluation patches never move
            top, left = (height - self.crop) // 2, (width - self.crop) // 2
        return image[:, top:top + self.crop, left:left + self.crop], index
