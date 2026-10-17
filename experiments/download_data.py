"""Download the datasets the later phases need (nothing is downloaded unless asked).

    python -m experiments.download_data --cifar10 --cifar100 --kodak
    python -m experiments.download_data --stl10        # 2.6 GB, optional

Sizes: CIFAR-10 170 MB, CIFAR-100 169 MB, Kodak 24 PNGs (~15 MB), STL-10 2.6 GB.
Everything lands under ``--data_dir`` (default ``./data``, git-ignored); Kodak goes
to ``<data_dir>/kodak``. CIFAR-100 is the recommended transfer set for Step A.1;
STL-10 is the roadmap's alternative if you have the bandwidth.
"""
import argparse
import os

from torchvision import datasets

from data.datasets import download_kodak


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--cifar10", action="store_true")
    parser.add_argument("--cifar100", action="store_true")
    parser.add_argument("--stl10", action="store_true")
    parser.add_argument("--kodak", action="store_true")
    args = parser.parse_args()
    if not any((args.cifar10, args.cifar100, args.stl10, args.kodak)):
        parser.error("choose at least one of --cifar10 --cifar100 --stl10 --kodak")

    if args.cifar10:
        for train in (True, False):
            datasets.CIFAR10(root=args.data_dir, train=train, download=True)
        print("CIFAR-10 ready")
    if args.cifar100:
        for train in (True, False):
            datasets.CIFAR100(root=args.data_dir, train=train, download=True)
        print("CIFAR-100 ready")
    if args.stl10:
        for split in ("train", "test"):
            datasets.STL10(root=args.data_dir, split=split, download=True)
        print("STL-10 ready")
    if args.kodak:
        paths = download_kodak(os.path.join(args.data_dir, "kodak"))
        print(f"Kodak ready: {len(paths)} images in {os.path.join(args.data_dir, 'kodak')}")


if __name__ == "__main__":
    main()
