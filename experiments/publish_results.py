"""Copy the SMALL, publishable result files from ``results/`` into the tracked ``published_results/``.

Heavy and regenerable files stay in ``results/`` (git-ignored): per-image arrays, label caches, datasets,
W&B run folders, raw Kaggle downloads. What is copied: result tables (json/md), figures (png/pdf), model
checkpoints and rankings, and per-step logs, each under a size cap. A manifest with sizes and SHA-256 hashes is
written so every published file is traceable.

    python -m experiments.publish_results                    # copy
    python -m experiments.publish_results --dry_run          # list only
"""
import argparse
import fnmatch
import os
import shutil
import time

from experiments.reproducibility import REPO_ROOT, file_sha256, write_json

INCLUDE = ("*.json", "*.md", "*.png", "*.pdf", "*.pth", "*.pt", "*.log", "*.jsonl", "*.txt")
EXCLUDE_DIRS = {"wandb", "kaggle", "kaggle_upload", "head_labels", "__pycache__", "images", "smoke", "smoke_train", "smoke_robust",
                "smoke_robust_inpaint", "smoke_head", "smoke_head_labels", "smoke_burst", "smoke_digital",
                "smoke_kodak", "smoke_transfer", "smoke_saliency", "smoke_attention", "smoke_multiseed",
                "smoke_report", "smoke_inpaint_eval"}
EXCLUDE_NAMES = ("*.per_image.npz", "labels_*.pt", "final.pth")
CAPS = {".pth": 3_000_000, ".pt": 3_000_000, ".log": 400_000, ".json": 3_000_000, ".md": 1_000_000,
        ".png": 2_000_000, ".pdf": 2_000_000, ".jsonl": 1_000_000, ".txt": 200_000}


def wanted(path: str, size: int) -> bool:
    name = os.path.basename(path)
    if any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDE_NAMES):
        return False
    if not any(fnmatch.fnmatch(name, pattern) for pattern in INCLUDE):
        return False
    return size <= CAPS.get(os.path.splitext(name)[1].lower(), 1_000_000)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="results")
    parser.add_argument("--target", default="published_results")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    source, target = REPO_ROOT / args.source, REPO_ROOT / args.target

    manifest, skipped_large, total = [], [], 0
    for folder, dirs, files in os.walk(source):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for name in files:
            path = os.path.join(folder, name)
            size = os.path.getsize(path)
            if not wanted(path, size):
                if size > 1_000_000 and not any(fnmatch.fnmatch(name, p) for p in EXCLUDE_NAMES):
                    skipped_large.append((os.path.relpath(path, source), size))
                continue
            relative = os.path.relpath(path, source)
            manifest.append({"file": relative.replace(os.sep, "/"), "bytes": size, "sha256": file_sha256(path)})
            total += size
            if not args.dry_run:
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    print(f"{len(manifest)} files, {total / 1e6:.1f} MB -> {target}" + (" (dry run)" if args.dry_run else ""))
    for relative, size in skipped_large[:15]:
        print(f"  left out (too large): {relative} ({size / 1e6:.1f} MB)")
    if not args.dry_run:
        write_json(str(target / "MANIFEST.json"), {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                    "source": args.source, "files": manifest})


if __name__ == "__main__":
    main()
