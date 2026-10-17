"""Reproducibility helpers shared by every training / evaluation entry point.

Three jobs, kept in one place so no script re-implements them:

* :func:`seed_everything` -- deterministic seeding (roadmap Rule 5).
* :func:`build_provenance` -- the metadata block every JSON output carries
  (roadmap section 10): git commit, dirty flag, versions, device, checkpoint
  SHA-256, command line.
* :func:`aggregate_over_seeds` -- mean +/- std (and 95% CI) across channel
  seeds. The per-image std in the raw rows measures image-to-image spread,
  not seed-to-seed spread, so it must not be reported as the error bar.
"""
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# Two-sided 95% Student-t critical values indexed by degrees of freedom.
_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 25: 2.060, 30: 2.042,
}


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------
def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG and (by default) force deterministic kernels.

    ``cudnn.deterministic`` makes cuDNN pick reproducible convolution
    algorithms; ``cudnn.benchmark = False`` stops the auto-tuner from
    switching algorithms between runs. ``use_deterministic_algorithms`` covers
    the remaining PyTorch ops and only warns (instead of raising) when an op
    has no deterministic implementation. Call this before building models or
    touching CUDA.
    """
    # PYTHONHASHSEED only affects child processes started after this point;
    # it is set so any subprocess (e.g. DataLoader workers) inherits it.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """``DataLoader(worker_init_fn=...)``: derive numpy/random seeds from the loader's torch seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _describe_file(path: str) -> Dict[str, Optional[str]]:
    exists = os.path.exists(path)
    return {"path": path, "sha256": file_sha256(path) if exists else None}


def build_provenance(
    args: Any = None,
    checkpoint_path: Optional[str] = None,
    extra_files: Optional[Dict[str, str]] = None,
    dataset: str = "CIFAR-10",
    split_info: Optional[Dict] = None,
    global_seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """Metadata block for a result file. Never raises on a missing git/checkpoint."""
    porcelain = _git("status", "--porcelain", "--untracked-files=no")
    if device is not None and getattr(device, "type", None) == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or platform.machine() or "cpu"
    metadata = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_repo": None if porcelain is None else bool(porcelain),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "device": str(device) if device is not None else None,
        "device_name": device_name,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "dataset": dataset,
        "split": split_info,
        "global_seed": global_seed,
        "determinism": {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "command_line": " ".join(sys.argv),
        "args": {key: _jsonable(value) for key, value in vars(args).items()} if args is not None else None,
        "checkpoint": _describe_file(checkpoint_path) if checkpoint_path else None,
        "files": {name: _describe_file(path) for name, path in (extra_files or {}).items()},
    }
    if extra:
        metadata.update(extra)
    return metadata


# --------------------------------------------------------------------------
# Statistics over channel seeds
# --------------------------------------------------------------------------
def t_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("need at least one degree of freedom")
    if degrees_of_freedom > 30:
        return 1.96
    # Largest tabulated df not above the requested one -> never under-covers.
    return _T_95[max(key for key in _T_95 if key <= degrees_of_freedom)]


def seed_statistics(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Mean, sample std (n-1) and 95% t-interval half-width across seeds."""
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    mean = float(np.mean(values))
    if n < 2:
        return {"n_seeds": n, "mean": mean, "std": None, "ci95_half_width": None}
    std = float(np.std(values, ddof=1))
    return {
        "n_seeds": n,
        "mean": mean,
        "std": std,
        "ci95_half_width": t_critical_95(n - 1) * std / n ** 0.5,
    }


# Fields that are identical for every seed of a (method, rate) cell.
_CONSTANT_FIELDS = (
    "side_info_bytes_per_image",
    "coding_overhead_fraction",
    "total_transmitted_bytes_per_image",
    "total_overhead_fraction_including_side_info",
)
DEFAULT_SUMMARY_METRICS = ("psnr_db_mean_per_image", "ssim_mean", "realized_erasure_rate")


def aggregate_over_seeds(
    rows: Iterable[Dict],
    group_keys: Sequence[str] = ("method", "erasure_rate"),
    metrics: Sequence[str] = DEFAULT_SUMMARY_METRICS,
    seed_key: str = "seed",
) -> List[Dict]:
    """Collapse per-seed rows into one row per group: mean / std / CI across seeds."""
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)

    summary = []
    for key, members in groups.items():
        seeds = [member[seed_key] for member in members]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seeds in group {key}: {seeds}")
        entry = dict(zip(group_keys, key))
        entry["seeds"] = sorted(seeds)
        entry["n_seeds"] = len(seeds)
        entry["images_per_seed"] = members[0].get("images")
        for metric in metrics:
            if metric not in members[0]:
                continue
            stats = seed_statistics([member[metric] for member in members])
            entry[f"{metric}__mean"] = stats["mean"]
            entry[f"{metric}__std_over_seeds"] = stats["std"]
            entry[f"{metric}__ci95_half_width"] = stats["ci95_half_width"]
        for field in _CONSTANT_FIELDS:
            if field in members[0]:
                entry[field] = members[0][field]
        summary.append(entry)
    return summary


# --------------------------------------------------------------------------
# Test-set access audit trail (roadmap Rule 1)
# --------------------------------------------------------------------------
TEST_ACCESS_LOG = "results/test_access_log.jsonl"


def log_test_access(script: str, checkpoint: Optional[str] = None, note: str = "",
                    log_path: str = TEST_ACCESS_LOG) -> None:
    """Append one line recording that the official test split was evaluated.

    The test set is meant to be used for final tables only. This log makes every
    access auditable (who/what/when/which model), so "we only looked once" is a
    checkable statement rather than a promise.
    """
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": script,
        "command_line": " ".join(sys.argv),
        "git_commit": _git("rev-parse", "HEAD"),
        "checkpoint": checkpoint,
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint and os.path.exists(checkpoint) else None,
        "note": note,
    }
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def write_json(path: str, payload: Any) -> None:
    """Create parent folders and write ``payload`` as indented JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
