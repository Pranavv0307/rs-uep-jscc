"""Pre-flight checks: is everything in place before any experiment runs?

Read-only by default: nothing is trained, downloaded or modified, and secret values are
never printed (only whether they exist and how long they are). Everything is looked up
inside this project folder; credentials live in git-ignored files here (``.env``,
``.kaggle/``).

    python -m experiments.preflight                    # environment, data, checkpoints, credentials present?
    python -m experiments.preflight --wandb_test       # also: log in to W&B and log one tiny test run
    python -m experiments.preflight --kaggle_test      # also: authenticate against the Kaggle API
    python -m experiments.preflight --need cifar100,kodak   # also require those optional datasets

Writes ``results/logs/preflight.{json,md}``. Exit status is non-zero if a required check fails.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from experiments.reproducibility import REPO_ROOT, file_sha256, load_local_env

RESULTS: List[dict] = []


def record(name: str, status: str, detail: str = "") -> None:
    """status: PASS / WARN / FAIL / SKIP."""
    RESULTS.append({"check": name, "status": status, "detail": detail})
    print(f"[{status:4s}] {name}" + (f": {detail}" if detail else ""))


def git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def check_environment() -> None:
    record("python", "PASS", f"{platform.python_version()} on {platform.platform()}")
    for name in ("torch", "torchvision", "numpy", "PIL", "matplotlib", "yaml", "tqdm", "pytest", "wandb"):
        try:
            module = importlib.import_module(name)
            record(f"package {name}", "PASS", str(getattr(module, "__version__", "ok")))
        except Exception as error:  # noqa: BLE001
            optional = name in ("wandb", "pytest")
            record(f"package {name}", "WARN" if optional else "FAIL", f"{type(error).__name__}: {error}")
    import torch

    record("cuda", "PASS" if torch.cuda.is_available() else "WARN",
           torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU: training will be CPU-only")
    record("cpu cores", "PASS", str(os.cpu_count()))
    try:
        from PIL import features

        for feature, label in (("jpg_2000", "JPEG 2000"), ("webp", "WebP")):
            record(f"Pillow {label}", "PASS" if features.check(feature) else "WARN",
                   "available" if features.check(feature) else "missing: that codec is skipped in the digital baseline")
    except Exception as error:  # noqa: BLE001
        record("Pillow codecs", "WARN", str(error))
    record("lpips (optional)", "PASS" if importlib.util.find_spec("lpips") else "WARN",
           "installed" if importlib.util.find_spec("lpips") else "not installed: LPIPS column is skipped on Kodak")
    bpg = shutil.which("bpgenc") and shutil.which("bpgdec")
    record("BPG codec (optional)", "PASS" if bpg else "WARN",
           "found" if bpg else "not installed: digital baseline uses JPEG 2000 / JPEG / WebP")


def check_repo() -> None:
    record("repo root", "PASS", str(REPO_ROOT))
    branch, dirty = git("rev-parse", "--abbrev-ref", "HEAD"), git("status", "--porcelain", "--untracked-files=no")
    record("git branch", "PASS", f"{branch} @ {git('rev-parse', '--short', 'HEAD')}")
    record("git working tree", "PASS" if not dirty else "WARN",
           "clean" if not dirty else "uncommitted changes (results will record dirty_repo=true)")
    ignored = {pattern: bool(git("check-ignore", pattern)) for pattern in (".env", ".kaggle/kaggle.json", "results/x", "x.pth")}
    bad = [p for p, ok in ignored.items() if not ok]
    record("secrets and outputs are git-ignored", "PASS" if not bad else "FAIL",
           "ok" if not bad else f"NOT ignored: {bad}")
    tracked_secrets = [f for f in git("ls-files").splitlines()
                       if f.endswith((".env", "kaggle.json", ".netrc")) or "/.kaggle/" in f]
    record("no credential file is tracked", "PASS" if not tracked_secrets else "FAIL", str(tracked_secrets or "ok"))
    usage = shutil.disk_usage(REPO_ROOT)
    record("free disk space", "PASS" if usage.free > 20e9 else "WARN", f"{usage.free / 1e9:.1f} GB")
    try:
        probe = REPO_ROOT / "results" / "logs" / ".write_test"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok")
        probe.unlink()
        record("results/ is writable", "PASS")
    except OSError as error:
        record("results/ is writable", "FAIL", str(error))


def check_data(need: List[str], data_dir: str) -> None:
    from torchvision import datasets

    root = REPO_ROOT / data_dir
    for name, cls, expected in (("cifar10", datasets.CIFAR10, (50000, 10000)),
                                ("cifar100", datasets.CIFAR100, (50000, 10000))):
        required = name == "cifar10" or name in need
        try:
            train = cls(root=str(root), train=True, download=False)
            test = cls(root=str(root), train=False, download=False)  # md5-verifies the files
            ok = (len(train), len(test)) == expected
            record(f"dataset {name}", "PASS" if ok else "FAIL", f"{len(train)} train / {len(test)} test, md5 verified")
        except Exception as error:  # noqa: BLE001
            record(f"dataset {name}", "FAIL" if required else "SKIP",
                   f"{type(error).__name__}: not usable in {root} (download: python -m experiments.download_data --{name})")
    kodak = sorted((root / "kodak").glob("kodim*.png")) if (root / "kodak").is_dir() else []
    required = "kodak" in need
    record("dataset kodak", "PASS" if len(kodak) == 24 else ("FAIL" if required else "SKIP"),
           f"{len(kodak)}/24 images in {root / 'kodak'}")


def check_checkpoint(folder: str) -> None:
    path = REPO_ROOT / folder
    for name in ("best_val.pth", "oracle_ranking.json"):
        target = path / name
        if not target.exists():
            record(f"seed-42 checkpoint {name}", "WARN", f"missing in {folder} (not needed: models are retrained from scratch)")
            continue
        detail = f"sha256 {file_sha256(str(target))[:16]}..."
        if name.endswith(".json"):
            ranking = json.loads(target.read_text()).get("ranking_channels_most_to_least_important", [])
            detail += f", ranking is a permutation: {sorted(ranking) == list(range(len(ranking)))}"
        record(f"seed-42 checkpoint {name}", "PASS", detail)


def check_wandb(test: bool, project: str) -> None:
    loaded = load_local_env()
    key = os.environ.get("WANDB_API_KEY", "")
    record("W&B key in environment or .env", "PASS" if key else "WARN",
           f"present ({len(key)} characters)" if key else "absent: set WANDB_API_KEY in .env or run `wandb login`")
    if not test:
        record("W&B live test", "SKIP", "re-run with --wandb_test")
        return
    try:
        import wandb

        run = wandb.init(project=project, name="preflight-test", job_type="preflight", reinit=True,
                         config={"purpose": "connectivity check", "python": platform.python_version()})
        run.log({"preflight/ok": 1.0, "preflight/timestamp": time.time()})
        url = run.url
        run.finish()
        record("W&B live test", "PASS", f"logged one run to project '{project}': {url}")
    except Exception as error:  # noqa: BLE001
        record("W&B live test", "FAIL", f"{type(error).__name__}: {error}")


def check_kaggle(test: bool, username: str) -> None:
    load_local_env()
    config = REPO_ROOT / ".kaggle" / "kaggle.json"
    if config.exists():
        try:
            data = json.loads(config.read_text())
            ok = data.get("username") == username and bool(data.get("key"))
            record("Kaggle credentials (.kaggle/kaggle.json)", "PASS" if ok else "FAIL",
                   f"username {data.get('username')!r}" + ("" if ok else f" (expected {username!r}) or key missing"))
        except json.JSONDecodeError as error:
            record("Kaggle credentials (.kaggle/kaggle.json)", "FAIL", f"not valid JSON: {error}")
    elif os.environ.get("KAGGLE_API_TOKEN"):
        record("Kaggle credentials", "PASS", "KAGGLE_API_TOKEN set in .env")
    else:
        record("Kaggle credentials", "WARN", "absent: create a token at kaggle.com/settings and save as .kaggle/kaggle.json")
    package = importlib.util.find_spec("kaggle")
    vendor = REPO_ROOT / ".vendor"
    record("kaggle package", "PASS" if package or vendor.exists() else "WARN",
           "importable" if package else "not installed (will be installed into .vendor/, inside this folder)")
    if test:
        try:
            if vendor.exists():
                sys.path.insert(0, str(vendor))
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            record("Kaggle API login", "PASS", "authenticated")
        except Exception as error:  # noqa: BLE001
            record("Kaggle API login", "FAIL", f"{type(error).__name__}: {error}")
    else:
        record("Kaggle API login", "SKIP", "re-run with --kaggle_test")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--need", default="", help="comma list of optional datasets to REQUIRE: cifar100,kodak")
    parser.add_argument("--seed42_dir", default="checkpoints/train_split_adjsccq")
    parser.add_argument("--wandb_test", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    parser.add_argument("--kaggle_test", action="store_true")
    parser.add_argument("--kaggle_username", default="vaishakprasadbhat")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    need = [n.strip() for n in args.need.split(",") if n.strip()]

    print(f"== preflight {time.strftime('%Y-%m-%d %H:%M:%S')} ==")
    check_environment()
    check_repo()
    check_data(need, args.data_dir)
    check_checkpoint(args.seed42_dir)
    check_wandb(args.wandb_test, args.wandb_project)
    check_kaggle(args.kaggle_test, args.kaggle_username)

    counts = {s: sum(r["status"] == s for r in RESULTS) for s in ("PASS", "WARN", "FAIL", "SKIP")}
    out_dir = REPO_ROOT / "results" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight.json").write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                        "counts": counts, "checks": RESULTS}, indent=2))
    lines = ["# Preflight", "", f"{counts}", "", "| status | check | detail |", "|---|---|---|"]
    lines += [f"| {r['status']} | {r['check']} | {r['detail']} |" for r in RESULTS]
    (out_dir / "preflight.md").write_text("\n".join(lines) + "\n")
    print(f"\nsummary: {counts}  (details: results/logs/preflight.md)")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
