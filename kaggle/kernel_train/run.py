"""Kaggle launcher: clone the branch, run a named job on the GPU, save everything.

This file runs INSIDE a Kaggle kernel (GPU + internet on). It carries no credentials:
Weights & Biases is forced offline here, and the run folders are downloaded with the
outputs and synced from the owner's machine (``wandb sync``), so the W&B key never
leaves that machine.

Edit JOB below, push the kernel, then fetch outputs with kaggle/kg.py (see kaggle/README.md).
Every step writes its own log; ``status.json`` is refreshed after each step, so partial
results survive a failure or a session timeout.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time

REPO_URL = "https://github.com/Subhashhari/rs-uep-jscc.git"
BRANCH = "phase0-5-pipeline"
JOB = "smoke"  # smoke | train_base | train_phase2 | train_kn

WORK = "/kaggle/working"
CODE = f"{WORK}/repo"
OUT = f"{WORK}/outputs"
PY = sys.executable
STATUS = {"job": JOB, "branch": BRANCH, "steps": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sync() -> None:
    """Copy results, logs and offline W&B runs to the folder Kaggle keeps."""
    for source, target in ((f"{CODE}/results", f"{OUT}/results"), (f"{CODE}/wandb", f"{OUT}/wandb")):
        if os.path.isdir(source):
            shutil.copytree(source, target, dirs_exist_ok=True)
    STATUS["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(f"{OUT}/status.json", "w", encoding="utf-8") as handle:
        json.dump(STATUS, handle, indent=2)


def step(name: str, command: list, required: bool = True) -> bool:
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    say(f"START {name}: {' '.join(command)}")
    started = time.time()
    environment = {**os.environ, "WANDB_MODE": "offline", "WANDB_SILENT": "true", "PYTHONUNBUFFERED": "1",
                   "PYTHONUTF8": "1"}
    with open(f"{OUT}/logs/{name}.log", "w", encoding="utf-8") as log:
        code = subprocess.call(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT, env=environment)
    seconds = round(time.time() - started, 1)
    STATUS["steps"].append({"name": name, "command": command, "returncode": code, "seconds": seconds})
    say(f"END   {name}: exit {code} in {seconds}s")
    sync()
    if code != 0:
        with open(f"{OUT}/logs/{name}.log", encoding="utf-8", errors="replace") as log:
            print("".join(log.readlines()[-30:]), flush=True)
        if required:
            raise SystemExit(f"required step {name} failed")
    return code == 0


def train(seed: int, folder: str, extra: tuple = ()) -> None:
    step(f"train_{folder.replace('/', '_')}", [
        PY, "-m", "experiments.train_adjsccq_train_split", "--seed", str(seed), "--split_seed", "42",
        "--skip_rs_sweep", "--data_dir", "./data", "--output_dir", f"results/{folder}",
        "--wandb", "--wandb_project", "rs-uep-jscc", "--wandb_name", f"kaggle-{folder.replace('/', '-')}", *extra])


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    say(f"job={JOB} python={platform.python_version()} platform={platform.platform()}")
    if os.path.isdir(CODE):
        shutil.rmtree(CODE)
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, CODE])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE, text=True).strip()
    STATUS["git_commit"] = commit
    say(f"cloned {BRANCH} @ {commit}")
    step("environment", [PY, "-c",
                         "import torch,torchvision,sys;print(sys.version);print('torch',torch.__version__,"
                         "'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '');"
                         "print('cpus',__import__('os').cpu_count())"], required=False)
    subprocess.call(["bash", "-c", f"nvidia-smi > {OUT}/nvidia-smi.txt 2>&1"])
    step("download_cifar10", [PY, "-m", "experiments.download_data", "--cifar10", "--data_dir", "./data"])

    if JOB == "smoke":
        step("pytest", [PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "--tb=short"], required=False)
        step("train_smoke", [PY, "-m", "experiments.train_adjsccq_train_split", "--seed", "7", "--split_seed", "42",
                             "--epochs", "2", "--skip_rs_sweep", "--oracle_max_images", "500",
                             "--eval_max_images", "500", "--data_dir", "./data", "--output_dir", "results/smoke_train",
                             "--wandb", "--wandb_project", "rs-uep-jscc", "--wandb_name", "kaggle-smoke"])
    elif JOB in ("train_base", "train_all"):
        for seed in (42, 100, 2024):
            train(seed, f"models/seed{seed}")
    if JOB in ("train_phase2", "train_all"):
        if JOB == "train_phase2" and not os.path.exists(f"{CODE}/results/models/seed42/best_val.pth"):
            raise SystemExit("train_phase2 alone needs results/models/seed42; use JOB=train_all")
        variants = {"robust_channel": ("--drop_unit", "channel"), "robust_packet": ("--drop_unit", "packet"),
                    "robust_inpaint": ("--drop_unit", "packet", "--inpaint")}
        for key, extra in variants.items():
            step(f"train_{key}", [
                PY, "-m", "experiments.train_erasure_robust", "--init_checkpoint", "results/models/seed42/best_val.pth",
                "--data_dir", "./data", "--output_dir", f"results/phase2/{key}", "--wandb", "--wandb_project",
                "rs-uep-jscc", "--wandb_name", f"kaggle-{key}", *extra])
    if JOB in ("train_kn", "train_all"):
        for ratio, tag in ((0.0625, "1_16"), (0.083333333, "1_12"), (0.333333333, "1_3")):
            train(42, f"phase3/kn_{tag}", ("--k_over_n", str(ratio)))
    if JOB not in ("smoke", "train_base", "train_phase2", "train_kn", "train_all"):
        raise SystemExit(f"unknown job {JOB}")
    STATUS["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    sync()
    say("done")


if __name__ == "__main__":
    main()
