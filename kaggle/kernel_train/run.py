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
JOB = "smoke"  # tests | timing | smoke | train_base | train_phase2 | train_kn | train_all

WORK = "/kaggle/working"
CODE = "/tmp/repo"  # outside /kaggle/working, so the clone and the dataset are not saved as outputs
OUT = f"{WORK}/outputs"
PY = sys.executable
STATUS = {"job": JOB, "branch": BRANCH, "steps": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sync() -> None:
    """Copy results to the folder Kaggle keeps and refresh status.json. Never raises."""
    try:
        source = f"{CODE}/results"
        if os.path.isdir(source):
            shutil.copytree(source, f"{OUT}/results", dirs_exist_ok=True, symlinks=True)
    except Exception as error:  # noqa: BLE001 - a copy problem must not kill a multi-hour job
        say(f"WARNING: could not copy results: {error!r}")
    STATUS["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        with open(f"{OUT}/status.json", "w", encoding="utf-8") as handle:
            json.dump(STATUS, handle, indent=2)
    except Exception as error:  # noqa: BLE001
        say(f"WARNING: could not write status.json: {error!r}")


def step(name: str, command: list, required: bool = True, timeout: float = None) -> bool:
    """Run one step with its own log. A step that exceeds ``timeout`` seconds is killed (exit 124), so a
    kernel can never hang silently: it always ends and reports where the time went."""
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    say(f"START {name}: {' '.join(command)}")
    started = time.time()
    environment = {**os.environ, "WANDB_MODE": "offline", "WANDB_SILENT": "true", "PYTHONUNBUFFERED": "1",
                   "PYTHONUTF8": "1", "WANDB_DIR": f"{OUT}/wandb"}
    with open(f"{OUT}/logs/{name}.log", "w", encoding="utf-8") as log:
        try:
            code = subprocess.call(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                   timeout=timeout)
        except subprocess.TimeoutExpired:
            code = 124
            log.write(f"\nKILLED after {timeout} s (timeout)\n")
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


def prepare_cifar10() -> None:
    """Use the CIFAR-10 copy attached as a Kaggle input (the Toronto server is far too slow), then verify it.

    torchvision's integrity check compares every batch file against the official md5 checksums, so a
    modified copy is rejected. Falls back to a bounded download if no attached copy is found.
    """
    import glob

    hits = glob.glob("/kaggle/input/**/data_batch_1", recursive=True)
    destination = f"{CODE}/data/cifar-10-batches-py"
    if hits:
        os.makedirs(f"{CODE}/data", exist_ok=True)
        if not os.path.exists(destination):
            shutil.copytree(os.path.dirname(hits[0]), destination)
        say(f"CIFAR-10 taken from the attached input {os.path.dirname(hits[0])}")
    else:
        step("download_cifar10", [PY, "-m", "experiments.download_data", "--cifar10", "--data_dir", "./data"],
             timeout=1200)
    step("verify_cifar10", [PY, "-c",
                            "from torchvision import datasets as d;a=d.CIFAR10('./data',train=True,download=False);"
                            "b=d.CIFAR10('./data',train=False,download=False);print('md5 verified',len(a),len(b))"],
         timeout=600)


def train(seed: int, folder: str, extra: tuple = ()) -> None:
    step(f"train_{folder.replace('/', '_')}", timeout=14400, command=[
        PY, "-m", "experiments.train_adjsccq_train_split", "--seed", str(seed), "--split_seed", "42",
        "--skip_rs_sweep", "--data_dir", "./data", "--output_dir", f"results/{folder}",
        "--wandb", "--wandb_project", "rs-uep-jscc", "--wandb_name", f"kaggle-{folder.replace('/', '-')}", *extra])


def main() -> None:
    os.makedirs(f"{OUT}/wandb", exist_ok=True)
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
    prepare_cifar10()

    if JOB == "tests":
        step("pytest", [PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "--tb=short", "-x"],
             required=False, timeout=1500)
    elif JOB == "timing":
        step("benchmark", [PY, "-m", "experiments.benchmark_training", "--data_dir", "./data"], required=False,
             timeout=900)
    elif JOB == "smoke":
        step("pytest", [PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "--tb=short"], required=False,
             timeout=1500)
        step("train_smoke", [PY, "-m", "experiments.train_adjsccq_train_split", "--seed", "7", "--split_seed", "42",
                             "--epochs", "2", "--skip_rs_sweep", "--oracle_max_images", "500",
                             "--eval_max_images", "500", "--data_dir", "./data", "--output_dir", "results/smoke_train",
                             "--wandb", "--wandb_project", "rs-uep-jscc", "--wandb_name", "kaggle-smoke"],
             timeout=2400)
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
                "rs-uep-jscc", "--wandb_name", f"kaggle-{key}", *extra], timeout=9000)
    if JOB in ("train_kn", "train_all"):
        for ratio, tag in ((0.0625, "1_16"), (0.083333333, "1_12"), (0.333333333, "1_3")):
            train(42, f"phase3/kn_{tag}", ("--k_over_n", str(ratio)))
    if JOB not in ("tests", "timing", "smoke", "train_base", "train_phase2", "train_kn", "train_all"):
        raise SystemExit(f"unknown job {JOB}")
    STATUS["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    sync()
    say("done")


if __name__ == "__main__":
    main()
