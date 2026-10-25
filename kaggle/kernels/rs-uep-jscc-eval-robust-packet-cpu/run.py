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
JOB = "eval_robust_packet"  # tests | timing | smoke | train_base | train_phase2 | train_kn | train_all

WORK = "/kaggle/working"
CODE = "/tmp/repo"  # outside /kaggle/working, so the clone and the dataset are not saved as outputs
OUT = f"{WORK}/outputs"
PY = sys.executable
GPU = shutil.which("nvidia-smi") is not None
TRAIN_TIMEOUT = 14400 if GPU else 40000  # seconds per model; a CPU session needs far longer than a GPU one
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
    step(f"train_{folder.replace('/', '_')}", timeout=TRAIN_TIMEOUT, command=[
        PY, "-m", "experiments.train_adjsccq_train_split", "--seed", str(seed), "--split_seed", "42",
        "--skip_rs_sweep", "--data_dir", "./data", "--output_dir", f"results/{folder}",
        "--wandb", "--wandb_project", "rs-uep-jscc", "--wandb_name", f"kaggle-{folder.replace('/', '-')}", *extra])


V2_RECIPE = ("--lr_schedule", "cosine", "--grad_clip", "1.0")  # stabilised recipe (see PROGRESS_LOG)


def v2_train(seed: int, folder: str, extra: tuple = ()) -> None:
    train(seed, f"v2/{folder}", (*V2_RECIPE, *extra))


V3_RECIPE = ("--lr", "3e-4", "--lr_schedule", "cosine", "--grad_clip", "1.0")  # lower base lr (see PROGRESS_LOG)


def v3_train(seed: int, folder: str, extra: tuple = ()) -> None:
    train(seed, f"v3/{folder}", (*V3_RECIPE, *extra))


def v2_robust(init_folder: str) -> None:
    variants = {"robust_channel": ("--drop_unit", "channel"), "robust_packet": ("--drop_unit", "packet"),
                "robust_inpaint": ("--drop_unit", "packet", "--inpaint")}
    for key, extra in variants.items():
        step(f"train_v2_{key}", [
            PY, "-m", "experiments.train_erasure_robust", "--init_checkpoint", f"results/{init_folder}/best_val.pth",
            "--data_dir", "./data", "--output_dir", f"results/v2/phase2/{key}", "--wandb", "--wandb_project",
            "rs-uep-jscc", "--wandb_name", f"kaggle-v2-{key}", *extra], timeout=9000)


EVAL_SCHEMES = "three_tier_25_25_50,three_tier_25_50_25,two_tier_25_75,two_tier_50_50,two_tier_75_25"  # set from the validation selection before launch
EVAL_SOURCES = "oracle,energy"


def find_model(name: str):
    """Checkpoint and ranking of ``name`` (seed100, kn_1_3, robust_channel, ...) in the attached models dataset."""
    import glob

    for filename in ("best_val.pth", "adjsccq_erasure_robust.pth"):
        hits = glob.glob(f"/kaggle/input/**/{name}/{filename}", recursive=True)
        if hits:
            return hits[0], os.path.join(os.path.dirname(hits[0]), "oracle_ranking.json")
    raise SystemExit(f"model {name} not found in /kaggle/input")


def evaluate_model(name: str) -> None:
    checkpoint, ranking = find_model(name)
    step(f"eval_{name}", [
        PY, "-m", "experiments.eval_rs_uep", "--checkpoint", checkpoint, "--oracle_ranking_file", ranking,
        "--data_dir", "./data", "--split", "test", "--importance", EVAL_SOURCES, "--schemes", EVAL_SCHEMES,
        "--seeds", "0,1,2,3,4", "--include_no_rs", "--workers", "4", "--device", "cpu",
        "--output_dir", f"results/{model_folder(name)}" + ("" if model_folder(name).startswith("multiseed") else "/test_sweep"),
        "--name", "test_sweep", "--wandb", "--wandb_project", "rs-uep-jscc",
        "--wandb_name", f"kaggle-eval-{name}"], timeout=40000)


def model_folder(name: str) -> str:
    """Where the pipeline expects this model's results (relative to the results root)."""
    if name.startswith("robust_"):
        return f"phase2/{name}"
    if name.startswith("kn_"):
        return f"phase3/{name}"
    return f"multiseed/{name}"  # base models trained from other seeds


def evaluate_validation(name: str, folder: str) -> None:
    """Validation sweep used to choose the tier allocation (3 seeds x 5,000 images, all schemes)."""
    checkpoint, ranking = find_model(name)
    step(f"evalval_{name}", [
        PY, "-m", "experiments.eval_rs_uep", "--checkpoint", checkpoint, "--oracle_ranking_file", ranking,
        "--data_dir", "./data", "--split", "val", "--importance", "oracle,energy", "--schemes", "all",
        "--seeds", "0,1,2", "--max_images", "5000", "--include_no_rs", "--workers", "4", "--device", "cpu",
        "--output_dir", f"results/{folder}/val_sweep", "--name", "val_sweep", "--wandb", "--wandb_project",
        "rs-uep-jscc", "--wandb_name", f"kaggle-evalval-{name}"], timeout=40000)


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
    # Original recipe (v1), extra seeds. Decided after recipes v2/v3 failed to improve on v1 (see PROGRESS_LOG).
    if JOB.startswith("evalval_"):
        model = JOB[len("evalval_"):]
        evaluate_validation(model, {"kn_1_16": "phase3/kn_1_16", "kn_1_12": "phase3/kn_1_12", "kn_1_3": "phase3/kn_1_3",
                                    "robust_channel": "phase2/robust_channel", "robust_packet": "phase2/robust_packet",
                                    "robust_inpaint": "phase2/robust_inpaint"}.get(model, f"models/{model}"))
    if JOB.startswith("eval_"):
        evaluate_model(JOB[len("eval_"):])
    if JOB == "v1x_a":
        for seed in (7, 13):
            train(seed, f"v1x/models/seed{seed}")
    if JOB == "v1x_b":
        train(21, "v1x/models/seed21")
        train(7, "v1x/phase3/kn_1_3_seed7", ("--k_over_n", "0.333333333"))
    if JOB == "v3_a":
        v3_train(2024, "models/seed2024")
        v3_train(42, "models/seed42")
    if JOB == "v3_b":
        v3_train(42, "phase3/kn_1_3", ("--k_over_n", "0.333333333"))
        v3_train(100, "models/seed100")
    if JOB == "v2_a":
        v2_train(42, "models/seed42")
        v2_robust("v2/models/seed42")
        v2_train(2024, "models/seed2024")
    if JOB == "v2_b":
        v2_train(100, "models/seed100")
        for ratio, tag in ((0.0625, "1_16"), (0.083333333, "1_12"), (0.333333333, "1_3")):
            v2_train(42, f"phase3/kn_{tag}", ("--k_over_n", str(ratio)))
    if JOB.startswith("train_seed"):
        seed = int(JOB[len("train_seed"):])
        train(seed, f"models/seed{seed}")
    if JOB.startswith("train_kn_"):
        tag = JOB[len("train_kn_"):]
        ratio = {"1_16": 0.0625, "1_12": 0.083333333, "1_3": 0.333333333}[tag]
        train(42, f"phase3/kn_{tag}", ("--k_over_n", str(ratio)))
    if not (JOB in ("tests", "timing", "smoke", "train_base", "train_phase2", "train_kn", "train_all")
            or JOB in ("v2_a", "v2_b", "v3_a", "v3_b", "v1x_a", "v1x_b") or JOB.startswith(("eval_", "evalval_")) or JOB.startswith(("train_seed", "train_kn_"))):
        raise SystemExit(f"unknown job {JOB}")
    STATUS["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    sync()
    say("done")


if __name__ == "__main__":
    main()
