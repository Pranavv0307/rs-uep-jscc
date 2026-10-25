"""Bring the output of a finished Kaggle kernel into the local results tree.

    python -m experiments.ingest_kaggle --kernel rs-uep-jscc-eval-seed13-cpu --dest results/gpu

Downloads ``outputs/`` to ``results/kaggle/<kernel>/``, copies every file under ``outputs/results/`` into
``--dest`` (merging, never deleting), appends the kernel's test-split audit entries to
``results/test_access_log.jsonl`` (once; duplicates are skipped) and syncs the offline W&B runs to the
project account in ``.env``. Prints the kernel's step table so a failed step is visible.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

from experiments.reproducibility import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel", required=True, help="kernel slug, e.g. rs-uep-jscc-eval-seed13-cpu")
    parser.add_argument("--dest", default="results/gpu", help="results root that receives outputs/results/*")
    parser.add_argument("--owner", default="vaishakprasadbhat")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)

    local = os.path.join("results", "kaggle", args.kernel)
    shutil.rmtree(local, ignore_errors=True)
    os.makedirs(local)
    subprocess.run([sys.executable, "kaggle/kg.py", "kernels", "output", f"{args.owner}/{args.kernel}", "-p", local],
                   check=True, capture_output=True)
    status_path = os.path.join(local, "outputs", "status.json")
    if not os.path.exists(status_path):
        raise SystemExit(f"{args.kernel}: no status.json in the output (kernel failed early?) -> see {local}")
    with open(status_path, encoding="utf-8") as handle:
        status = json.load(handle)
    failed = [s for s in status["steps"] if s["returncode"] != 0]
    for step in status["steps"]:
        print(f"  {step['name']:36s} exit {step['returncode']:>3}  {step['seconds']:>9}s")
    if failed:
        print(f"WARNING: {len(failed)} step(s) failed; results below may be partial")

    source = os.path.join(local, "outputs", "results")
    audit = os.path.join(source, "test_access_log.jsonl")
    for folder, _dirs, files in os.walk(source):
        for name in files:
            if name == "test_access_log.jsonl":
                continue
            target = os.path.join(args.dest, os.path.relpath(os.path.join(folder, name), source))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(os.path.join(folder, name), target)
    if os.path.exists(audit):
        log = os.path.join("results", "test_access_log.jsonl")
        known = set(open(log, encoding="utf-8").read().splitlines()) if os.path.exists(log) else set()
        new = [line for line in open(audit, encoding="utf-8").read().splitlines() if line and line not in known]
        with open(log, "a", encoding="utf-8") as handle:
            handle.writelines(line + "\n" for line in new)
        print(f"audit log: {len(new)} new test-access entries")
    if not args.no_wandb:
        for run in sorted(glob.glob(os.path.join(local, "outputs", "wandb", "wandb", "offline-run-*"))):
            result = subprocess.run([sys.executable, "-m", "wandb", "sync", run], capture_output=True, text=True)
            print("wandb sync:", (result.stdout.strip().splitlines() or ["?"])[-1])
    print(f"copied results into {args.dest}")


if __name__ == "__main__":
    main()
