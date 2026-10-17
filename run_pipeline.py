"""One entry point for every phase of the roadmap.

By default this only **prints** the commands (a dry run). Nothing is executed,
trained or downloaded until you add ``--execute``.

    python run_pipeline.py --list                       # all steps, grouped by phase
    python run_pipeline.py --phase verify               # show the commands of one phase
    python run_pipeline.py --phase phase1 --execute     # run it (logs in results/logs/)
    python run_pipeline.py --only train_seed100 --execute
    python run_pipeline.py --phase all --skip_existing --execute --workers 8 --wandb

Phases (in order): verify, phase1 (multi-seed + Track A), phase2 (Track B),
phase3 (Track C), phase4 (Track D), final (test tables + report).

Rules baked in (roadmap section 4): tuning (scheme selection, checkpoint choice)
uses the validation split; the test split is evaluated only in ``final``, once,
with the schemes chosen on validation; every full run uses 5 channel seeds
(0-4) and the whole split; every step is logged and recorded in
``results/logs/pipeline_manifest.jsonl`` with its git commit.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PHASES = ("verify", "phase1", "phase2", "phase3", "phase4", "final")


@dataclass
class Step:
    name: str
    phase: str
    description: str
    command: List[str]
    produces: List[str] = field(default_factory=list)   # a step is "done" when all exist
    wandb: bool = False                                  # append --wandb flags when requested
    workers: bool = False                                # append --workers when requested


def build_steps(cfg: argparse.Namespace) -> List[Step]:
    py = sys.executable
    data = cfg.data_dir
    base42 = cfg.seed42_dir
    r = cfg.results

    def mod(name: str, *arguments) -> List[str]:
        return [py, "-m", f"experiments.{name}", *[str(a) for a in arguments]]

    def best(folder: str) -> str:
        return f"{folder}/best_val.pth"

    def ranking(folder: str) -> str:
        return f"{folder}/oracle_ranking.json"

    seeds = "0,1,2,3,4"
    rates = "0.0,0.1,0.2,0.3,0.35,0.4,0.45,0.5"

    def sweep(name, phase, description, ckpt, split, importance, out_dir, label, rank=None, head=None,
              schemes="all", rates_arg=rates, extra=(), produces=None):
        command = mod("eval_rs_uep", "--checkpoint", ckpt, "--data_dir", data, "--split", split,
                      "--importance", importance, "--schemes", schemes, "--erasure_rates", rates_arg,
                      "--seeds", seeds, "--output_dir", out_dir, "--name", label, *extra)
        if rank:
            command += ["--oracle_ranking_file", rank]
        if head:
            command += ["--head_checkpoint", head]
        return Step(name, phase, description, command, produces or [f"{out_dir}/{label}.json"], True, True)

    steps: List[Step] = []

    # ------------------------------------------------------------------ verify
    steps += [
        Step("unit_tests", "verify", "Run every unit test", [py, "-m", "pytest", "tests", "-q"]),
        sweep("smoke_eval", "verify", "Smoke test (200 images, 2 seeds) - numbers are NOT quoted anywhere",
              base42 + "/best_val.pth", "val", "attention,oracle", f"{r}/smoke", "smoke", rank=ranking(base42),
              schemes="three_tier_25_50_25", rates_arg="0.0,0.4,0.5",
              extra=("--max_images", 200, "--include_no_rs")),
    ]
    steps[-1].command[steps[-1].command.index("--seeds") + 1] = "0,1"

    # ------------------------------------------------------------------ phase 1
    for seed in (100, 2024):
        folder = f"{r}/models/seed{seed}"
        steps.append(Step(f"train_seed{seed}", "phase1",
                          f"Train a model from scratch with init seed {seed} (same 40k/10k split, no in-script sweep)",
                          mod("train_adjsccq_train_split", "--seed", seed, "--split_seed", 42, "--skip_rs_sweep",
                              "--data_dir", data, "--output_dir", folder),
                          [best(folder), ranking(folder)], True))
    runs = ["--run", f"seed42={base42}", "--run", f"seed100={r}/models/seed100", "--run",
            f"seed2024={r}/models/seed2024"]
    steps += [
        Step("multiseed", "phase1", "Compare the oracle damage profile across initialisation seeds",
             mod("multiseed_analysis", *runs, "--output_dir", f"{r}/phase1/multiseed"),
             [f"{r}/phase1/multiseed/multiseed.json"], True),
        Step("attention_vs_damage", "phase1", "Figure 2: attention vs true damage (validation images)",
             mod("attention_vs_damage", "--checkpoint", best(base42), "--data_dir", data, "--split", "val",
                 "--output_dir", f"{r}/phase1/attention"),
             [f"{r}/phase1/attention/attention_vs_damage.json"], True),
        Step("download_cifar100", "phase1", "Download CIFAR-100 (transfer set, 169 MB)",
             mod("download_data", "--cifar100", "--data_dir", data), [f"{data}/cifar-100-python"]),
        Step("transfer_A1", "phase1", "Step A.1: does the ranking transfer to CIFAR-100?",
             mod("transfer_sensitivity", "--checkpoint", best(base42), "--reference_ranking", ranking(base42),
                 "--targets", "cifar100", "--data_dir", data, "--output_dir", f"{r}/phase1/transfer"),
             [f"{r}/phase1/transfer/transfer.json"], True),
        Step("saliency_A2", "phase1", "Step A.2: proxy quality (energy, variance, L1, Taylor) vs true damage",
             mod("feature_saliency", "--checkpoint", best(base42), "--reference_ranking", ranking(base42),
                 "--data_dir", data, "--output_dir", f"{r}/phase1/saliency"),
             [f"{r}/phase1/saliency/saliency.json"], True),
        Step("head_A3", "phase1", "Step A.3: label 40k+10k images and train the sensitivity head (5 head seeds)",
             mod("train_sensitivity_head", "--checkpoint", best(base42), "--reference_ranking", ranking(base42),
                 "--data_dir", data, "--labels_dir", f"{r}/phase1/head_labels", "--output_dir", f"{r}/phase1/head",
                 "--seeds", seeds),
             [f"{r}/phase1/head/sensitivity_head.pt"], True),
        sweep("saliency_sweep", "phase1", "Validation sweep at 40% loss: every importance source vs oracle bounds",
              best(base42), "val", "attention,energy,variance,l1,taylor,predicted,oracle,oracle_perimage",
              f"{r}/phase1/saliency_sweep", "saliency_sweep", rank=ranking(base42),
              head=f"{r}/phase1/head/sensitivity_head.pt", schemes="three_tier_25_50_25", rates_arg="0.3,0.4,0.5",
              extra=("--include_random_control", "--include_no_rs")),
        sweep("val_sweep_base", "phase1", "Validation sweep, ALL schemes, for choosing the tier allocation",
              best(base42), "val", "attention,oracle,predicted", f"{r}/phase1/val_sweep", "val_sweep",
              rank=ranking(base42), head=f"{r}/phase1/head/sensitivity_head.pt",
              extra=("--include_random_control", "--include_no_rs")),
        Step("select_base", "phase1", "Select the tier allocation on the validation sweep",
             mod("select_scheme", "--sweep", f"{r}/phase1/val_sweep/val_sweep.json", "--output",
                 f"{r}/phase1/val_sweep/selected_schemes.json"),
             [f"{r}/phase1/val_sweep/selected_schemes.json"]),
    ]

    # ------------------------------------------------------------------ phase 2
    variants = {
        "robust_channel": ("Step B.1: channel-dropout fine-tuning (roadmap setting)", ("--drop_unit", "channel")),
        "robust_packet": ("Step B.1 ablation: packet-level dropout (what a failed RS block produces)",
                          ("--drop_unit", "packet")),
        "robust_inpaint": ("Step B.2: packet dropout + masked-latent transformer", ("--drop_unit", "packet", "--inpaint")),
    }
    for key, (description, extra) in variants.items():
        folder = f"{r}/phase2/{key}"
        steps.append(Step(f"train_{key}", "phase2", description,
                          mod("train_erasure_robust", "--init_checkpoint", best(base42), "--data_dir", data,
                              "--output_dir", folder, *extra),
                          [f"{folder}/adjsccq_erasure_robust.pth", ranking(folder)], True))
        steps.append(sweep(f"val_{key}", "phase2", f"Validation sweep of {key} (its own oracle ranking)",
                           f"{folder}/adjsccq_erasure_robust.pth", "val", "oracle", f"{folder}/val_sweep", "val_sweep",
                           rank=ranking(folder), extra=("--include_no_rs",)))
        steps.append(Step(f"select_{key}", "phase2", f"Select the tier allocation for {key} on its validation sweep",
                          mod("select_scheme", "--sweep", f"{folder}/val_sweep/val_sweep.json", "--output",
                              f"{folder}/val_sweep/selected_schemes.json"),
                          [f"{folder}/val_sweep/selected_schemes.json"]))

    # ------------------------------------------------------------------ phase 3
    steps += [
        Step("download_kodak", "phase3", "Download the 24 Kodak images (~15 MB)",
             mod("download_data", "--kodak", "--data_dir", data), [f"{data}/kodak/kodim24.png"]),
        Step("kodak_cifar_model", "phase3", "Step C.1: Kodak at 768x512 with the CIFAR-trained model",
             mod("eval_kodak", "--checkpoint", best(base42), "--oracle_ranking_file", ranking(base42),
                 "--kodak_dir", f"{data}/kodak", "--importance", "oracle,attention", "--seeds", seeds,
                 "--include_no_rs", "--save_images_rate", 0.4, "--output_dir", f"{r}/phase3/kodak"),
             [f"{r}/phase3/kodak/kodak.json"], True, True),
    ]
    if cfg.highres_dir:
        folder = f"{r}/phase3/highres"
        steps += [
            Step("train_highres", "phase3", "Step C.1 fallback: fine-tune on high-resolution patches",
                 mod("train_highres", "--init_checkpoint", best(base42), "--train_dir", cfg.highres_dir,
                     "--output_dir", folder), [best(folder), ranking(folder)], True),
            Step("kodak_highres_model", "phase3", "Kodak with the high-resolution model",
                 mod("eval_kodak", "--checkpoint", best(folder), "--oracle_ranking_file", ranking(folder),
                     "--kodak_dir", f"{data}/kodak", "--importance", "oracle,attention", "--seeds", seeds,
                     "--include_no_rs", "--save_images_rate", 0.4, "--output_dir", f"{r}/phase3/kodak_highres"),
                 [f"{r}/phase3/kodak_highres/kodak.json"], True, True),
        ]
    for ratio, tag in ((0.0625, "1_16"), (0.083333333, "1_12"), (0.333333333, "1_3")):
        folder = f"{r}/phase3/kn_{tag}"
        steps.append(Step(f"train_kn_{tag}", "phase3", f"Step C.2: train a model at k/n = {ratio:.4f}",
                          mod("train_adjsccq_train_split", "--k_over_n", ratio, "--seed", 42, "--split_seed", 42,
                              "--skip_rs_sweep", "--data_dir", data, "--output_dir", folder),
                          [best(folder), ranking(folder)], True))
        steps.append(sweep(f"val_kn_{tag}", "phase3", f"Validation sweep at k/n = {ratio:.4f}", best(folder), "val",
                           "oracle", f"{folder}/val_sweep", "val_sweep", rank=ranking(folder),
                           extra=("--include_no_rs",)))
        steps.append(Step(f"select_kn_{tag}", "phase3", f"Select the tier allocation at k/n = {ratio:.4f}",
                          mod("select_scheme", "--sweep", f"{folder}/val_sweep/val_sweep.json", "--output",
                              f"{folder}/val_sweep/selected_schemes.json"),
                          [f"{folder}/val_sweep/selected_schemes.json"]))

    # ------------------------------------------------------------------ phase 4
    steps += [
        Step("burst_val", "phase4", "Step D.1: Gilbert-Elliott bursts, interleaved vs sequential (validation)",
             mod("eval_burst", "--checkpoint", best(base42), "--oracle_ranking_file", ranking(base42),
                 "--data_dir", data, "--split", "val", "--avg_losses", "0.35,0.45", "--burst_lengths", "1,2,4,8,16",
                 "--seeds", seeds, "--include_no_rs", "--output_dir", f"{r}/phase4/burst"),
             [f"{r}/phase4/burst/burst_all.json"], True, True),
        Step("digital_val", "phase4", "Step D.2: JPEG2000 / JPEG / WebP (+BPG if installed) + RS over the same losses",
             mod("eval_digital_baseline", "--data_dir", data, "--split", "val", "--seeds", seeds,
                 "--output_dir", f"{r}/phase4/digital"),
             [f"{r}/phase4/digital/digital.json"], True, True),
    ]

    # ------------------------------------------------------------------ final
    sel = f"{r}/phase1/val_sweep/selected_schemes.json"
    final_common = ("--include_random_control", "--include_no_rs")
    steps += [
        sweep("final_base", "final", "FINAL TEST sweep, base model (validation-selected schemes, 5 seeds, full test set)",
              best(base42), "test", "attention,oracle,predicted", f"{r}/final", "final_test", rank=ranking(base42),
              head=f"{r}/phase1/head/sensitivity_head.pt", schemes=f"{{selected:{sel}}}", extra=final_common),
    ]
    for key in variants:
        folder = f"{r}/phase2/{key}"
        own = f"{folder}/val_sweep/selected_schemes.json"
        steps.append(sweep(f"final_{key}", "final", f"FINAL TEST sweep, {key} (schemes chosen on its own validation)",
                           f"{folder}/adjsccq_erasure_robust.pth", "test", "oracle", f"{folder}/test_sweep",
                           "test_sweep", rank=ranking(folder), schemes=f"{{selected:{own}}}",
                           extra=("--include_no_rs",)))
    for tag in ("1_16", "1_12", "1_3"):
        folder = f"{r}/phase3/kn_{tag}"
        own = f"{folder}/val_sweep/selected_schemes.json"
        steps.append(sweep(f"final_kn_{tag}", "final", f"FINAL TEST sweep, k/n sweep point {tag}", best(folder),
                           "test", "oracle", f"{folder}/test_sweep", "test_sweep", rank=ranking(folder),
                           schemes=f"{{selected:{own}}}", extra=("--include_no_rs",)))
    steps += [
        Step("burst_test", "final", "FINAL TEST burst study (Gilbert-Elliott, interleaved vs sequential)",
             mod("eval_burst", "--checkpoint", best(base42), "--oracle_ranking_file", ranking(base42),
                 "--data_dir", data, "--split", "test", "--avg_losses", "0.35,0.45", "--burst_lengths",
                 "1,2,4,8,16", "--seeds", seeds, "--include_no_rs", "--schemes", f"{{selected:{sel}}}",
                 "--output_dir", f"{r}/final/burst"), [f"{r}/final/burst/burst_all.json"], True, True),
        Step("digital_test", "final", "FINAL TEST digital baseline",
             mod("eval_digital_baseline", "--data_dir", data, "--split", "test", "--seeds", seeds,
                 "--output_dir", f"{r}/final/digital"), [f"{r}/final/digital/digital.json"], True, True),
        Step("report", "final", "Figures, headline table and INSIGHTS.md",
             mod("make_report", "--main", f"{r}/final/final_test.json",
                 "--compare", f"base={r}/final/final_test.json",
                 "--compare", f"robust_channel={r}/phase2/robust_channel/test_sweep/test_sweep.json",
                 "--compare", f"robust_packet={r}/phase2/robust_packet/test_sweep/test_sweep.json",
                 "--compare", f"robust_inpaint={r}/phase2/robust_inpaint/test_sweep/test_sweep.json",
                 "--digital", f"{r}/final/digital/digital.json",
                 "--kn", f"0.0625={r}/phase3/kn_1_16/test_sweep/test_sweep.json",
                 "--kn", f"0.0833={r}/phase3/kn_1_12/test_sweep/test_sweep.json",
                 "--kn", f"0.1667={r}/final/final_test.json",
                 "--kn", f"0.3333={r}/phase3/kn_1_3/test_sweep/test_sweep.json",
                 "--extra_insights", f"multi-seed oracle={r}/phase1/multiseed/multiseed.json",
                 "--extra_insights", f"attention vs damage={r}/phase1/attention/attention_vs_damage.json",
                 "--extra_insights", f"cross-dataset transfer={r}/phase1/transfer/transfer.json",
                 "--extra_insights", f"proxy quality={r}/phase1/saliency/saliency.json",
                 "--extra_insights", f"sensitivity head={r}/phase1/head/head_metrics.json",
                 "--out_dir", f"{r}/report"), [f"{r}/report/INSIGHTS.md"], True),
    ]
    return steps


def resolve(command: List[str], dry_run: bool) -> List[str]:
    """Replace ``{selected:<file>}`` by the schemes recorded on the validation sweep."""
    out = []
    for token in command:
        if token.startswith("{selected:") and token.endswith("}"):
            path = token[len("{selected:"):-1]
            if os.path.exists(path):
                with open(path, encoding="utf-8") as handle:
                    token = json.load(handle)["schemes_arg"]
            elif dry_run:
                token = f"<schemes chosen on validation: read from {path}>"
            else:
                raise SystemExit(f"{path} is missing: run the validation sweep and select_base first")
        out.append(token)
    return out


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", default=None, help=f"one of {PHASES} or 'all'")
    parser.add_argument("--only", default=None, help="comma list of step names")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--execute", action="store_true", help="actually run (default: print only)")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    parser.add_argument("--workers", type=int, default=1, help="CPU workers for the evaluation sweeps")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--results", default="results")
    parser.add_argument("--seed42_dir", default="checkpoints/train_split_adjsccq",
                        help="folder with best_val.pth + oracle_ranking.json of the seed-42 model")
    parser.add_argument("--highres_dir", default=None, help="folder of high-res images for the Kodak fallback")
    cfg = parser.parse_args()

    steps = build_steps(cfg)
    if cfg.list:
        for phase in PHASES:
            print(f"\n[{phase}]")
            for step in (s for s in steps if s.phase == phase):
                print(f"  {step.name:24s} {step.description}")
        return
    selected = steps
    if cfg.only:
        names = {n.strip() for n in cfg.only.split(",")}
        unknown = names - {s.name for s in steps}
        if unknown:
            raise SystemExit(f"unknown steps: {sorted(unknown)}")
        selected = [s for s in steps if s.name in names]
    elif cfg.phase and cfg.phase != "all":
        if cfg.phase not in PHASES:
            raise SystemExit(f"--phase must be one of {PHASES} or 'all'")
        selected = [s for s in steps if s.phase == cfg.phase]
    elif not cfg.phase:
        parser.error("choose --phase, --only or --list")

    os.makedirs(f"{cfg.results}/logs", exist_ok=True)
    for number, step in enumerate(selected, 1):
        command = resolve(list(step.command), dry_run=not cfg.execute)
        if cfg.wandb and step.wandb:
            command += ["--wandb", "--wandb_project", cfg.wandb_project]
        if step.workers and cfg.workers > 1:
            command += ["--workers", str(cfg.workers)]
        print(f"\n[{number}/{len(selected)}] {step.phase}/{step.name}: {step.description}")
        print("  " + subprocess.list2cmdline(command))
        if not cfg.execute:
            continue
        if cfg.skip_existing and step.produces and all(os.path.exists(p) for p in step.produces):
            print("  -> skipped (outputs exist)")
            continue
        started = time.time()
        log_path = f"{cfg.results}/logs/{step.name}.log"
        with open(log_path, "w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record = {"step": step.name, "phase": step.phase, "command": command, "returncode": process.returncode,
                  "seconds": round(time.time() - started, 1), "git_commit": git_commit(), "log": log_path,
                  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(f"{cfg.results}/logs/pipeline_manifest.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"  -> exit {process.returncode} in {record['seconds']} s (log: {log_path})")
        if process.returncode != 0:
            raise SystemExit(f"step {step.name} failed; see {log_path}")


if __name__ == "__main__":
    main()
