"""Packet-erasure sweep of tiered Reed-Solomon coding on a trained ADJSCC-Q model.

For every channel seed and loss rate, every method is run on the same images
and the same loss realisation:

* ``no_rs``          64 packets, no parity (optional)
* ``uniform``        one RS block over all channels, no ranking, 0 B side info
* ``<scheme>@<source>``  a tiered scheme whose channel order comes from an
  importance *source* (``attention``, ``oracle``, ``energy``, ``variance``,
  ``l1``, ``taylor``, ``predicted``, ``oracle_perimage``, ``static:<name>``,
  ``random``; see ``experiments/importance_sources.py``); 16 B/img side info.

Rules enforced (roadmap section 4): one master loss mask shared by all methods
(Rule 3), 0.0 latent fallback with no ground-truth leakage (Rule 2), overhead
reported with and without side information (Rule 4), deterministic seeding
(Rule 5). Outputs: JSON (rows, seed summary, paired gains, insights, provenance),
markdown, per-image PSNR arrays, and vector+PNG figures.

``--split val`` is the default so tuning never touches the test set; use
``--split test`` only for final tables (every test access is logged to
``results/test_access_log.jsonl``).

Example (full protocol, final test table)::

    python -m experiments.eval_rs_uep --checkpoint checkpoints/train_split_adjsccq/best_val.pth \\
        --split test --importance attention,oracle --oracle_ranking_file \\
        checkpoints/train_split_adjsccq/oracle_ranking.json --seeds 0,1,2,3,4 \\
        --include_no_rs --include_random_control --workers 8 --output_dir results/eval/final
"""
import argparse
import itertools
import os
from typing import Dict, List

import numpy as np
import torch

from coding.erasure_channel import sample_master_mask
from coding.interleave import packet_mask_from_timeline
from coding.rs_pipeline import SCHEME_CANDIDATES
from coding.scaling import scheme_catalog
from data.datasets import is_official_test, load_image_tensor
from experiments import wandb_utils
from experiments.model_io import load_model_bundle, load_sensitivity_head
from experiments.pipeline_core import (
    CellAccumulator,
    batch_ranges,
    canonical_source,
    make_context,
    make_method_specs,
    method_order,
    module_function,
    precompute,
    run_tasks,
    transmit,
)
from experiments.reporting import write_sweep_outputs
from experiments.reproducibility import (
    build_provenance,
    file_sha256,
    log_test_access,
    seed_everything,
)

DEFAULT_ERASURE_RATES = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5)
ORACLE_KEYS = ("ranking_channels_most_to_least_important", "oracle_ranking_channels")
MODULE = "experiments.eval_rs_uep"


def parse_csv(value, cast=str):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_ranking_file(path: str, channels: int = 16, warn_split: str = "train"):
    """Read a frozen ranking written by either oracle workflow; validate it is a permutation."""
    import json

    with open(path, "r") as handle:
        data = json.load(handle)
    for key in ORACLE_KEYS:
        if key in data:
            ranking = [int(c) for c in data[key]]
            break
    else:
        raise KeyError(f"{path} has none of the ranking keys {ORACLE_KEYS}")
    if sorted(ranking) != list(range(len(ranking))):
        raise ValueError(f"ranking must be a permutation of 0..{len(ranking) - 1}, got {ranking}")
    source_split = data.get("source_split", "unknown")
    if warn_split and source_split != warn_split:
        print(f"WARNING: ranking {path} has source_split='{source_split}', not '{warn_split}'. A ranking derived "
              "from evaluation data leaks into the evaluation (roadmap Rule 1); only acceptable as an upper bound.")
    return ranking, {"path": path, "sha256": file_sha256(path), "source_split": source_split}


# Backwards-compatible name used by earlier scripts and tests.
load_oracle_ranking = load_ranking_file


def collect_rankings(args) -> Dict[str, dict]:
    """``{name: {"ranking": [...], "info": {...}}}`` from --oracle_ranking_file and --rankings."""
    entries = []
    if args.oracle_ranking_file:
        entries.append(("oracle", args.oracle_ranking_file))
    for item in parse_csv(args.rankings or ""):
        name, _, path = item.partition("=")
        entries.append((name, path))
    out = {}
    for name, path in entries:
        ranking, info = load_ranking_file(path)
        out[name] = {"ranking": ranking, "info": info}
    return out


def build_context(payload):
    args = payload["args"]
    device = torch.device(payload["device"])
    bundle = load_model_bundle(args["checkpoint"], device)
    specs = payload["specs"]
    schemes = scheme_catalog(bundle.channels, bundle.symbols_per_channel, args["packet_size"], args["overhead"],
                             names=sorted({spec.scheme for spec in specs}))
    context = make_context(bundle, payload["X"], payload["prep"], schemes, args["batch_size"])
    context.specs = specs
    context.args = args
    context.master_n = max(scheme.total_packets for scheme in schemes.values())
    return context


def run_cell(context, task):
    """One (channel seed, loss rate) cell: every method on every image."""
    seed, rate = task
    args = context.args
    constants = {"erasure_rate": rate, "seed": seed, "dataset": args["dataset"], "split": args["split"],
                 "checkpoint": args["checkpoint"], "channels": context.channels}
    accumulators = {
        spec.name: CellAccumulator(spec, context.schemes[spec.scheme], context.channels, constants)
        for spec in context.specs
    }
    for batch_index, start, stop in batch_ranges(len(context.X), context.batch_size):
        x = context.X[start:stop].to(context.device)
        idx = context.prep["idx"][start:stop].long().to(context.device)
        mask_generator = torch.Generator().manual_seed(seed * 100000 + batch_index)
        master = sample_master_mask(stop - start, rate, mask_generator, num_packets=context.master_n)
        for spec in context.specs:
            scheme = context.schemes[spec.scheme]
            order_generator = torch.Generator().manual_seed(seed * 100000 + batch_index + 50000)
            order = method_order(spec, context.prep["orders"], start, stop, context.channels, order_generator,
                                 context.device)
            mask = packet_mask_from_timeline(master, scheme, spec.interleave)
            recon, decoded, erased, total = transmit(context.bundle, idx, scheme, order, mask, context.snr_db)
            accumulators[spec.name].add(recon, x, decoded, erased, total)
    rows, per_image = [], {}
    for name, accumulator in accumulators.items():
        row, psnr = accumulator.finish()
        rows.append(row)
        per_image[f"{name}__s{seed}__r{rate:.3f}"] = psnr
    return rows, per_image


def build_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "stl10"])
    parser.add_argument("--split", default="val", help="cifar10: train|val|test; others: train|test")
    parser.add_argument("--split_seed", type=int, default=None, help="default: the checkpoint's own split seed")
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--max_images", type=int, default=None, help="default: the whole split")
    parser.add_argument("--schemes", default="all",
                        help="comma list from SCHEME_CANDIDATES, or 'all'")
    parser.add_argument("--importance", default="attention",
                        help="comma list: attention, oracle, energy, variance, l1, taylor, predicted, "
                             "oracle_perimage, static:<name>")
    parser.add_argument("--oracle_ranking_file", default=None, help="ranking used by source 'oracle'")
    parser.add_argument("--rankings", default=None, help="extra static rankings: name=path,name=path")
    parser.add_argument("--head_checkpoint", default=None, help="sensitivity head for source 'predicted'")
    parser.add_argument("--erasure_rates", default=",".join(map(str, DEFAULT_ERASURE_RATES)))
    parser.add_argument("--seeds", default="0,1,2,3,4", help="channel (erasure-mask) seeds")
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--packet_size", type=int, default=16)
    parser.add_argument("--overhead", type=float, default=0.5)
    parser.add_argument("--include_no_rs", action="store_true")
    parser.add_argument("--include_random_control", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="parallel (seed, rate) cells; results are identical")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | auto (workers always use cpu)")
    parser.add_argument("--output_dir", default="results/eval/sweep")
    parser.add_argument("--name", default="sweep")
    parser.add_argument("--cdf_rate", type=float, default=0.4)
    parser.add_argument("--no_per_image", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_group", default=None)
    return parser


def prepare_data(args, sources=None, note: str = ""):
    """Load model, images and rankings; precompute indices, orders and clean PSNR.

    Shared by every packet-loss study. Returns a dict with ``bundle``, ``device``,
    ``split_seed``, ``rankings``, ``sources``, ``images``, ``prep``.
    """
    device = pick_device(args.device)
    bundle = load_model_bundle(args.checkpoint, device)
    split_seed = args.split_seed if args.split_seed is not None else bundle.split_seed
    if is_official_test(args.dataset, args.split):
        log_test_access(f"experiments.{note or 'eval'}", args.checkpoint, note=f"name={args.name}")
    rankings = collect_rankings(args)
    sources = sources if sources is not None else [canonical_source(s) for s in parse_csv(args.importance or "attention")]
    for source in sources:
        if source.startswith("static:") and source[7:] not in rankings:
            raise SystemExit(f"importance '{source}' needs a ranking named '{source[7:]}' "
                             "(--rankings name=path / --oracle_ranking_file)")
    images = load_image_tensor(args.dataset, args.split, args.data_dir, args.max_images, split_seed)
    head = load_sensitivity_head(args.head_checkpoint, device) if args.head_checkpoint else None
    try:
        from tqdm import tqdm as progress
    except ImportError:  # pragma: no cover
        progress = None
    prep = precompute(bundle, images, list(sources), args.batch_size,
                      {name: entry["ranking"] for name, entry in rankings.items()}, head, progress=progress)
    return {"bundle": bundle, "device": device, "split_seed": split_seed, "rankings": rankings,
            "sources": list(sources), "images": images, "prep": prep}


def evaluation_metadata(args, data: dict, extra: dict) -> dict:
    """Provenance block shared by every study built on ``prepare_data``."""
    bundle, rankings = data["bundle"], data["rankings"]
    extra_files = {f"ranking_{name}": entry["info"]["path"] for name, entry in rankings.items()}
    if args.head_checkpoint:
        extra_files["head"] = args.head_checkpoint
    evaluation = {
        "importance_sources": data["sources"],
        "rankings": {name: {**entry["info"], "ranking": entry["ranking"]} for name, entry in rankings.items()},
        "images_per_seed": len(data["images"]),
        "is_full_test_set": bool(is_official_test(args.dataset, args.split) and len(data["images"]) >= 10_000),
        "channels": bundle.channels, "packet_size": args.packet_size, "overhead": args.overhead,
        "clean_psnr_mean": float(np.mean(data["prep"]["clean_psnr"])),
        "side_info_rule": "0 B for uniform/no_rs, one byte per channel for sorted methods",
        **extra,
    }
    return build_provenance(
        args=args, checkpoint_path=args.checkpoint, global_seed=args.global_seed, device=data["device"],
        extra_files=extra_files,
        split_info={"dataset": args.dataset, "split": args.split, "split_seed": data["split_seed"]},
        extra={"evaluation": evaluation})


def run_and_save(args, default_sources=None):
    seed_everything(args.global_seed)
    if not args.importance and default_sources:
        args.importance = ",".join(default_sources)
    schemes = list(SCHEME_CANDIDATES) if args.schemes == "all" else list(parse_csv(args.schemes))
    seeds, rates = parse_csv(args.seeds, int), parse_csv(args.erasure_rates, float)
    data = prepare_data(args, note="eval_rs_uep")
    bundle, device, images, prep, rankings, sources = (
        data["bundle"], data["device"], data["images"], data["prep"], data["rankings"], data["sources"])
    specs = make_method_specs(schemes, sources, args.include_no_rs, args.include_random_control)
    print(f"{len(images)} images ({args.dataset}/{args.split}), {len(specs)} methods, "
          f"{len(seeds)} seeds x {len(rates)} rates, channels={bundle.channels}")

    run = wandb_utils.init_run(args.wandb, args.wandb_project, args.wandb_name or f"eval-{args.name}",
                               config=vars(args), tags=["eval", args.dataset, args.split], group=args.wandb_group,
                               job_type="eval")
    payload = {"args": vars(args), "device": "cpu" if args.workers > 1 else str(device), "specs": specs,
               "X": images, "prep": prep}

    def log_cell(task, result):
        seed, rate = task
        metrics = {"seed": seed, "erasure_rate": rate}
        for row in result[0]:
            metrics[f"{row['method']}/psnr_db"] = row["psnr_db_mean_per_image"]
        wandb_utils.log_metrics(run, metrics)

    tasks = list(itertools.product(seeds, rates))
    results = run_tasks(module_function(MODULE, "run_cell"), tasks, module_function(MODULE, "build_context"),
                        payload, args.workers, "evaluating", log_cell)
    rows: List[dict] = [row for cell_rows, _ in results for row in cell_rows]
    per_image = {} if args.no_per_image else {k: v for _, cell in results for k, v in cell.items()}
    if not args.no_per_image:
        per_image["clean_psnr"] = prep["clean_psnr"]

    metadata = evaluation_metadata(args, data, {
        "channel_seeds": list(seeds), "erasure_rates": list(rates),
        "mask_protocol": "one master mask per (batch, seed, rate) shared by all methods; "
                         "smaller schemes use its first columns"})
    outputs = write_sweep_outputs(
        args.output_dir, args.name, rows, metadata, per_image or None, cdf_rate=args.cdf_rate,
        title=f"{args.dataset}/{args.split} erasure sweep")

    print("\n".join(f"- {line}" for line in outputs["insights"]))
    print(f"saved: {outputs['json']}\n       {outputs['markdown']}")
    wandb_utils.log_table(run, "seed_summary", outputs["summary"])
    wandb_utils.log_table(run, "paired_vs_uniform", outputs["paired"])
    wandb_utils.log_images(run, "figures", outputs["figures"])
    wandb_utils.log_artifact(run, f"{args.name}-results", "eval", [outputs["json"], outputs["markdown"]])
    wandb_utils.finish(run)
    return outputs


def main():
    run_and_save(build_parser().parse_args())


if __name__ == "__main__":
    main()
