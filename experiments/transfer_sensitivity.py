"""Step A.1: is a channel's importance a property of the trained filters or of the image content?

The encoder/decoder are **frozen** (never fine-tuned on the new data). For the
same model the mean leave-one-out damage vector is measured on

* a reference set: CIFAR-10 *validation* images (never the test split), and
* each transfer set: CIFAR-100 and/or STL-10 (downsampled to 32x32),

and compared with Spearman / Pearson / Kendall coefficients, the overlap of the
top-4 channels, and bootstrap confidence intervals over images. It also writes
a **native ranking** per transfer set (``ranking_<dataset>.json``, labelled as
derived from that set, so it can only serve as an upper bound) next to the
CIFAR-10 training ranking you pass in. Feed them to the erasure sweep as

    --rankings cifar100=<native.json> --oracle_ranking_file <train ranking.json> \\
    --importance oracle,static:cifar100 --dataset cifar100 --split test

to see whether the transferred ranking delivers the same gain (the roadmap's
"+2.0 to +2.8 dB on STL-10" test).

Decision rule (roadmap): r_s >= 0.85 and top-4 preserved -> hard-code the static
table, side-info = 0 B; r_s <= 0.50 -> sensitivity is content dependent, go to
the online predictor (A.2 / A.3); in between -> report as partial.

    python -m experiments.transfer_sensitivity --checkpoint checkpoints/train_split_adjsccq/best_val.pth \\
        --reference_ranking checkpoints/train_split_adjsccq/oracle_ranking.json \\
        --targets cifar100 --max_images 2000 --output_dir results/phase1/transfer
"""
import argparse
import json
import os

import numpy as np
import torch

from data.datasets import load_image_tensor
from experiments import plotting, wandb_utils
from experiments.model_io import load_model_bundle
from experiments.reproducibility import build_provenance, seed_everything, write_json
from experiments.sensitivity import damage_for_tensor, ranking_from_damage
from experiments.stats_utils import kendall_tau, pearson, spearman


def correlate(reference: np.ndarray, target: np.ndarray, n_boot: int) -> dict:
    """Coefficients of the mean damage vectors, with bootstrap CIs resampling images."""
    ref_mean, tgt_mean = reference.mean(axis=0), target.mean(axis=0)
    # The two image sets are different images, so each is resampled independently.
    rng = np.random.default_rng(0)
    boot = [
        spearman(reference[rng.integers(0, len(reference), len(reference))].mean(axis=0),
                 target[rng.integers(0, len(target), len(target))].mean(axis=0))
        for _ in range(n_boot)
    ]
    s_low, s_high = (float(v) for v in np.nanpercentile(boot, [2.5, 97.5]))
    top4_ref, top4_tgt = set(np.argsort(-ref_mean)[:4].tolist()), set(np.argsort(-tgt_mean)[:4].tolist())
    return {
        "spearman": spearman(ref_mean, tgt_mean), "spearman_ci95": [s_low, s_high],
        "pearson": pearson(ref_mean, tgt_mean), "kendall": kendall_tau(ref_mean, tgt_mean),
        "top4_overlap": len(top4_ref & top4_tgt) / 4,
        "top4_reference": sorted(top4_ref), "top4_target": sorted(top4_tgt),
        "worst_channel_reference": int(np.argmax(ref_mean)), "worst_channel_target": int(np.argmax(tgt_mean)),
    }


def decide(stats: dict) -> str:
    if stats["spearman"] >= 0.85 and stats["top4_overlap"] >= 0.75:
        return "STATIC: ranking transfers; hard-code the table (0 B side information)."
    if stats["spearman"] <= 0.50:
        return "CONTENT-DEPENDENT: ranking does not transfer; use the online predictor (A.2 / A.3)."
    return "PARTIAL: ranking transfers only partly; report both static and predicted rankings."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference_ranking", required=True, help="CIFAR-10 train-split oracle_ranking.json")
    parser.add_argument("--targets", default="cifar100", help="comma list from: cifar100, stl10")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--max_images", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--n_boot", type=int, default=500)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/phase1/transfer")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    args = parser.parse_args()

    if args.max_images < 1000:
        raise SystemExit("--max_images must be >= 1000 (roadmap A.1: fewer images make rankings noisy)")
    seed_everything(args.global_seed)
    bundle = load_model_bundle(args.checkpoint, torch.device("cpu"))
    try:
        from tqdm import tqdm as progress
    except ImportError:  # pragma: no cover
        progress = None

    reference_images = load_image_tensor("cifar10", "val", args.data_dir, args.max_images, bundle.split_seed)
    reference = damage_for_tensor(bundle, reference_images, args.batch_size, progress)["damage"]
    with open(args.reference_ranking) as handle:
        train_ranking = json.load(handle)["ranking_channels_most_to_least_important"]

    os.makedirs(args.output_dir, exist_ok=True)
    results, damages = {}, {"cifar10_val": reference}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        images = load_image_tensor(target, "test", args.data_dir, args.max_images)
        damage = damage_for_tensor(bundle, images, args.batch_size, progress)["damage"]
        damages[target] = damage
        stats = correlate(reference, damage, args.n_boot)
        stats["train_ranking_top4"] = sorted(train_ranking[:4])
        stats["train_ranking_vs_target_spearman"] = spearman(
            [len(train_ranking) - train_ranking.index(c) for c in range(len(train_ranking))], damage.mean(axis=0))
        stats["decision"] = decide(stats)
        stats["clean_damage_mean_db"] = float(damage.mean())
        results[target] = stats
        native = ranking_from_damage(damage)
        write_json(os.path.join(args.output_dir, f"ranking_{target}.json"), {
            "ranking_channels_most_to_least_important": native, "source_split": f"{target}_test_native",
            "psnr_drop_db_by_channel": damage.mean(axis=0).tolist(), "num_images": len(damage),
            "note": "derived from the evaluation set itself: an upper bound, never a deployable ranking"})

    metadata = build_provenance(args=args, checkpoint_path=args.checkpoint, global_seed=args.global_seed,
                                extra_files={"reference_ranking": args.reference_ranking},
                                split_info={"reference": "cifar10/val", "targets": list(results)})
    write_json(os.path.join(args.output_dir, "transfer.json"), {"metadata": metadata, "results": results})
    np.savez_compressed(os.path.join(args.output_dir, "damage_vectors.npz"), **damages)

    figures = []
    channel_labels = [str(c) for c in range(bundle.channels)]
    ref_mean = reference.mean(axis=0)
    for target, stats in results.items():
        tgt_mean = damages[target].mean(axis=0)
        figures += plotting.plot_scatter(
            ref_mean, tgt_mean, os.path.join(args.output_dir, f"scatter_cifar10_vs_{target}"),
            xlabel="CIFAR-10 mean damage (dB)", ylabel=f"{target} mean damage (dB)", annotate=channel_labels,
            alpha=0.9, size=22, diagonal=True,
            title=f"Spearman {stats['spearman']:+.2f}, top-4 overlap {stats['top4_overlap']:.2f}")
        order = np.argsort(-ref_mean)
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(5.2, 3.0))
        width = 0.4
        axis.bar(np.arange(len(order)) - width / 2, ref_mean[order], width, label="CIFAR-10 (val)",
                 color=plotting.OKABE_ITO[0])
        axis.bar(np.arange(len(order)) + width / 2, tgt_mean[order], width, label=target,
                 color=plotting.OKABE_ITO[1])
        axis.set_xticks(range(len(order)), [str(c) for c in order])
        axis.set(xlabel="Channel (sorted by CIFAR-10 damage)", ylabel="Mean PSNR drop (dB)",
                 title="Same channels matter on both datasets?")
        axis.legend()
        figures += plotting.save_figure(fig, os.path.join(args.output_dir, f"bars_cifar10_vs_{target}"))

    with open(os.path.join(args.output_dir, "transfer.md"), "w") as handle:
        handle.write("# Cross-dataset invariance of channel sensitivity (frozen model)\n\n"
                     "| target | Spearman (95% CI) | Pearson | Kendall | top-4 overlap | decision |\n|---|---|---|---|---|---|\n")
        for target, s in results.items():
            handle.write(f"| {target} | {s['spearman']:+.3f} ({s['spearman_ci95'][0]:+.2f}..{s['spearman_ci95'][1]:+.2f}) "
                         f"| {s['pearson']:+.3f} | {s['kendall']:+.3f} | {s['top4_overlap']:.2f} | {s['decision']} |\n")
    for target, s in results.items():
        print(f"{target}: Spearman {s['spearman']:+.3f}, top-4 overlap {s['top4_overlap']:.2f} -> {s['decision']}")

    run = wandb_utils.init_run(args.wandb, args.wandb_project, "transfer-sensitivity", vars(args),
                               job_type="analysis", group="phase1")
    wandb_utils.log_table(run, "transfer", [{"target": t, **{k: v for k, v in s.items()}} for t, s in results.items()])
    wandb_utils.log_images(run, "figures", figures)
    wandb_utils.finish(run)


if __name__ == "__main__":
    main()
