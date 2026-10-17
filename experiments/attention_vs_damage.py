"""Figure 2 of the paper: attention weights do not predict erasure damage.

For N validation images (never the test set) this computes the leave-one-out
PSNR damage of each of the C latent channels and pairs it with the ADJSCC
bottleneck attention gate of the same channel, then measures

* the pooled Pearson / Spearman correlation (the headline rho of the roadmap),
* the **within-image** rank correlation (does attention rank the channels of one
  image correctly?) with a bootstrap confidence interval over images,
* top-4 overlap between the attention ranking and the true ranking,
* the same numbers for the cheap latent statistics (energy / variance / L1)
  for context.

Outputs the left panel (scatter) and right panel (sorted oracle damage bars) as
vector+PNG figures, plus raw arrays for re-plotting.

    python -m experiments.attention_vs_damage --checkpoint checkpoints/train_split_adjsccq/best_val.pth \\
        --split val --max_images 2000 --output_dir results/phase1/attention
"""
import argparse
import os

import numpy as np
import torch

from data.datasets import is_official_test, load_image_tensor
from experiments import plotting, wandb_utils
from experiments.model_io import encode_full, load_model_bundle
from experiments.reproducibility import build_provenance, log_test_access, seed_everything, write_json
from experiments.sensitivity import loo_damage_batch
from experiments.stats_utils import bootstrap_ci, pearson, row_spearman, spearman, topk_overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_images", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/phase1/attention")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    args = parser.parse_args()

    seed_everything(args.global_seed)
    device = torch.device("cpu")
    bundle = load_model_bundle(args.checkpoint, device)
    if is_official_test("cifar10", args.split):
        log_test_access("experiments.attention_vs_damage", args.checkpoint)
    images = load_image_tensor("cifar10", args.split, args.data_dir, args.max_images, bundle.split_seed)

    scores = {"attention": [], "energy": [], "variance": [], "l1": []}
    damages = []
    with torch.no_grad():
        for start in range(0, len(images), args.batch_size):
            x = images[start:start + args.batch_size]
            info = encode_full(bundle.encoder, x, bundle.snr_db)
            latent = bundle.encoder.quantizer.dequantize_indices(info["idx"])
            damage, _ = loo_damage_batch(bundle, latent, x, bundle.snr_db)
            damages.append(damage)
            per_channel = info["z"].reshape(x.shape[0], bundle.channels, -1)
            scores["attention"].append(info["attn"]["af5_bottleneck"])
            scores["energy"].append(per_channel.pow(2).mean(-1))
            scores["variance"].append(per_channel.var(-1, unbiased=False))
            scores["l1"].append(per_channel.abs().mean(-1))
    damage = torch.cat(damages).numpy()
    scores = {k: torch.cat(v).numpy() for k, v in scores.items()}

    stats = {}
    for name, score in scores.items():
        within = row_spearman(score, damage)
        point, low, high = bootstrap_ci(within[np.isfinite(within)], np.mean, n_boot=500, seed=0)
        stats[name] = {
            "pooled_pearson": pearson(score.ravel(), damage.ravel()),
            "pooled_spearman": spearman(score.ravel(), damage.ravel()),
            "within_image_spearman_mean": point,
            "within_image_spearman_ci95": [low, high],
            "top4_overlap_mean": float(topk_overlap(score, damage, 4).mean()),
            "top1_hit_rate": float((score.argmax(1) == damage.argmax(1)).mean()),
        }
    mean_damage = damage.mean(axis=0)
    mean_attention = scores["attention"].mean(axis=0)
    stats["channel_level"] = {
        "spearman_mean_attention_vs_mean_damage": spearman(mean_attention, mean_damage),
        "chance_top4_overlap": 4 / bundle.channels,
    }

    notes = [
        f"Pooled Pearson between attention and damage: {stats['attention']['pooled_pearson']:+.3f} "
        f"({len(images)} images x {bundle.channels} channels).",
        f"Within an image, attention ranks the channels with Spearman "
        f"{stats['attention']['within_image_spearman_mean']:+.3f} "
        f"(95% CI {stats['attention']['within_image_spearman_ci95'][0]:+.3f}..."
        f"{stats['attention']['within_image_spearman_ci95'][1]:+.3f}).",
        f"Attention's top-4 overlaps the true top-4 by {stats['attention']['top4_overlap_mean']:.2f} "
        f"(random guessing: {4 / bundle.channels:.2f}).",
        "Best cheap latent statistic by within-image Spearman: " + max(
            ((k, stats[k]["within_image_spearman_mean"]) for k in ("energy", "variance", "l1")),
            key=lambda kv: kv[1])[0] + ".",
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.output_dir, "attention_vs_damage.npz"), damage=damage, **scores)
    write_json(os.path.join(args.output_dir, "attention_vs_damage.json"), {
        "metadata": build_provenance(args=args, checkpoint_path=args.checkpoint, global_seed=args.global_seed,
                                     device=device, split_info={"dataset": "cifar10", "split": args.split}),
        "stats": stats, "insights": notes})

    figures = plotting.plot_scatter(
        scores["attention"].ravel(), damage.ravel(), os.path.join(args.output_dir, "fig2_left_attention_scatter"),
        xlabel="Attention gate", ylabel="PSNR damage when lost (dB)",
        title=f"Pearson ρ = {stats['attention']['pooled_pearson']:+.3f}", alpha=0.08, size=5)
    order = np.argsort(-mean_damage)
    figures += plotting.plot_bars(mean_damage[order], [str(c) for c in order],
                                  os.path.join(args.output_dir, "fig2_right_oracle_damage"),
                                  ylabel="Mean PSNR drop (dB)", title="True damage per channel (sorted)",
                                  highlight=[0])
    labels = list(stats.keys())[:4]
    figures += plotting.plot_bars([stats[k]["within_image_spearman_mean"] for k in labels], labels,
                                  os.path.join(args.output_dir, "within_image_spearman"),
                                  ylabel="Within-image Spearman with true damage",
                                  errors=[(stats[k]["within_image_spearman_ci95"][1]
                                           - stats[k]["within_image_spearman_ci95"][0]) / 2 for k in labels])
    print("\n".join(f"- {line}" for line in notes))

    run = wandb_utils.init_run(args.wandb, args.wandb_project, "attention-vs-damage", vars(args),
                               job_type="analysis", group="phase1")
    wandb_utils.set_summary(run, {f"attention/{k}": v for k, v in stats["attention"].items()
                                  if not isinstance(v, list)})
    wandb_utils.log_images(run, "figures", figures)
    wandb_utils.finish(run)


if __name__ == "__main__":
    main()
