"""Phase 1a: is unequal latent importance a property of deep JSCC, or one lucky seed?

Several ADJSCC-Q models are trained from different initialisation seeds (same
40k/10k split). For each model this script reads the training-split
leave-one-channel-out damage vector (``oracle_ranking.json``) and tests the
roadmap's claim: *the exact channel indices may permute, but the distribution of
damage is the same*.

Why not just correlate the damage vectors index by index? A convolutional
encoder is symmetric under relabelling its channels, so "channel 3" in one
model has no relation to "channel 3" in another; an index-aligned Spearman
coefficient is expected to be ~0 and says nothing. (The roadmap's
"assert r_s > 0.85" therefore cannot hold across independently trained models.)
The right comparisons are permutation-invariant:

* the **sorted damage profile** (largest to smallest) and its distance between models,
* the **counts per damage band** (catastrophic > 10 dB, high 5-8 dB, low < 2 dB),
* the **concentration**: share of total damage carried by the top-1 / top-4 channels,
* the UEP **gain over uniform** that each model's own oracle achieves (optional,
  from ``eval_rs_uep`` outputs).

The index-aligned matrix is still plotted, as evidence that indices do permute.

Example::

    python -m experiments.multiseed_analysis \\
        --run seed42=checkpoints/train_split_adjsccq --run seed100=results/models/seed100 \\
        --run seed2024=results/models/seed2024 --output_dir results/phase1/multiseed
"""
import argparse
import json
import os
from typing import Dict, List

import numpy as np

from experiments import plotting, wandb_utils
from experiments.reproducibility import build_provenance, write_json
from experiments.stats_utils import pearson, spearman

BANDS = (("catastrophic (>10 dB)", 10.0, np.inf), ("high (5-10 dB)", 5.0, 10.0),
         ("medium (2-5 dB)", 2.0, 5.0), ("low (<2 dB)", -np.inf, 2.0))


def load_run(spec: str):
    name, _, folder = spec.partition("=")
    with open(os.path.join(folder, "oracle_ranking.json")) as handle:
        oracle = json.load(handle)
    metrics_path = os.path.join(folder, "metrics.json")
    clean = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as handle:
            clean = json.load(handle).get("clean_validation", {}).get("psnr_db_mean_per_image")
    return name, folder, oracle, clean


def band_counts(damage: np.ndarray) -> Dict[str, int]:
    return {label: int(((damage >= lo) & (damage < hi)).sum()) for label, lo, hi in BANDS}


def analyse(runs) -> dict:
    names = [name for name, *_ in runs]
    damages = np.array([run[2]["psnr_drop_db_by_channel"] for run in runs], dtype=np.float64)
    rankings = [run[2]["ranking_channels_most_to_least_important"] for run in runs]
    sorted_profiles = -np.sort(-damages, axis=1)
    count = len(runs)

    per_model = []
    for i, name in enumerate(names):
        profile = sorted_profiles[i]
        per_model.append({
            "name": name,
            "clean_val_psnr_db": runs[i][3],
            "top4_channels": rankings[i][:4],
            "worst_channel_damage_db": float(profile[0]),
            "least_critical_damage_db": float(profile[-1]),
            "top1_share_of_total_damage": float(profile[0] / profile.sum()),
            "top4_share_of_total_damage": float(profile[:4].sum() / profile.sum()),
            "bands": band_counts(damages[i]),
        })

    aligned_spearman = np.eye(count)
    profile_pearson = np.eye(count)
    profile_l1 = np.zeros((count, count))
    for i in range(count):
        for j in range(count):
            if i != j:
                aligned_spearman[i, j] = spearman(damages[i], damages[j])
                profile_pearson[i, j] = pearson(sorted_profiles[i], sorted_profiles[j])
            profile_l1[i, j] = np.abs(sorted_profiles[i] - sorted_profiles[j]).mean()

    off = ~np.eye(count, dtype=bool)
    summary = {
        "n_models": count,
        "mean_index_aligned_spearman": float(aligned_spearman[off].mean()) if count > 1 else None,
        "mean_sorted_profile_pearson": float(profile_pearson[off].mean()) if count > 1 else None,
        "mean_sorted_profile_l1_db": float(profile_l1[off].mean()) if count > 1 else None,
        "worst_channel_damage_mean_db": float(sorted_profiles[:, 0].mean()),
        "worst_channel_damage_std_db": float(sorted_profiles[:, 0].std(ddof=1)) if count > 1 else None,
        "every_model_has_a_catastrophic_channel": bool((sorted_profiles[:, 0] > 10).all()),
        "top_channel_indices_differ": len({r[0] for r in rankings}) > 1,
    }
    return {"per_model": per_model, "summary": summary, "damages": damages.tolist(),
            "aligned_spearman": aligned_spearman.tolist(), "profile_pearson": profile_pearson.tolist(),
            "profile_l1": profile_l1.tolist(), "names": names}


def insights(result: dict) -> List[str]:
    s, models = result["summary"], result["per_model"]
    lines = []
    worst = ", ".join(f"{m['name']} {m['worst_channel_damage_db']:.1f} dB (channel {m['top4_channels'][0]})"
                      for m in models)
    lines.append(f"Most damaging channel per model: {worst}.")
    if s["n_models"] > 1:
        lines.append(f"Index-aligned Spearman between models averages {s['mean_index_aligned_spearman']:+.2f}: "
                     + ("channel labels do permute between initialisations, as expected."
                        if abs(s["mean_index_aligned_spearman"]) < 0.5 else
                        "unexpectedly high, which would mean labels are consistent across seeds."))
        lines.append(f"Sorted damage profiles agree: mean Pearson {s['mean_sorted_profile_pearson']:.3f}, "
                     f"mean absolute difference {s['mean_sorted_profile_l1_db']:.2f} dB per channel.")
    lines.append("Every model has a catastrophic (>10 dB) channel." if s["every_model_has_a_catastrophic_channel"]
                 else "NOT every model has a >10 dB channel: the 'one catastrophic channel' claim needs softening.")
    top4 = ", ".join(f"{m['name']} {100 * m['top4_share_of_total_damage']:.0f}%" for m in models)
    lines.append(f"Share of total damage carried by the top-4 channels: {top4}.")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="name=folder (folder has oracle_ranking.json)")
    parser.add_argument("--output_dir", default="results/phase1/multiseed")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    args = parser.parse_args()

    runs = [load_run(spec) for spec in args.run]
    result = analyse(runs)
    notes = insights(result)
    metadata = build_provenance(args=args, extra_files={f"oracle_{n}": os.path.join(f, "oracle_ranking.json")
                                                         for n, f, *_ in runs})
    write_json(os.path.join(args.output_dir, "multiseed.json"),
               {"metadata": metadata, **result, "insights": notes})

    figures = []
    n_channels = len(result["damages"][0])
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(4.4, 3.4))
    for name, damage in zip(result["names"], result["damages"]):
        profile = -np.sort(-np.array(damage))
        axis.plot(range(1, n_channels + 1), profile, label=name, linewidth=1.3, **plotting.style_for(name))
    axis.axhline(10, color="black", linewidth=0.7, linestyle=":")
    axis.set(xlabel="Channel rank (most to least damaging)", ylabel="PSNR drop when lost (dB)",
             title="Sorted damage profile per model")
    axis.legend()
    figures += plotting.save_figure(fig, os.path.join(args.output_dir, "damage_profiles"))

    fig, axes = plt.subplots(1, len(result["names"]), figsize=(2.8 * len(result["names"]), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, name, damage in zip(axes, result["names"], result["damages"]):
        order = np.argsort(-np.array(damage))
        axis.bar(range(n_channels), np.array(damage)[order], color=plotting.OKABE_ITO[0])
        axis.set_xticks(range(n_channels), [str(c) for c in order], fontsize=5, rotation=90)
        axis.set(title=name, xlabel="channel index (sorted)")
    axes[0].set_ylabel("PSNR drop (dB)")
    figures += plotting.save_figure(fig, os.path.join(args.output_dir, "damage_bars"))

    if len(result["names"]) > 1:
        figures += plotting.plot_heatmap(np.array(result["aligned_spearman"]), result["names"], result["names"],
                                         os.path.join(args.output_dir, "aligned_spearman"),
                                         title="Index-aligned Spearman (≈0 → labels permute)",
                                         cbar_label="rho", fmt="{:.2f}", center_zero=True)
        figures += plotting.plot_heatmap(np.array(result["profile_pearson"]), result["names"], result["names"],
                                         os.path.join(args.output_dir, "profile_pearson"),
                                         title="Sorted-profile Pearson", cbar_label="r", fmt="{:.3f}")

    with open(os.path.join(args.output_dir, "multiseed.md"), "w") as handle:
        handle.write("# Multi-seed oracle comparison\n\n| model | clean val PSNR | top-4 channels | worst (dB) | least (dB) | top-4 share | bands |\n"
                     "|---|---|---|---|---|---|---|\n")
        for m in result["per_model"]:
            clean = "n/a" if m["clean_val_psnr_db"] is None else f"{m['clean_val_psnr_db']:.2f}"
            handle.write(f"| {m['name']} | {clean} | {m['top4_channels']} | {m['worst_channel_damage_db']:.2f} | "
                         f"{m['least_critical_damage_db']:.2f} | {100 * m['top4_share_of_total_damage']:.0f}% | "
                         f"{m['bands']} |\n")
        handle.write("\n## Insights\n\n" + "\n".join(f"- {line}" for line in notes) + "\n")
    print("\n".join(f"- {line}" for line in notes))

    run = wandb_utils.init_run(args.wandb, args.wandb_project, "multiseed-analysis", vars(args),
                               job_type="analysis", group="phase1")
    wandb_utils.log_table(run, "per_model", [{k: v for k, v in m.items()} for m in result["per_model"]])
    wandb_utils.log_images(run, "figures", figures)
    wandb_utils.set_summary(run, result["summary"])
    wandb_utils.finish(run)


if __name__ == "__main__":
    main()
