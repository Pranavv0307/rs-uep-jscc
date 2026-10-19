"""Step A.2: can cheap latent statistics rank the channels without any decoding?

Candidates (all computed on the latent *after* power normalisation, no labels):

* ``energy``    mean z^2 per channel
* ``variance``  variance of z per channel
* ``l1``        mean |z| = (1/2) mean |d||z||^2/dz| -- the roadmap's gradient
                saliency in closed form (the gradient of the squared norm is 2z)
* ``taylor``    |sum (dMSE/dz) z| -- first-order estimate of the damage of dropping
                the channel (needs a decoder copy and the image at the transmitter)
* ``attention`` ADJSCC gate (reference for the "attention failed" result)

Each is compared with the true per-image leave-one-out damage by the
within-image Spearman correlation, top-1 hit rate and top-4 overlap (with
bootstrap CIs over images), and ``damage-agnostic`` baselines are added: a random
ranking and the static train-split ranking. Whether a proxy is *good enough* is
decided by the erasure sweep, e.g.

    python -m experiments.eval_rs_uep --checkpoint <ckpt> --split val --erasure_rates 0.4 \\
        --importance energy,variance,l1,taylor,attention,oracle,oracle_perimage \\
        --oracle_ranking_file <train ranking> --include_random_control \\
        --schemes three_tier_25_50_25 --output_dir results/phase1/saliency_sweep

Good result (roadmap): a proxy reaches >= 80% of the oracle gain; bad: within-image
Spearman < 0.30, i.e. no better than random, so go to A.3.

    python -m experiments.feature_saliency --checkpoint <ckpt> --reference_ranking <ranking.json> \\
        --max_images 2000 --output_dir results/phase1/saliency
"""
import argparse
import json
import os

import numpy as np
import torch

from data.datasets import is_official_test, load_image_tensor
from experiments import plotting, wandb_utils
from experiments.importance_sources import taylor_scores
from experiments.model_io import encode_full, load_model_bundle
from experiments.reproducibility import build_provenance, log_test_access, seed_everything, write_json
from experiments.sensitivity import loo_damage_batch
from experiments.stats_utils import bootstrap_ci, row_spearman, topk_overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference_ranking", required=True, help="train-split oracle_ranking.json")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_images", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/phase1/saliency")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    args = parser.parse_args()

    seed_everything(args.global_seed)
    bundle = load_model_bundle(args.checkpoint, torch.device("cpu"))
    if is_official_test("cifar10", args.split):
        log_test_access("experiments.feature_saliency", args.checkpoint)
    images = load_image_tensor("cifar10", args.split, args.data_dir, args.max_images, bundle.split_seed)
    with open(args.reference_ranking, encoding="utf-8") as handle:
        static = json.load(handle)["ranking_channels_most_to_least_important"]
    static_scores = np.zeros(bundle.channels)
    static_scores[static] = np.arange(bundle.channels, 0, -1)

    names = ("attention", "energy", "variance", "l1", "taylor")
    scores = {name: [] for name in names}
    damages = []
    for start in range(0, len(images), args.batch_size):
        x = images[start:start + args.batch_size]
        with torch.no_grad():
            info = encode_full(bundle.encoder, x, bundle.snr_db)
            latent = bundle.encoder.quantizer.dequantize_indices(info["idx"])
            damage, _ = loo_damage_batch(bundle, latent, x, bundle.snr_db)
        per_channel = info["z"].reshape(x.shape[0], bundle.channels, -1)
        scores["attention"].append(info["attn"]["af5_bottleneck"])
        scores["energy"].append(per_channel.pow(2).mean(-1))
        scores["variance"].append(per_channel.var(-1, unbiased=False))
        scores["l1"].append(per_channel.abs().mean(-1))
        scores["taylor"].append(taylor_scores(bundle, info["z_q"], x, bundle.snr_db))
        damages.append(damage)
    damage = torch.cat(damages).numpy()
    scores = {k: torch.cat(v).numpy() for k, v in scores.items()}
    rng = np.random.default_rng(args.global_seed)
    scores["random"] = rng.random(damage.shape)
    scores["static_train_ranking"] = np.tile(static_scores, (len(damage), 1))

    stats = {}
    for name, score in scores.items():
        within = row_spearman(score + 1e-9 * rng.random(score.shape), damage)  # tiny jitter breaks exact ties
        within = within[np.isfinite(within)]
        mean, low, high = bootstrap_ci(within, np.mean, n_boot=500, seed=1)
        stats[name] = {
            "within_image_spearman": mean, "ci95": [low, high],
            "top1_hit_rate": float((score.argmax(1) == damage.argmax(1)).mean()),
            "top4_overlap": float(topk_overlap(score, damage, 4).mean()),
            "verdict": "useful" if mean >= 0.5 else ("weak" if mean >= 0.3 else "no better than chance"),
        }

    ranking = sorted(((k, v["within_image_spearman"]) for k, v in stats.items()), key=lambda kv: -kv[1])
    notes = ["Proxies ranked by within-image Spearman with the true damage: "
             + ", ".join(f"{k} {v:+.2f}" for k, v in ranking) + ".",
             f"Static train ranking reaches {stats['static_train_ranking']['within_image_spearman']:+.2f}: "
             "how much of the per-image structure a single global table already explains.",
             f"Best proxy beyond the static table: {max((k for k in names), key=lambda k: stats[k]['within_image_spearman'])} "
             f"(top-4 overlap {max(stats[k]['top4_overlap'] for k in names):.2f} vs chance 0.25)."]

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.output_dir, "saliency_scores.npz"), damage=damage, **scores)
    write_json(os.path.join(args.output_dir, "saliency.json"), {
        "metadata": build_provenance(args=args, checkpoint_path=args.checkpoint, global_seed=args.global_seed,
                                     extra_files={"reference_ranking": args.reference_ranking},
                                     split_info={"dataset": "cifar10", "split": args.split}),
        "stats": stats, "insights": notes})

    labels = list(stats)
    figures = plotting.plot_bars(
        [stats[k]["within_image_spearman"] for k in labels], labels,
        os.path.join(args.output_dir, "within_image_spearman"), ylabel="Within-image Spearman with true damage",
        errors=[(stats[k]["ci95"][1] - stats[k]["ci95"][0]) / 2 for k in labels],
        highlight=[labels.index("random")], figsize=(6.0, 3.2))
    figures += plotting.plot_bars([stats[k]["top4_overlap"] for k in labels], labels,
                                  os.path.join(args.output_dir, "top4_overlap"),
                                  ylabel="Top-4 overlap with true top-4", highlight=[labels.index("random")],
                                  figsize=(6.0, 3.2))
    with open(os.path.join(args.output_dir, "saliency.md"), "w", encoding="utf-8") as handle:
        handle.write("# Feature-saliency proxies vs true leave-one-out damage\n\n"
                     "| proxy | within-image Spearman (95% CI) | top-1 hit | top-4 overlap | verdict |\n|---|---|---|---|---|\n")
        for k in labels:
            s = stats[k]
            handle.write(f"| {k} | {s['within_image_spearman']:+.3f} ({s['ci95'][0]:+.2f}..{s['ci95'][1]:+.2f}) | "
                         f"{s['top1_hit_rate']:.2f} | {s['top4_overlap']:.2f} | {s['verdict']} |\n")
        handle.write("\n## Insights\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    print("\n".join(f"- {n}" for n in notes))

    run = wandb_utils.init_run(args.wandb, args.wandb_project, "feature-saliency", vars(args),
                               job_type="analysis", group="phase1")
    wandb_utils.log_table(run, "proxy_quality", [{"proxy": k, **{a: b for a, b in v.items() if a != "ci95"}}
                                                 for k, v in stats.items()])
    wandb_utils.log_images(run, "figures", figures)
    wandb_utils.finish(run)


if __name__ == "__main__":
    main()
