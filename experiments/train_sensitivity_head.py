"""Step A.3: a tiny online head that predicts per-channel damage from the latent.

Two stages, both deterministic:

``labels``  For every image of the 40k training split (and the 10k validation
            split) compute the ground-truth leave-one-out damage vector
            ``d in R^C`` through the quantised path, together with the latent
            features the head will see (per-channel mean and variance of z) and
            the attention gate. Cached to ``<labels_dir>/labels_<split>.pt``.
``train``   Train ``SensitivityHead`` on the cached training features with
            ``SmoothL1(d_hat, d) + lambda * PairwiseHinge`` (margin 0.5,
            lambda 0.1), Adam 1e-3, batch 128, 30 epochs, for several head
            seeds. The encoder is frozen by construction (features are
            pre-computed from it). The best epoch per seed is chosen on the
            **validation** split; the test split is only touched with
            ``--eval_test`` (logged).

The head is judged against honest baselines on the same images: a random
ranking, the static train-split ranking, and the closed-form proxies
(energy / variance from the pooled features, attention). Metrics: top-4 overlap,
top-1 hit rate, within-image Spearman, R^2, and *protection regret* (dB of true
damage in the true top-4 that the predicted top-4 fails to cover).

    python -m experiments.train_sensitivity_head --checkpoint checkpoints/train_split_adjsccq/best_val.pth \\
        --reference_ranking checkpoints/train_split_adjsccq/oracle_ranking.json \\
        --labels_dir results/phase1/head_labels --output_dir results/phase1/head --seeds 0,1,2,3,4
"""
import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from data.datasets import is_official_test, load_image_tensor
from experiments import plotting, wandb_utils
from experiments.model_io import encode_full, load_model_bundle
from experiments.reproducibility import (
    build_provenance,
    file_sha256,
    log_test_access,
    seed_everything,
    write_json,
)
from experiments.sensitivity import count_macs, loo_damage_batch, pairwise_ranking_loss
from experiments.stats_utils import bootstrap_ci, r_squared, row_spearman, seed_statistics, topk_overlap
from models.adjsccq import SensitivityHead


# --------------------------------------------------------------------------
# Stage 1: labels
# --------------------------------------------------------------------------
@torch.no_grad()
def compute_labels(bundle, images: torch.Tensor, batch_size: int, progress=None) -> Dict[str, torch.Tensor]:
    snr_db = bundle.snr_db
    parts = {"damage": [], "mean": [], "var": [], "attention": [], "clean_psnr": []}
    starts = range(0, len(images), batch_size)
    for start in (progress(starts) if progress else starts):
        x = images[start:start + batch_size]
        info = encode_full(bundle.encoder, x, snr_db)
        latent = bundle.encoder.quantizer.dequantize_indices(info["idx"])
        damage, clean = loo_damage_batch(bundle, latent, x, snr_db)
        per_channel = info["z"].reshape(x.shape[0], bundle.channels, -1)
        parts["damage"].append(damage)
        parts["mean"].append(per_channel.mean(-1))
        parts["var"].append(per_channel.var(-1, unbiased=False))
        parts["attention"].append(info["attn"]["af5_bottleneck"])
        parts["clean_psnr"].append(clean)
    return {k: torch.cat(v).cpu() for k, v in parts.items()}


def get_labels(bundle, args, split: str, progress) -> Dict[str, torch.Tensor]:
    path = os.path.join(args.labels_dir, f"labels_{split}.pt")
    if os.path.exists(path) and not args.recompute:
        print(f"using cached labels {path}")
        return torch.load(path, weights_only=True)["labels"]
    if is_official_test("cifar10", split):
        log_test_access("experiments.train_sensitivity_head", args.checkpoint, note="evaluating the sensitivity head")
    images = load_image_tensor("cifar10", split, args.data_dir, args.max_images, bundle.split_seed)
    labels = compute_labels(bundle, images, args.batch_size, progress)
    os.makedirs(args.labels_dir, exist_ok=True)
    torch.save({"labels": labels, "checkpoint": args.checkpoint, "checkpoint_sha256": file_sha256(args.checkpoint),
                "split": split, "num_images": len(images)}, path)
    return labels


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def ranking_metrics(scores: np.ndarray, damage: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    within = row_spearman(scores + 1e-9 * np.random.default_rng(0).random(scores.shape), damage)
    top4_pred = np.argsort(-scores, axis=1, kind="stable")[:, :4]
    top4_true = np.argsort(-damage, axis=1, kind="stable")[:, :4]
    covered_pred = np.take_along_axis(damage, top4_pred, axis=1).sum(1)
    covered_true = np.take_along_axis(damage, top4_true, axis=1).sum(1)
    return {
        "top4_overlap": float(topk_overlap(scores, damage, 4).mean()),
        "top1_hit_rate": float((scores.argmax(1) == damage.argmax(1)).mean()),
        "within_image_spearman": float(np.nanmean(within)),
        "protection_regret_db": float((covered_true - covered_pred).mean()),
    }


def baseline_scores(labels: Dict[str, torch.Tensor], static_ranking: List[int], seed: int) -> Dict[str, np.ndarray]:
    mean, var = labels["mean"].numpy(), labels["var"].numpy()
    n, channels = mean.shape
    static = np.zeros(channels)
    static[static_ranking] = np.arange(channels, 0, -1)
    return {
        "random": np.random.default_rng(seed).random((n, channels)),
        "static_train_ranking": np.tile(static, (n, 1)),
        "energy": var + mean ** 2,
        "variance": var,
        "attention": labels["attention"].numpy(),
    }


# --------------------------------------------------------------------------
# Stage 2: training
# --------------------------------------------------------------------------
def build_features(labels: Dict[str, torch.Tensor], use_variance: bool) -> torch.Tensor:
    return torch.cat([labels["mean"], labels["var"]], dim=1) if use_variance else labels["mean"]


def train_head(train_x, train_y, val_x, val_y, args, seed: int, channels: int, use_variance: bool, run=None):
    seed_everything(seed)
    head = SensitivityHead(channels, args.hidden_dim, use_variance)
    head.feat_mean.copy_(train_x.mean(0))
    head.feat_std.copy_(train_x.std(0).clamp_min(1e-6))
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(seed)
    history, best = [], {"top4_overlap": -1.0, "state": None, "epoch": 0}
    val_np = val_y.numpy()
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(len(train_x), generator=generator)
        total, batches = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            predicted = head.net(head.normalise(train_x[index]))
            loss = F.smooth_l1_loss(predicted, train_y[index]) + args.rank_weight * pairwise_ranking_loss(
                predicted, train_y[index], args.margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        head.eval()
        with torch.no_grad():
            val_pred = head.net(head.normalise(val_x)).numpy()
        metrics = ranking_metrics(val_pred, val_np)
        metrics.update(epoch=epoch, train_loss=total / batches,
                       val_smooth_l1=float(F.smooth_l1_loss(torch.from_numpy(val_pred), val_y)),
                       val_r2=r_squared(val_pred, val_np))
        history.append(metrics)
        if run is not None and seed == args.log_seed:
            wandb_utils.log_metrics(run, {f"head/{k}": v for k, v in metrics.items() if k != "epoch"})
        if metrics["top4_overlap"] > best["top4_overlap"]:
            best = {"top4_overlap": metrics["top4_overlap"], "epoch": epoch,
                    "state": {k: v.clone() for k, v in head.state_dict().items()}}
    head.load_state_dict(best["state"])
    return head.eval(), history, best["epoch"]


def save_head(path: str, head: SensitivityHead, extra: dict) -> None:
    torch.save({"state_dict": head.state_dict(), "in_channels": head.in_channels,
                "hidden_dim": head.net[0].out_features, "use_variance": head.use_variance, **extra}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference_ranking", required=True, help="train-split oracle_ranking.json (baseline)")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--labels_dir", default="results/phase1/head_labels")
    parser.add_argument("--output_dir", default="results/phase1/head")
    parser.add_argument("--stage", default="all", choices=["all", "labels", "train"])
    parser.add_argument("--seeds", default="0,1,2,3,4", help="head-training seeds")
    parser.add_argument("--use_variance", action="store_true", help="features = [mean, variance] (roadmap fix)")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--label_batch_size", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank_weight", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--max_images", type=int, default=None, help="smoke tests only")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--eval_test", action="store_true", help="also evaluate on the official test split (logged)")
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    args = parser.parse_args()
    args.log_seed = int(args.seeds.split(",")[0])
    args.batch_size_labels = args.label_batch_size

    seed_everything(args.global_seed)
    bundle = load_model_bundle(args.checkpoint, torch.device("cpu"))
    try:
        from tqdm import tqdm as progress
    except ImportError:  # pragma: no cover
        progress = None
    label_args = argparse.Namespace(**{**vars(args), "batch_size": args.label_batch_size})

    train_labels = get_labels(bundle, label_args, "train", progress)
    val_labels = get_labels(bundle, label_args, "val", progress)
    if args.stage == "labels":
        print("labels written; run again with --stage train")
        return

    with open(args.reference_ranking, encoding="utf-8") as handle:
        static_ranking = json.load(handle)["ranking_channels_most_to_least_important"]
    channels = bundle.channels
    train_x, val_x = build_features(train_labels, args.use_variance), build_features(val_labels, args.use_variance)
    train_y, val_y = train_labels["damage"], val_labels["damage"]

    run = wandb_utils.init_run(args.wandb, args.wandb_project, "sensitivity-head", vars(args), job_type="train",
                               group="phase1")
    os.makedirs(args.output_dir, exist_ok=True)

    per_seed, histories, heads = [], {}, {}
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        head, history, best_epoch = train_head(train_x, train_y, val_x, val_y, args, seed, channels,
                                               args.use_variance, run)
        with torch.no_grad():
            val_pred = head.net(head.normalise(val_x)).numpy()
        metrics = ranking_metrics(val_pred, val_y.numpy())
        metrics.update(seed=seed, best_epoch=best_epoch, val_r2=r_squared(val_pred, val_y.numpy()))
        per_seed.append(metrics)
        histories[seed], heads[seed] = history, head
        save_head(os.path.join(args.output_dir, f"head_seed{seed}.pt"), head,
                  {"seed": seed, "val_metrics": metrics, "labels_checkpoint": args.checkpoint})
        print(f"seed {seed}: top-4 overlap {metrics['top4_overlap']:.3f}, Spearman "
              f"{metrics['within_image_spearman']:+.3f}, regret {metrics['protection_regret_db']:.2f} dB")

    best_seed = max(per_seed, key=lambda m: m["top4_overlap"])["seed"]
    save_head(os.path.join(args.output_dir, "sensitivity_head.pt"), heads[best_seed],
              {"seed": best_seed, "selected_on": "validation top-4 overlap",
               "labels_checkpoint": args.checkpoint, "labels_checkpoint_sha256": file_sha256(args.checkpoint)})

    seed_summary = {key: seed_statistics([m[key] for m in per_seed])
                    for key in ("top4_overlap", "top1_hit_rate", "within_image_spearman", "protection_regret_db", "val_r2")}
    val_np = val_y.numpy()
    comparison = {"head (mean over seeds)": {k: seed_summary[k]["mean"] for k in
                                             ("top4_overlap", "top1_hit_rate", "within_image_spearman",
                                              "protection_regret_db")}}
    for name, scores in baseline_scores(val_labels, static_ranking, args.global_seed).items():
        comparison[name] = ranking_metrics(scores, val_np)

    # Cost check: the head must be a negligible part of the encoder (roadmap: < 2% of FLOPs).
    dummy = torch.zeros(1, 3, bundle.image_size, bundle.image_size)
    encoder_macs = count_macs(bundle.encoder.encoder, lambda: bundle.encoder.encoder(dummy, bundle.snr_db))
    head_macs = heads[best_seed].macs()
    cost = {"encoder_macs": encoder_macs, "head_macs": head_macs, "head_fraction_of_encoder": head_macs / encoder_macs,
            "head_parameters": sum(p.numel() for p in heads[best_seed].parameters())}
    if cost["head_fraction_of_encoder"] > 0.02:
        print(f"WARNING: head is {100 * cost['head_fraction_of_encoder']:.1f}% of encoder MACs (> 2%)")

    test_block = None
    if args.eval_test:
        test_labels = get_labels(bundle, label_args, "test", progress)
        with torch.no_grad():
            test_pred = heads[best_seed].net(heads[best_seed].normalise(
                build_features(test_labels, args.use_variance))).numpy()
        test_np = test_labels["damage"].numpy()
        test_block = {"head": ranking_metrics(test_pred, test_np)}
        for name, scores in baseline_scores(test_labels, static_ranking, args.global_seed).items():
            test_block[name] = ranking_metrics(scores, test_np)

    notes = [
        f"Head top-4 overlap on validation: {seed_summary['top4_overlap']['mean']:.3f} ± "
        f"{seed_summary['top4_overlap']['std'] or 0:.3f} over {len(per_seed)} head seeds "
        f"(random {comparison['random']['top4_overlap']:.2f}, static table "
        f"{comparison['static_train_ranking']['top4_overlap']:.2f}).",
        f"Protection regret: head {seed_summary['protection_regret_db']['mean']:.2f} dB vs static table "
        f"{comparison['static_train_ranking']['protection_regret_db']:.2f} dB vs random "
        f"{comparison['random']['protection_regret_db']:.2f} dB per image.",
        f"The head costs {head_macs} MACs = {100 * cost['head_fraction_of_encoder']:.3f}% of the encoder.",
    ]
    gain_over_static = (comparison["head (mean over seeds)"]["top4_overlap"]
                        - comparison["static_train_ranking"]["top4_overlap"])
    notes.append(("The head beats the static table" if gain_over_static > 0.02 else
                  "The head does NOT clearly beat the static table")
                  + f" on top-4 overlap ({gain_over_static:+.3f}); if images share one ranking the online "
                    "predictor is unnecessary.")
    write_json(os.path.join(args.output_dir, "head_metrics.json"), {
        "metadata": build_provenance(args=args, checkpoint_path=args.checkpoint, global_seed=args.global_seed,
                                     extra_files={"reference_ranking": args.reference_ranking},
                                     split_info={"train_images": len(train_y), "val_images": len(val_y)}),
        "per_seed": per_seed, "seed_summary": seed_summary, "validation_comparison": comparison,
        "test_comparison": test_block, "cost": cost, "best_seed": best_seed, "histories": histories,
        "insights": notes})

    figures = []
    figures += plotting.plot_lines(
        {f"seed {s}": [m["train_loss"] for m in h] for s, h in histories.items()}, list(range(1, args.epochs + 1)),
        os.path.join(args.output_dir, "training_loss"), xlabel="Epoch", ylabel="Train loss", log_y=True)
    figures += plotting.plot_lines(
        {f"seed {s}": [m["top4_overlap"] for m in h] for s, h in histories.items()}, list(range(1, args.epochs + 1)),
        os.path.join(args.output_dir, "val_top4_overlap"), xlabel="Epoch", ylabel="Validation top-4 overlap")
    labels_for_bar = list(comparison)
    figures += plotting.plot_bars([comparison[k]["top4_overlap"] for k in labels_for_bar], labels_for_bar,
                                  os.path.join(args.output_dir, "top4_overlap_comparison"),
                                  ylabel="Top-4 overlap (validation)", highlight=[0], figsize=(6.4, 3.2))
    figures += plotting.plot_bars([comparison[k]["protection_regret_db"] for k in labels_for_bar], labels_for_bar,
                                  os.path.join(args.output_dir, "protection_regret_comparison"),
                                  ylabel="Protection regret (dB, lower is better)", highlight=[0], figsize=(6.4, 3.2))
    with torch.no_grad():
        pred = heads[best_seed].net(heads[best_seed].normalise(val_x)).numpy()
    figures += plotting.plot_scatter(val_np.ravel(), pred.ravel(), os.path.join(args.output_dir, "predicted_vs_true"),
                                     xlabel="True damage (dB)", ylabel="Predicted damage (dB)",
                                     title=f"Validation, R² = {r_squared(pred, val_np):.3f}", alpha=0.05, size=4,
                                     diagonal=True)
    true_freq = np.zeros(channels)
    pred_freq = np.zeros(channels)
    for row_t, row_p in zip(np.argsort(-val_np, axis=1)[:, :4], np.argsort(-pred, axis=1)[:, :4]):
        true_freq[row_t] += 1
        pred_freq[row_p] += 1
    figures += plotting.plot_heatmap(np.stack([true_freq, pred_freq]) / len(val_np), ["true top-4", "head top-4"],
                                     [str(c) for c in range(channels)],
                                     os.path.join(args.output_dir, "top4_selection_frequency"),
                                     title="How often each channel is in the top-4", cbar_label="fraction of images",
                                     fmt="{:.2f}")
    with open(os.path.join(args.output_dir, "head_metrics.md"), "w", encoding="utf-8") as handle:
        handle.write("# Sensitivity head (validation split)\n\n| ranking | top-4 overlap | top-1 hit | within-image Spearman | regret (dB) |\n|---|---|---|---|---|\n")
        for name, m in comparison.items():
            handle.write(f"| {name} | {m['top4_overlap']:.3f} | {m['top1_hit_rate']:.3f} | "
                         f"{m['within_image_spearman']:+.3f} | {m['protection_regret_db']:.2f} |\n")
        handle.write("\n## Insights\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    print("\n".join(f"- {n}" for n in notes))

    wandb_utils.log_table(run, "per_seed", per_seed)
    wandb_utils.log_table(run, "comparison", [{"ranking": k, **v} for k, v in comparison.items()])
    wandb_utils.log_images(run, "figures", figures)
    wandb_utils.set_summary(run, {f"head/{k}_mean": v["mean"] for k, v in seed_summary.items()})
    wandb_utils.log_artifact(run, "sensitivity-head", "model", [os.path.join(args.output_dir, "sensitivity_head.pt")])
    wandb_utils.finish(run)


if __name__ == "__main__":
    main()
