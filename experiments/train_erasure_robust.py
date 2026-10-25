"""Track B: teach the decoder to cope with erased latents (removes the low-loss penalty).

The base model was trained on an identity channel, so a zero-filled gap is an
out-of-distribution shock. This script fine-tunes it with random erasures in
the loop:

* ``--drop_unit channel``  whole channels (64 symbols) vanish -- roadmap B.1;
  ``packet`` drops 16-symbol packets, which is what a failed RS block actually
  produces; ``mixed`` picks one per batch. Compare them (ablation).
* erasures are applied in latent space with **no re-scaling** (the receiver gets
  surviving packets unchanged; this is why ``nn.Dropout`` is not used).
* a per-batch rate ``p ~ U(0, p_max)`` (``--p_mode sample`` draws it per image),
  optional ``--curriculum_epochs`` ramping ``p_max`` from 0, and a fraction of
  clean batches (``--clean_fraction``) to protect the clean PSNR.
* schedule: encoder frozen for ``--freeze_epochs`` then jointly fine-tuned.
* ``--inpaint`` (Step B.2): adds a masked-latent transformer between the RS
  decoder and the image decoder (``models/inpainting.py``).

Checkpoint selection uses a **validation** robustness score: the mean PSNR over a
fixed grid of erasure rates, averaged over channel- and packet-level drops. The
oracle ranking is recomputed for the fine-tuned model on the 40k training split
(the encoder may have moved), and everything is saved next to the checkpoint.

    python -m experiments.train_erasure_robust --init_checkpoint checkpoints/train_split_adjsccq/best_val.pth \\
        --output_dir results/phase2/robust_channel --drop_unit channel --p_max 0.5
"""
import argparse
import copy
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from data.datasets import load_image_tensor
from data.splits import make_loaders
from experiments import plotting, wandb_utils
from experiments.metrics import psnr_per_image, ssim
from experiments.model_io import load_model_bundle
from experiments.reproducibility import build_provenance, file_sha256, seed_everything, write_json
from experiments.sensitivity import damage_for_tensor, ranking_from_damage
from models.inpainting import LatentInpainter

UNITS = ("channel", "packet")
EVAL_GRID = (0.0, 0.1, 0.2, 0.3, 0.4)


def make_drop_mask(batch: int, channels: int, symbols_per_channel: int, packet_size: int, p: float, unit: str,
                   generator: torch.Generator, device) -> torch.Tensor:
    """``(B, C*D)`` bool, ``True`` where the symbol is lost. ``p`` is a float or a ``(B,)`` tensor."""
    p = torch.as_tensor(p, dtype=torch.float32).reshape(-1, 1) if not isinstance(p, float) else p
    if unit == "channel":
        lost = torch.rand(batch, channels, generator=generator) < p
        lost = lost.unsqueeze(-1).expand(batch, channels, symbols_per_channel).reshape(batch, -1)
    elif unit == "packet":
        packets = channels * (symbols_per_channel // packet_size)
        lost = (torch.rand(batch, packets, generator=generator) < p).repeat_interleave(packet_size, dim=1)
    else:
        raise ValueError(f"unknown drop unit {unit!r}")
    return lost.to(device)


def reconstruct_masked(decoder, inpainter, z_q: torch.Tensor, failed: torch.Tensor, snr_db: float) -> torch.Tensor:
    latent = z_q.masked_fill(failed, 0.0)  # fallback value 0.0, never rescaled
    if inpainter is not None:
        latent = inpainter(latent, failed)
    return decoder(latent, snr_db)


@torch.no_grad()
def evaluate_robust(encoder, decoder, inpainter, images: torch.Tensor, channels: int, spc: int, packet_size: int,
                    snr_db: float, device, batch_size: int = 250) -> Dict[str, Dict[float, float]]:
    """Validation PSNR vs erasure rate for each drop unit, with a fixed mask generator."""
    encoder.eval()
    decoder.eval()
    if inpainter is not None:
        inpainter.eval()
    out: Dict[str, Dict[float, float]] = {unit: {p: 0.0 for p in EVAL_GRID} for unit in UNITS}
    for unit in UNITS:
        for p in EVAL_GRID:
            generator = torch.Generator().manual_seed(1234)
            total, count = 0.0, 0
            for start in range(0, len(images), batch_size):
                x = images[start:start + batch_size].to(device)
                z_q = encoder(x, snr_db)
                failed = make_drop_mask(x.shape[0], channels, spc, packet_size, float(p), unit, generator, device)
                recon = reconstruct_masked(decoder, inpainter, z_q, failed, snr_db)
                total += psnr_per_image(recon, x).sum().item()
                count += x.shape[0]
            out[unit][p] = total / count
    return out


def robust_score(curves: Dict[str, Dict[float, float]]) -> float:
    return float(np.mean([v for unit in curves.values() for v in unit.values()]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--drop_unit", default="channel", choices=["channel", "packet", "mixed"])
    parser.add_argument("--p_max", type=float, default=0.5)
    parser.add_argument("--p_mode", default="batch", choices=["batch", "sample"])
    parser.add_argument("--curriculum_epochs", type=int, default=0, help="ramp p_max from 0 over this many epochs")
    parser.add_argument("--clean_fraction", type=float, default=0.1)
    parser.add_argument("--inpaint", action="store_true", help="Step B.2: add the masked-latent transformer")
    parser.add_argument("--inpaint_layers", type=int, default=2)
    parser.add_argument("--inpaint_heads", type=int, default=4)
    parser.add_argument("--packet_size", type=int, default=16)
    parser.add_argument("--val_images", type=int, default=2000, help="validation images used each epoch")
    parser.add_argument("--max_batches", type=int, default=None, help="limit batches per epoch (smoke tests only)")
    parser.add_argument("--oracle_max_images", type=int, default=None, help="default: all 40k training images")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    parser.add_argument("--wandb_name", default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    bundle = load_model_bundle(args.init_checkpoint, device)
    cfg = copy.deepcopy(bundle.config)
    encoder, decoder = bundle.encoder, bundle.decoder
    channels, spc, snr_db = bundle.channels, bundle.symbols_per_channel, bundle.snr_db
    split_seed = bundle.split_seed
    encoder.quantizer.set_sigma(cfg["quantizer"]["sigma_end"])

    inpainter = None
    if args.inpaint:
        inpainter = LatentInpainter(channels, spc, args.inpaint_heads, args.inpaint_layers).to(device)
    train_loader, val_loader, _ = make_loaders(args.data_dir, args.batch_size, args.num_workers, split_seed,
                                               10_000, loader_seed=args.seed + 1)
    val_images = load_image_tensor("cifar10", "val", args.data_dir, args.val_images, split_seed)

    run = wandb_utils.init_run(args.wandb, args.wandb_project, args.wandb_name or os.path.basename(args.output_dir),
                               vars(args), tags=["erasure-robust", args.drop_unit] + (["inpaint"] if args.inpaint else []),
                               group="phase2", job_type="train")

    decoder_params = list(decoder.parameters()) + (list(inpainter.parameters()) if inpainter else [])
    optimizer = torch.optim.Adam(decoder_params, lr=args.lr)
    criterion = nn.MSELoss(reduction="none")
    generator = torch.Generator().manual_seed(args.seed + 7)
    unit_rng = np.random.default_rng(args.seed + 11)

    baseline_curves = evaluate_robust(encoder, decoder, inpainter, val_images, channels, spc, args.packet_size,
                                      snr_db, device)
    history = [{"epoch": 0, "score": robust_score(baseline_curves), "curves": baseline_curves}]
    best = {"score": -1.0, "epoch": 0, "state": None}

    def snapshot(epoch, score):
        state = {"encoder": copy.deepcopy(encoder.state_dict()), "decoder": copy.deepcopy(decoder.state_dict())}
        if inpainter is not None:
            state["inpainter"] = copy.deepcopy(inpainter.state_dict())
        return {"score": score, "epoch": epoch, "state": state}

    best = snapshot(0, history[0]["score"]) if not args.inpaint else {"score": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        for parameter in encoder.parameters():
            parameter.requires_grad = not frozen
        if not frozen and not any(g.get("tag") == "encoder" for g in optimizer.param_groups):
            optimizer.add_param_group({"params": list(encoder.parameters()), "lr": args.lr, "tag": "encoder"})
        p_cap = args.p_max * (min(1.0, epoch / args.curriculum_epochs) if args.curriculum_epochs else 1.0)
        encoder.train()
        decoder.train()
        if inpainter is not None:
            inpainter.train()
        total, batches = 0.0, 0
        for batch_number, (x, _) in enumerate(train_loader):
            if args.max_batches is not None and batch_number >= args.max_batches:
                break
            x = x.to(device)
            with torch.set_grad_enabled(not frozen):
                z_q = encoder(x, snr_db)
            clean_batch = float(torch.rand(1, generator=generator)) < args.clean_fraction
            if args.p_mode == "batch":
                p = 0.0 if clean_batch else float(torch.rand(1, generator=generator)) * p_cap
            else:
                p = torch.rand(x.shape[0], generator=generator) * p_cap
                if clean_batch:
                    p = torch.zeros_like(p)
            unit = args.drop_unit if args.drop_unit != "mixed" else UNITS[int(unit_rng.integers(0, 2))]
            failed = make_drop_mask(x.shape[0], channels, spc, args.packet_size, p, unit, generator, device)
            recon = reconstruct_masked(decoder, inpainter, z_q, failed, snr_db)
            loss = criterion(recon, x).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1

        curves = evaluate_robust(encoder, decoder, inpainter, val_images, channels, spc, args.packet_size,
                                 snr_db, device)
        score = robust_score(curves)
        history.append({"epoch": epoch, "score": score, "train_loss": total / batches, "curves": curves,
                        "frozen_encoder": frozen, "p_cap": p_cap})
        metrics = {"epoch": epoch, "train/mse": total / batches, "val/robust_score_db": score,
                   "val/clean_psnr_db": curves["channel"][0.0], "train/p_cap": p_cap}
        for unit_name in UNITS:
            for rate, value in curves[unit_name].items():
                metrics[f"val/{unit_name}_drop_{rate:g}"] = value
        wandb_utils.log_metrics(run, metrics)
        print(f"epoch {epoch:2d} | frozen={frozen} | train mse {total / batches:.5f} | clean {curves['channel'][0.0]:.2f} dB "
              f"| ch@0.4 {curves['channel'][0.4]:.2f} | pkt@0.4 {curves['packet'][0.4]:.2f} | score {score:.2f}")
        if score > best["score"]:
            best = snapshot(epoch, score)

    final_state = snapshot(args.epochs, history[-1]["score"])["state"]
    erasure_meta = {k: v for k, v in vars(args).items() if k in (
        "drop_unit", "p_max", "p_mode", "curriculum_epochs", "clean_fraction", "freeze_epochs", "lr", "epochs",
        "inpaint", "packet_size", "seed")}

    def write_checkpoint(path, state, epoch, score):
        payload = {**state, "config": {**cfg, "erasure_training": erasure_meta}, "epoch": epoch,
                   "split": {"seed": split_seed, "model_seed": args.seed},
                   "validation_metrics": {"robust_score_db": score}, "init_checkpoint": args.init_checkpoint}
        if inpainter is not None:
            payload["inpainter_config"] = {"heads": args.inpaint_heads, "layers": args.inpaint_layers}
        torch.save(payload, path)

    best_path = os.path.join(args.output_dir, "adjsccq_erasure_robust.pth")
    write_checkpoint(best_path, best["state"], best["epoch"], best["score"])
    write_checkpoint(os.path.join(args.output_dir, "final.pth"), final_state, args.epochs, history[-1]["score"])

    # Reload the selected model and recompute everything that depends on its weights.
    selected = load_model_bundle(best_path, device)
    train_images = load_image_tensor("cifar10", "train", args.data_dir, args.oracle_max_images, split_seed)
    try:
        from tqdm import tqdm as progress
    except ImportError:  # pragma: no cover
        progress = None
    labels = damage_for_tensor(selected, train_images, 250, progress)
    ranking = ranking_from_damage(labels["damage"])
    mean_damage = labels["damage"].mean(axis=0)
    split_info = {"seed": split_seed, "model_seed": args.seed, "train_images": len(train_images)}
    write_json(os.path.join(args.output_dir, "oracle_ranking.json"), {
        "checkpoint": best_path, "source_split": "train", "split": split_info, "method": "oracle_leave_one_channel_out",
        "ranking_channels_most_to_least_important": ranking, "psnr_drop_db_by_channel": mean_damage.tolist(),
        "num_images": len(train_images), "channel_count": channels, "latent_channel_width": spc,
        "note": "recomputed for the erasure-robust model (with inpainter if present)"})

    final_curves = evaluate_robust(selected.encoder, selected.decoder, selected.inpainter, val_images, channels, spc,
                                   args.packet_size, snr_db, device)
    clean_drop = baseline_curves["channel"][0.0] - final_curves["channel"][0.0]
    notes = [
        f"Clean validation PSNR: {baseline_curves['channel'][0.0]:.2f} -> {final_curves['channel'][0.0]:.2f} dB "
        f"({-clean_drop:+.2f} dB; roadmap budget: lose < 0.3 dB).",
        f"Channel-drop PSNR at 40%: {baseline_curves['channel'][0.4]:.2f} -> {final_curves['channel'][0.4]:.2f} dB; "
        f"packet-drop at 40%: {baseline_curves['packet'][0.4]:.2f} -> {final_curves['packet'][0.4]:.2f} dB.",
        f"Best epoch {best['epoch']} of {args.epochs}; robust score {history[0]['score']:.2f} -> {best['score']:.2f} dB.",
    ]
    if clean_drop > 2.0:
        notes.append("Clean PSNR collapsed (> 2 dB): over-regularised. Lower --p_max to 0.3 or use --curriculum_epochs 20.")
    write_json(os.path.join(args.output_dir, "metrics.json"), {
        "metadata": build_provenance(args=args, checkpoint_path=best_path, global_seed=args.seed, device=device,
                                     extra_files={"init_checkpoint": args.init_checkpoint,
                                                  "oracle_ranking": os.path.join(args.output_dir, "oracle_ranking.json")},
                                     split_info=split_info),
        "best_epoch": best["epoch"], "history": history, "before": baseline_curves, "after": final_curves,
        "oracle_ranking": ranking, "oracle_damage_db": mean_damage.tolist(), "insights": notes})

    figures = []
    for unit_name in UNITS:
        rates = list(EVAL_GRID)
        figures += plotting.plot_lines(
            {"before fine-tuning": [baseline_curves[unit_name][r] for r in rates],
             "after fine-tuning": [final_curves[unit_name][r] for r in rates]}, rates,
            os.path.join(args.output_dir, f"robustness_{unit_name}_drop"), xlabel=f"{unit_name} drop rate",
            ylabel="Validation PSNR (dB)", title=f"Decoder robustness to {unit_name} erasures")
    figures += plotting.plot_lines(
        {"robust score": [h["score"] for h in history], "clean (channel)": [h["curves"]["channel"][0.0] for h in history],
         "40% channel drop": [h["curves"]["channel"][0.4] for h in history]},
        list(range(len(history))), os.path.join(args.output_dir, "training_progress"), xlabel="Epoch",
        ylabel="Validation PSNR (dB)")
    figures += plotting.plot_bars(np.sort(mean_damage)[::-1], [str(c) for c in np.argsort(-mean_damage)],
                                  os.path.join(args.output_dir, "oracle_damage"), ylabel="Mean PSNR drop (dB)",
                                  title="Per-channel damage after fine-tuning", highlight=[0])
    with open(os.path.join(args.output_dir, "metrics.md"), "w", encoding="utf-8") as handle:
        handle.write("# Erasure-robust fine-tuning\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    print("\n".join(f"- {n}" for n in notes))
    wandb_utils.log_images(run, "figures", figures)
    wandb_utils.set_summary(run, {"best_epoch": best["epoch"], "robust_score_before": history[0]["score"],
                                  "robust_score_after": best["score"]})
    wandb_utils.log_artifact(run, os.path.basename(args.output_dir), "model",
                             [best_path, os.path.join(args.output_dir, "oracle_ranking.json")])
    wandb_utils.finish(run)


if __name__ == "__main__":
    main()
