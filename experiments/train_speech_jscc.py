"""Train a speech JSCC-Q model (S1) on LibriSpeech segments, mirroring the image training script.

Same protocol as ``experiments/train_adjsccq_train_split.py`` so that the two
modalities differ only in the data and the 1-D layers (``docs/02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md``,
AD-14 to AD-18):

* recipe v1: Adam (lr 1e-3, constant), no clipping, MSE loss, quantiser sigma annealed
  1.0 -> 0.05 over 50 of 60 epochs, codebook calibrated on the first training batch;
* the best checkpoint is chosen on the **validation** split (dev-clean);
* the oracle ranking (mean leave-one-channel-out SNR drop) comes from the **training**
  split only; the test split (test-clean) is touched once, for the clean-quality
  number, and logged;
* the checkpoint format and ``metrics.json`` keys are those of the image script, so
  ``load_model_bundle``, ``eval_rs_uep``, ``multiseed_analysis``, ``select_scheme`` and
  ``compare_architectures`` read the outputs unchanged. Quality is per-segment SNR (dB)
  stored under the ``psnr_db_*`` keys (``quality_metric`` in metrics.json says so).

    python -m experiments.train_speech_jscc --config configs/speech_deepjsccq1d.yaml --seed 42 \\
        --device mps --data_dir ./data --output_dir results/speech/deepjsccq1d/seed42
"""
import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data.speech import make_speech_loaders, split_info as speech_split_info
from experiments.metrics import quality_db_per_item, si_sdr_db_per_item
from experiments.model_io import ModelBundle
from experiments.reproducibility import build_provenance, log_test_access, seed_everything
from experiments.sensitivity import loo_damage_batch
from experiments.train_adjsccq_train_split import DEVICE_CHOICES, anneal_sigma, resolve_device
from models.adjsccq import NOMINAL_SNR_DB_IDENTITY
from models.registry import ARCHITECTURES, build_jscc, model_arch, model_modality

DEFAULT_CONFIG = "configs/speech_adjsccq1d.yaml"
QUALITY_METRIC = "per-segment SNR (dB), stored in the psnr_db_* fields"


@torch.no_grad()
def evaluate_clean(encoder, decoder, loader, snr_db: float, device, max_items: Optional[int] = None) -> Dict:
    encoder.eval()
    decoder.eval()
    mse_sum = snr_sum = si_sdr_sum = 0.0
    samples = items = 0
    levels = torch.zeros(encoder.quantizer.num_levels, dtype=torch.long)
    for x, _ in tqdm(loader, desc="Clean Eval", leave=False):
        if max_items is not None:
            remaining = max_items - items
            if remaining <= 0:
                break
            x = x[:remaining]
        x = x.to(device)
        _, indices = encoder(x, snr_db, return_indices=True)
        x_hat = decoder(encoder.quantizer.dequantize_indices(indices), snr_db)
        mse_sum += ((x_hat - x) ** 2).sum().item()
        samples += x.numel()
        snr_sum += quality_db_per_item(x_hat, x).sum().item()
        si_sdr_sum += si_sdr_db_per_item(x_hat, x).sum().item()
        items += x.shape[0]
        levels += torch.bincount(indices.flatten().cpu(), minlength=levels.numel())
    return {
        "num_images": items,                      # key name shared with the image script
        "mse": mse_sum / samples,
        "psnr_db_mean_per_image": snr_sum / items,  # = mean per-segment SNR (QUALITY_METRIC)
        "si_sdr_db_mean": si_sdr_sum / items,
        "codebook_levels_used": int((levels > 0).sum()),
        "codebook_levels_total": levels.numel(),
    }


@torch.no_grad()
def compute_oracle_ranking(bundle: ModelBundle, loader, max_items: Optional[int] = None) -> Dict:
    """Mean leave-one-channel-out SNR drop over the training split (same definition as the image oracle)."""
    bundle.eval()
    channels = bundle.channels
    drop = torch.zeros(channels, dtype=torch.float64)   # accumulated on the CPU (MPS has no float64)
    items = 0
    for x, _ in tqdm(loader, desc="Computing Oracle", leave=False):
        if max_items is not None:
            remaining = max_items - items
            if remaining <= 0:
                break
            x = x[:remaining]
        x = x.to(bundle.device)
        _, indices = bundle.encoder(x, bundle.snr_db, return_indices=True)
        latent = bundle.encoder.quantizer.dequantize_indices(indices)
        damage, _ = loo_damage_batch(bundle, latent, x, bundle.snr_db)  # the same LOO code as every other study
        drop += damage.sum(dim=0).cpu().double()
        items += x.shape[0]
    if items == 0:
        raise ValueError("cannot compute the oracle ranking on an empty loader")
    drop /= items
    return {
        "method": "oracle_leave_one_channel_out",
        "score_definition": "mean clean SNR minus mean SNR after zeroing one latent channel (dB)",
        "ranking_channels_most_to_least_important": torch.argsort(drop, descending=True, stable=True).tolist(),
        "psnr_drop_db_by_channel": drop.tolist(),   # key shared with the image oracle (SNR drop here)
        "num_images": items,
        "channel_count": channels,
        "latent_channel_width": bundle.symbols_per_channel,
        "quality_metric": QUALITY_METRIC,
    }


def _save_checkpoint(path, encoder, decoder, cfg, split, epoch, val_metrics):
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "config": cfg, "split": split,
                "epoch": epoch, "validation_metrics": val_metrics}, path)


def main(args) -> None:
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.arch is not None:
        cfg["model"]["arch"] = args.arch
    arch = cfg["model"]["arch"] = model_arch(cfg)
    if model_modality(cfg) != "speech":
        raise SystemExit(f"arch {arch!r} is not a speech model; use experiments.train_adjsccq_train_split")
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    max_train = args.max_train if args.max_train is not None else cfg.get("data", {}).get("max_train")
    train_loader, val_loader, test_loader = make_speech_loaders(
        args.data_dir, cfg["data"]["batch_size"], cfg["data"]["num_workers"], loader_seed=seed + 1,
        max_train=max_train)
    split = {
        "seed": 0,   # LibriSpeech splits are fixed; the segment draw is seeded in the cache (see "segments")
        "model_seed": seed,
        "dataset": "librispeech",
        "train_images": len(train_loader.dataset),
        "validation_images": len(val_loader.dataset),
        "test_images": len(test_loader.dataset),
        "segments": speech_split_info(args.data_dir),
        "test_split_untouched_until_final_evaluation": True,
    }
    cfg["train_split_experiment"] = split

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.wandb_name or f"{arch}-librispeech-seed{seed}",
                         tags=["speech", arch, f"seed{seed}"], group="speech-models", job_type="train", config=cfg)

    snr_db = cfg.get("af_conditioning", {}).get("nominal_snr_db", NOMINAL_SNR_DB_IDENTITY)
    encoder, decoder = build_jscc(cfg, device)
    x_calib, _ = next(iter(train_loader))
    encoder.calibrate_quantizer(x_calib.to(device), snr_db=snr_db, low_pct=cfg["quantizer"]["calibrate_low_pct"],
                                high_pct=cfg["quantizer"]["calibrate_high_pct"])
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=cfg["train"]["lr"])
    criterion = nn.MSELoss()
    best_path, final_path = os.path.join(args.output_dir, "best_val.pth"), os.path.join(args.output_dir, "final.pth")
    best_val, best_epoch, step, history = float("-inf"), 0, 0, []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        sigma = anneal_sigma(epoch - 1, cfg["quantizer"]["sigma_start"], cfg["quantizer"]["sigma_end"],
                             cfg["quantizer"]["sigma_anneal_epochs"])
        encoder.quantizer.set_sigma(sigma)
        encoder.train()
        decoder.train()
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}", leave=False):
            x = x.to(device)
            optimizer.zero_grad()
            loss = criterion(decoder(encoder(x, snr_db), snr_db), x)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            step += 1
            if run is not None and step % cfg["train"].get("log_every", 100) == 0:
                run.log({"train/mse": loss.item(), "train/sigma_q": sigma}, step=step)
        val = evaluate_clean(encoder, decoder, val_loader, snr_db, device)
        history.append({"epoch": epoch, "sigma_q": sigma, "val_psnr_db": val["psnr_db_mean_per_image"],
                        "val_si_sdr_db": val["si_sdr_db_mean"]})
        if run is not None:
            run.log({"validation/snr_db": val["psnr_db_mean_per_image"], "validation/si_sdr_db": val["si_sdr_db_mean"],
                     "validation/mse": val["mse"]}, step=step)
        if val["psnr_db_mean_per_image"] > best_val:
            best_val, best_epoch = val["psnr_db_mean_per_image"], epoch
            _save_checkpoint(best_path, encoder, decoder, cfg, split, epoch, {**val, "epoch": epoch})
        print(f"epoch {epoch:3d} | sigma_q {sigma:.4f} | validation SNR {val['psnr_db_mean_per_image']:.2f} dB | "
              f"SI-SDR {val['si_sdr_db_mean']:.2f} dB")

    encoder.quantizer.set_sigma(cfg["quantizer"]["sigma_end"])
    _save_checkpoint(final_path, encoder, decoder, cfg, split, cfg["train"]["epochs"],
                     evaluate_clean(encoder, decoder, val_loader, snr_db, device))

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    bundle = ModelBundle(encoder, decoder, None, cfg, best_path, device, 0).eval()

    clean_train = evaluate_clean(encoder, decoder, train_loader, snr_db, device, args.oracle_max_items)
    clean_val = evaluate_clean(encoder, decoder, val_loader, snr_db, device)
    log_test_access("experiments.train_speech_jscc", best_path, note=f"clean test SNR of {arch} seed {seed}")
    clean_test = evaluate_clean(encoder, decoder, test_loader, snr_db, device)
    oracle = compute_oracle_ranking(bundle, train_loader, args.oracle_max_items)
    oracle_path = os.path.join(args.output_dir, "oracle_ranking.json")
    with open(oracle_path, "w", encoding="utf-8") as handle:
        json.dump({"checkpoint": best_path, "source_split": "train", "split": split, **oracle}, handle, indent=2)

    metrics = {
        "metadata": build_provenance(args=args, checkpoint_path=best_path, dataset="LibriSpeech",
                                     extra_files={"oracle_ranking": oracle_path, "config": args.config},
                                     split_info=split, global_seed=seed, device=device,
                                     extra={"model_seed": seed, "arch": arch, "channels": oracle["channel_count"],
                                            "quality_metric": QUALITY_METRIC, "oracle_source_split": "train"}),
        "checkpoint_best": best_path, "checkpoint_final": final_path, "oracle_ranking": oracle_path,
        "best_epoch": best_epoch, "split": split,
        "clean_train": clean_train, "clean_validation": clean_val, "clean_test": clean_test,
        "oracle": oracle, "validation_history": history,
        "recipe": {"lr_schedule": "constant", "grad_clip": args.grad_clip, "lr": cfg["train"]["lr"],
                   "epochs": cfg["train"]["epochs"]},
        "arch": arch, "train_device": str(device), "quality_metric": QUALITY_METRIC,
        "rs_uep_test": [], "rs_uep_validation": [],   # sweeps are run by experiments.eval_rs_uep
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    if run is not None:
        run.summary.update({"best_epoch": best_epoch, "test/snr_db": clean_test["psnr_db_mean_per_image"],
                            "oracle/ranking": oracle["ranking_channels_most_to_least_important"]})
        run.finish()
    print(json.dumps({"best_checkpoint": best_path, "best_epoch": best_epoch,
                      "test_snr_db": clean_test["psnr_db_mean_per_image"],
                      "oracle_ranking": oracle["ranking_channels_most_to_least_important"],
                      "worst_channel_damage_db": float(np.max(oracle["psnr_drop_db_by_channel"]))}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--arch", default=None, choices=sorted(a for a, s in ARCHITECTURES.items() if s.modality == "speech"))
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=None, help="model seed (default: config seed, 42)")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs (smoke tests)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="0 = recipe v1 (no clipping)")
    parser.add_argument("--max_train", type=int, default=None, help="use only the first N training segments (smoke)")
    parser.add_argument("--oracle_max_items", type=int, default=None, help="cap the train-split oracle (smoke)")
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    parser.add_argument("--wandb_name", default=None)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
