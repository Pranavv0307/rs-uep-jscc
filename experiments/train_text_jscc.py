"""Train a text JSCC-Q model (T1) on text8 windows, mirroring the speech / image training scripts.

Same protocol as ``experiments/train_speech_jscc.py`` so that the modalities differ only in the
data, the input/output layers and the loss (``docs/02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md``,
AD-22 to AD-28):

* recipe v1: Adam (lr 1e-3, constant), no clipping, quantiser sigma annealed 1.0 -> 0.05 over
  50 of 60 epochs, codebook calibrated on the first training batch. The loss is **cross-entropy**
  of the per-character logits (the categorical counterpart of MSE; AD-24);
* the best checkpoint is chosen on the **validation** split by character accuracy;
* the oracle ranking (mean leave-one-channel-out drop of character accuracy, in percentage
  points) comes from the **training** split only; the test split is touched once, for the clean
  number, and logged;
* the checkpoint format and ``metrics.json`` keys are those of the image / speech scripts, so
  ``load_model_bundle``, ``eval_rs_uep``, ``multiseed_analysis``, ``select_scheme`` and
  ``compare_architectures`` read the outputs unchanged. Quality is per-window character accuracy
  (%) stored under the ``psnr_db_*`` keys (``quality_metric`` in metrics.json says so).

    python -m experiments.train_text_jscc --config configs/text_deepjsccq.yaml --seed 42 \\
        --device mps --data_dir ./data --output_dir results/text/deepjsccq_text/seed42
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

from data.text import make_text_loaders, split_info as text_split_info
from experiments.metrics import bits_per_char_per_item, quality_db_per_item, word_accuracy_per_item
from experiments.model_io import ModelBundle
from experiments.reproducibility import build_provenance, log_test_access, seed_everything
from experiments.sensitivity import loo_damage_batch
from experiments.train_adjsccq_train_split import DEVICE_CHOICES, anneal_sigma, resolve_device
from models.adjsccq import NOMINAL_SNR_DB_IDENTITY
from models.registry import ARCHITECTURES, build_jscc, model_arch, model_modality

DEFAULT_CONFIG = "configs/text_adjsccq.yaml"
QUALITY_METRIC = ("per-window character accuracy (%), stored in the psnr_db_* fields; "
                  "damages and gains are in percentage points")


@torch.no_grad()
def evaluate_clean(encoder, decoder, loader, snr_db: float, device, max_items: Optional[int] = None) -> Dict:
    encoder.eval()
    decoder.eval()
    acc_sum = word_sum = bpc_sum = 0.0
    items = 0
    levels = torch.zeros(encoder.quantizer.num_levels, dtype=torch.long)
    for x, _ in tqdm(loader, desc="Clean Eval", leave=False):
        if max_items is not None:
            remaining = max_items - items
            if remaining <= 0:
                break
            x = x[:remaining]
        x = x.to(device)
        _, indices = encoder(x, snr_db, return_indices=True)
        logits = decoder(encoder.quantizer.dequantize_indices(indices), snr_db)
        acc_sum += quality_db_per_item(logits, x).sum().item()
        word_sum += word_accuracy_per_item(logits, x).sum().item()
        bpc_sum += bits_per_char_per_item(logits, x).sum().item()
        items += x.shape[0]
        levels += torch.bincount(indices.flatten().cpu(), minlength=levels.numel())
    return {
        "num_images": items,                        # key name shared with the image script
        "psnr_db_mean_per_image": acc_sum / items,  # = mean per-window character accuracy in % (QUALITY_METRIC)
        "char_acc_mean": acc_sum / items,           # the same number under its own name
        "word_acc_mean": word_sum / items,
        "bits_per_char": bpc_sum / items,
        "codebook_levels_used": int((levels > 0).sum()),
        "codebook_levels_total": levels.numel(),
    }


@torch.no_grad()
def compute_oracle_ranking(bundle: ModelBundle, loader, max_items: Optional[int] = None) -> Dict:
    """Mean leave-one-channel-out accuracy drop over the training split (same definition as the image oracle)."""
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
        "score_definition": "mean clean character accuracy minus mean accuracy after zeroing one latent channel "
                            "(percentage points)",
        "ranking_channels_most_to_least_important": torch.argsort(drop, descending=True, stable=True).tolist(),
        "psnr_drop_db_by_channel": drop.tolist(),   # key shared with the image oracle (accuracy drop in pp here)
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
    if model_modality(cfg) != "text":
        raise SystemExit(f"arch {arch!r} is not a text model; use experiments.train_speech_jscc or "
                         "experiments.train_adjsccq_train_split")
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    max_train = args.max_train if args.max_train is not None else cfg.get("data", {}).get("max_train")
    train_loader, val_loader, test_loader = make_text_loaders(
        args.data_dir, cfg["data"]["batch_size"], cfg["data"]["num_workers"], loader_seed=seed + 1,
        max_train=max_train)
    split = {
        "seed": 0,   # text8 splits are fixed; the window draw is seeded in the cache (see "segments")
        "model_seed": seed,
        "dataset": "text8",
        "train_images": len(train_loader.dataset),
        "validation_images": len(val_loader.dataset),
        "test_images": len(test_loader.dataset),
        "segments": text_split_info(args.data_dir),
        "test_split_untouched_until_final_evaluation": True,
    }
    cfg["train_split_experiment"] = split

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.wandb_name or f"{arch}-text8-seed{seed}",
                         tags=["text", arch, f"seed{seed}"], group="text-models", job_type="train", config=cfg)

    snr_db = cfg.get("af_conditioning", {}).get("nominal_snr_db", NOMINAL_SNR_DB_IDENTITY)
    encoder, decoder = build_jscc(cfg, device)
    x_calib, _ = next(iter(train_loader))
    encoder.calibrate_quantizer(x_calib.to(device), snr_db=snr_db, low_pct=cfg["quantizer"]["calibrate_low_pct"],
                                high_pct=cfg["quantizer"]["calibrate_high_pct"])
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=cfg["train"]["lr"])
    criterion = nn.CrossEntropyLoss()   # logits (B, V, L) vs ids (B, L)
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
                run.log({"train/cross_entropy": loss.item(), "train/sigma_q": sigma}, step=step)
        val = evaluate_clean(encoder, decoder, val_loader, snr_db, device)
        history.append({"epoch": epoch, "sigma_q": sigma, "val_psnr_db": val["psnr_db_mean_per_image"],
                        "val_word_acc": val["word_acc_mean"], "val_bits_per_char": val["bits_per_char"]})
        if run is not None:
            run.log({"validation/char_acc": val["char_acc_mean"], "validation/word_acc": val["word_acc_mean"],
                     "validation/bits_per_char": val["bits_per_char"]}, step=step)
        if val["psnr_db_mean_per_image"] > best_val:
            best_val, best_epoch = val["psnr_db_mean_per_image"], epoch
            _save_checkpoint(best_path, encoder, decoder, cfg, split, epoch, {**val, "epoch": epoch})
        print(f"epoch {epoch:3d} | sigma_q {sigma:.4f} | validation char acc {val['char_acc_mean']:.2f}% | "
              f"word acc {val['word_acc_mean']:.2f}% | {val['bits_per_char']:.3f} bits/char")

    encoder.quantizer.set_sigma(cfg["quantizer"]["sigma_end"])
    _save_checkpoint(final_path, encoder, decoder, cfg, split, cfg["train"]["epochs"],
                     evaluate_clean(encoder, decoder, val_loader, snr_db, device))

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    bundle = ModelBundle(encoder, decoder, None, cfg, best_path, device, 0).eval()

    clean_train = evaluate_clean(encoder, decoder, train_loader, snr_db, device, args.oracle_max_items)
    clean_val = evaluate_clean(encoder, decoder, val_loader, snr_db, device)
    log_test_access("experiments.train_text_jscc", best_path, note=f"clean test accuracy of {arch} seed {seed}")
    clean_test = evaluate_clean(encoder, decoder, test_loader, snr_db, device)
    oracle = compute_oracle_ranking(bundle, train_loader, args.oracle_max_items)
    oracle_path = os.path.join(args.output_dir, "oracle_ranking.json")
    with open(oracle_path, "w", encoding="utf-8") as handle:
        json.dump({"checkpoint": best_path, "source_split": "train", "split": split, **oracle}, handle, indent=2)

    metrics = {
        "metadata": build_provenance(args=args, checkpoint_path=best_path, dataset="text8",
                                     extra_files={"oracle_ranking": oracle_path, "config": args.config},
                                     split_info=split, global_seed=seed, device=device,
                                     extra={"model_seed": seed, "arch": arch, "channels": oracle["channel_count"],
                                            "quality_metric": QUALITY_METRIC, "oracle_source_split": "train"}),
        "checkpoint_best": best_path, "checkpoint_final": final_path, "oracle_ranking": oracle_path,
        "best_epoch": best_epoch, "split": split,
        "clean_train": clean_train, "clean_validation": clean_val, "clean_test": clean_test,
        "oracle": oracle, "validation_history": history,
        "recipe": {"lr_schedule": "constant", "grad_clip": args.grad_clip, "lr": cfg["train"]["lr"],
                   "epochs": cfg["train"]["epochs"], "loss": "cross_entropy"},
        "arch": arch, "train_device": str(device), "quality_metric": QUALITY_METRIC,
        "rs_uep_test": [], "rs_uep_validation": [],   # sweeps are run by experiments.eval_rs_uep
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    if run is not None:
        run.summary.update({"best_epoch": best_epoch, "test/char_acc": clean_test["char_acc_mean"],
                            "oracle/ranking": oracle["ranking_channels_most_to_least_important"]})
        run.finish()
    print(json.dumps({"best_checkpoint": best_path, "best_epoch": best_epoch,
                      "test_char_acc": clean_test["char_acc_mean"], "test_word_acc": clean_test["word_acc_mean"],
                      "oracle_ranking": oracle["ranking_channels_most_to_least_important"],
                      "worst_channel_damage_pp": float(np.max(oracle["psnr_drop_db_by_channel"]))}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--arch", default=None, choices=sorted(a for a, s in ARCHITECTURES.items() if s.modality == "text"))
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=None, help="model seed (default: config seed, 42)")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs (smoke tests)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="0 = recipe v1 (no clipping)")
    parser.add_argument("--max_train", type=int, default=None, help="use only the first N training windows (smoke)")
    parser.add_argument("--oracle_max_items", type=int, default=None, help="cap the train-split oracle (smoke)")
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    parser.add_argument("--wandb_name", default=None)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
