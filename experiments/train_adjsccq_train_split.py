"""Train and evaluate an isolated CIFAR-10 train/validation/test experiment.

The existing ADJSCC-Q and RS-UEP modules are reused unchanged. This entry
point only changes dataset ownership: 40,000 images from CIFAR-10's official
training split are used for optimization, 10,000 for validation, and the
official 10,000-image test split is held out until final evaluation.

The workflow saves best/final checkpoints, computes a frozen training-derived
oracle channel ranking by leave-one-channel-out PSNR drop, and evaluates both
attention and oracle importance under the complete RS-UEP sweep.
"""
import argparse
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from coding.erasure_channel import apply_erasure_mask, sample_erasure_mask
from coding.importance import channel_order_from_attention
from coding.rs_pipeline import (
    NO_RS,
    SCHEME_CANDIDATES,
    decode_blocks,
    dequantize_with_fallback,
    encode_blocks,
    random_channel_order,
)
from experiments.metrics import psnr_per_image, ssim
from models.adjsccq import ADJSCCQDecoder, ADJSCCQEncoder, NOMINAL_SNR_DB_IDENTITY


DEFAULT_CONFIG = "configs/adjsccq_identity_train.yaml"
DEFAULT_OUTPUT_DIR = "results/train_split_adjsccq"
DEFAULT_PROJECT = "rs-uep-jscc"
DEFAULT_VAL_SIZE = 10_000


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def anneal_sigma(epoch: int, sigma_start: float, sigma_end: float, anneal_epochs: int) -> float:
    if epoch >= anneal_epochs:
        return sigma_end
    fraction = epoch / max(anneal_epochs, 1)
    return sigma_start + fraction * (sigma_end - sigma_start)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(data_dir: str, batch_size: int, num_workers: int,
                 seed: int, val_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    if not 0 < val_size < len(train_dataset):
        raise ValueError(f"val_size must be between 1 and {len(train_dataset) - 1}")

    split_generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(train_dataset), generator=split_generator).tolist()
    train_indices = permutation[:-val_size]
    val_indices = permutation[-val_size:]
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    loader_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        generator=loader_generator, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def _metric_totals(x_hat: torch.Tensor, x: torch.Tensor) -> Tuple[float, float, float, int]:
    mse_sum = ((x_hat - x) ** 2).sum().item()
    pixel_count = x.numel()
    psnr_sum = psnr_per_image(x_hat, x).sum().item()
    ssim_sum = ssim(x_hat, x).sum().item()
    return mse_sum, psnr_sum, ssim_sum, x.shape[0]


@torch.no_grad()
def evaluate_clean(encoder, decoder, loader, snr_db: float, device,
                   max_images: Optional[int] = None) -> Dict[str, float]:
    encoder.eval()
    decoder.eval()
    mse_sum = psnr_sum = ssim_sum = 0.0
    pixel_count = image_count = 0
    levels = torch.zeros(encoder.quantizer.num_levels, dtype=torch.long)

    for x, _ in tqdm(loader, desc="Clean Eval", leave=False):
        if max_images is not None:
            remaining = max_images - image_count
            if remaining <= 0:
                break
            x = x[:remaining]
        x = x.to(device)
        _, attention, indices = encoder(x, snr_db, return_attn=True, return_indices=True)
        del attention
        x_hat = decoder(encoder.quantizer.dequantize_indices(indices), snr_db)
        batch_mse, batch_psnr, batch_ssim, batch_images = _metric_totals(x_hat, x)
        mse_sum += batch_mse
        psnr_sum += batch_psnr
        ssim_sum += batch_ssim
        pixel_count += x.numel()
        image_count += batch_images
        levels += torch.bincount(indices.flatten().cpu(), minlength=levels.numel())

    dataset_mse = mse_sum / pixel_count
    return {
        "num_images": image_count,
        "mse": dataset_mse,
        "psnr_db_mean_per_image": psnr_sum / image_count,
        "psnr_db_from_dataset_mse": psnr_from_mse(dataset_mse),
        "ssim_mean": ssim_sum / image_count,
        "codebook_levels_used": int((levels > 0).sum()),
        "codebook_levels_total": levels.numel(),
    }


@torch.no_grad()
def compute_oracle_ranking(encoder, decoder, loader, snr_db: float, device,
                           max_images: Optional[int] = None) -> Dict:
    """Rank channels by mean leave-one-channel-out reconstruction PSNR drop."""
    encoder.eval()
    decoder.eval()
    channel_count = 16
    psnr_drop = torch.zeros(channel_count, dtype=torch.float64)
    mse_increase = torch.zeros(channel_count, dtype=torch.float64)
    ssim_drop = torch.zeros(channel_count, dtype=torch.float64)
    image_count = 0

    for x, _ in tqdm(loader, desc="Computing Oracle", leave=False):
        if max_images is not None:
            remaining = max_images - image_count
            if remaining <= 0:
                break
            x = x[:remaining]
        x = x.to(device)
        _, _, indices = encoder(x, snr_db, return_attn=True, return_indices=True)
        latent = encoder.quantizer.dequantize_indices(indices)
        clean = decoder(latent, snr_db)
        clean_psnr = psnr_per_image(clean, x)
        clean_mse = ((clean - x) ** 2).mean(dim=[1, 2, 3])
        clean_ssim = ssim(clean, x)
        batch_size, latent_length = latent.shape
        if latent_length % channel_count != 0:
            raise ValueError("Oracle channel count must divide the latent length")
        channel_width = latent_length // channel_count

        for channel in range(channel_count):
            ablated = latent.clone()
            start = channel * channel_width
            ablated[:, start:start + channel_width] = 0.0
            reconstruction = decoder(ablated, snr_db)
            psnr_drop[channel] += (clean_psnr - psnr_per_image(reconstruction, x)).sum().item()
            mse_increase[channel] += (
                ((reconstruction - x) ** 2).mean(dim=[1, 2, 3]) - clean_mse
            ).sum().item()
            ssim_drop[channel] += (clean_ssim - ssim(reconstruction, x)).sum().item()
        image_count += batch_size

    if image_count == 0:
        raise ValueError("Cannot compute oracle ranking on an empty loader")
    psnr_drop /= image_count
    mse_increase /= image_count
    ssim_drop /= image_count
    ranking = torch.argsort(psnr_drop, descending=True, stable=True).tolist()
    return {
        "method": "oracle_leave_one_channel_out",
        "score_definition": "mean clean PSNR minus mean PSNR after zeroing one latent channel",
        "ranking_channels_most_to_least_important": ranking,
        "psnr_drop_db_by_channel": psnr_drop.tolist(),
        "mse_increase_by_channel": mse_increase.tolist(),
        "ssim_drop_by_channel": ssim_drop.tolist(),
        "num_images": image_count,
        "channel_count": channel_count,
        "latent_channel_width": channel_width,
    }


def _methods() -> Tuple[str, ...]:
    return ("attention", "oracle")


def _scheme_for_method(method: str, scheme_name: str):
    if scheme_name == "no_rs":
        return NO_RS
    return SCHEME_CANDIDATES[scheme_name]


def _empty_stats(method: str, scheme_name: str, rate: float, seed: int, scheme) -> Dict:
    return {
        "method": method,
        "scheme": scheme_name,
        "erasure_rate": rate,
        "seed": seed,
        "images": 0,
        "mse_sum": 0.0,
        "psnr_sum": 0.0,
        "ssim_sum": 0.0,
        "pixel_count": 0,
        "failed_symbols": 0,
        "erased_packets": 0,
        "total_packets": 0,
        "block_failures": [0 for _ in scheme.tiers],
        "coding_symbols": scheme.transmitted_symbols,
        "data_symbols": scheme.data_symbols,
        "side_info_bytes_per_image": 16,
    }


def _finish_stats(stats: Dict) -> Dict:
    images = stats["images"]
    data_symbols = stats["data_symbols"]
    transmitted = stats["coding_symbols"] + stats["side_info_bytes_per_image"]
    return {
        **stats,
        "mse": stats["mse_sum"] / stats["pixel_count"],
        "psnr_db_mean_per_image": stats["psnr_sum"] / images,
        "ssim_mean": stats["ssim_sum"] / images,
        "failed_symbol_fraction": stats["failed_symbols"] / (images * data_symbols),
        "realized_erasure_rate": stats["erased_packets"] / stats["total_packets"],
        "total_transmitted_bytes_per_image": transmitted,
        "total_overhead_fraction_including_side_info": transmitted / data_symbols - 1.0,
        "block_failure_rate_by_tier": [
            count / images for count in stats["block_failures"]
        ],
    }


@torch.no_grad()
def evaluate_rs(encoder, decoder, loader, snr_db: float, device,
                oracle_ranking: Sequence[int], rates: Sequence[float],
                seeds: Sequence[int], max_images: Optional[int] = None) -> List[Dict]:
    encoder.eval()
    decoder.eval()
    rows = []
    scheme_names = list(SCHEME_CANDIDATES)
    for method in _methods():
        for rate in rates:
            for seed in seeds:
                accumulators = {
                    name: _empty_stats(method, name, rate, seed, _scheme_for_method(method, name))
                    for name in scheme_names
                }
                image_count = 0
                for batch_index, (x, _) in enumerate(tqdm(loader, desc=f"RS-UEP {method} r={rate} s={seed}", leave=False)):
                    if max_images is not None:
                        remaining = max_images - image_count
                        if remaining <= 0:
                            break
                        x = x[:remaining]
                    x = x.to(device)
                    batch_size = x.shape[0]
                    _, attention, indices = encoder(
                        x, snr_db, return_attn=True, return_indices=True
                    )
                    attention_order = channel_order_from_attention(
                        attention["af5_bottleneck"]
                    )
                    oracle_order = torch.tensor(
                        oracle_ranking, device=device
                    ).unsqueeze(0).expand(batch_size, -1)
                    mask_generator = torch.Generator().manual_seed(
                        seed * 100000 + batch_index
                    )
                    masks = {
                        96: sample_erasure_mask(batch_size, 96, rate, mask_generator),
                        64: sample_erasure_mask(batch_size, 64, rate, mask_generator),
                    }
                    for scheme_name in scheme_names:
                        scheme = _scheme_for_method(method, scheme_name)
                        order = attention_order if method == "attention" else oracle_order
                        packets = encode_blocks(indices, scheme, order)
                        received = apply_erasure_mask(
                            packets, masks[scheme.total_packets]
                        ).received
                        decoded = decode_blocks(received, scheme, order)
                        latent = dequantize_with_fallback(
                            encoder.quantizer, decoded.indices, decoded.failed
                        )
                        reconstruction = decoder(latent, snr_db)
                        stats = accumulators[scheme_name]
                        batch_mse, batch_psnr, batch_ssim, batch_images = _metric_totals(
                            reconstruction, x
                        )
                        stats["mse_sum"] += batch_mse
                        stats["psnr_sum"] += batch_psnr
                        stats["ssim_sum"] += batch_ssim
                        stats["pixel_count"] += x.numel()
                        stats["images"] += batch_images
                        stats["failed_symbols"] += decoded.failed.sum().item()
                        stats["erased_packets"] += received.erased.sum().item()
                        stats["total_packets"] += received.erased.numel()
                        stats["block_failures"] = [
                            total + int(failed)
                            for total, failed in zip(
                                stats["block_failures"],
                                decoded.block_failed.sum(dim=0).tolist(),
                            )
                        ]
                    image_count += batch_size
                rows.extend(_finish_stats(accumulators[name]) for name in scheme_names)
    return rows


def _log_table(wandb, key: str, rows: List[Dict]) -> None:
    if not rows:
        return
    table = wandb.Table(columns=list(rows[0].keys()))
    for row in rows:
        table.add_data(*[row[column] for column in table.columns])
    wandb.log({key: table})


def _save_checkpoint(path: str, encoder, decoder, cfg: Dict, split_info: Dict,
                     epoch: int, val_metrics: Dict) -> None:
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "config": cfg,
        "split": split_info,
        "epoch": epoch,
        "validation_metrics": val_metrics,
    }, path)


def main(config_path: str, data_dir: str, output_dir: str, val_size: int,
         oracle_max_images: Optional[int], eval_max_images: Optional[int],
         use_wandb: bool, wandb_project: str, wandb_name: Optional[str]) -> None:
    with open(config_path) as handle:
        cfg = yaml.safe_load(handle)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        data_dir, cfg["data"]["batch_size"], cfg["data"]["num_workers"], seed, val_size
    )
    split_info = {
        "seed": seed,
        "train_images": len(train_loader.dataset),
        "validation_images": len(val_loader.dataset),
        "test_images": len(test_loader.dataset),
        "test_split_untouched_until_final_evaluation": True,
    }
    cfg["train_split_experiment"] = split_info

    wandb = None
    run = None
    if use_wandb:
        import wandb as wandb_module
        wandb = wandb_module
        run = wandb.init(
            project=wandb_project,
            name=wandb_name or "adjsccq-train-split",
            tags=["train-validation-test", "adjsccq", "rs-uep", "oracle"],
            config=cfg,
        )

    nominal_snr_db = cfg["af_conditioning"]["nominal_snr_db"]
    encoder = ADJSCCQEncoder(
        k_over_n=cfg["model"]["k_over_n"],
        image_size=cfg["model"]["image_size"],
        num_levels=cfg["quantizer"]["num_levels"],
        init_range=tuple(cfg["quantizer"]["init_range"]),
        init_sigma=cfg["quantizer"]["sigma_start"],
    ).to(device)
    decoder = ADJSCCQDecoder(k=encoder.k, image_size=cfg["model"]["image_size"]).to(device)
    if wandb is not None:
        wandb.config.update({"k": encoder.k, "split": split_info})

    x_calib, _ = next(iter(train_loader))
    encoder.calibrate_quantizer(
        x_calib.to(device), snr_db=nominal_snr_db,
        low_pct=cfg["quantizer"]["calibrate_low_pct"],
        high_pct=cfg["quantizer"]["calibrate_high_pct"],
    )
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg["train"]["lr"],
    )
    criterion = nn.MSELoss()
    best_val_psnr = float("-inf")
    best_epoch = 0
    step = 0
    best_path = os.path.join(output_dir, "best_val.pth")
    final_path = os.path.join(output_dir, "final.pth")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        sigma = anneal_sigma(
            epoch - 1, cfg["quantizer"]["sigma_start"],
            cfg["quantizer"]["sigma_end"], cfg["quantizer"]["sigma_anneal_epochs"],
        )
        encoder.quantizer.set_sigma(sigma)
        encoder.train()
        decoder.train()
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}", leave=False):
            x = x.to(device)
            optimizer.zero_grad()
            reconstruction = decoder(encoder(x, nominal_snr_db), nominal_snr_db)
            loss = criterion(reconstruction, x)
            loss.backward()
            optimizer.step()
            step += 1
            if wandb is not None and step % cfg["train"]["log_every"] == 0:
                wandb.log({
                    "train/mse": loss.item(),
                    "train/psnr_db": psnr_from_mse(loss.item()),
                    "train/sigma_q": sigma,
                }, step=step)

        val_metrics = evaluate_clean(
            encoder, decoder, val_loader, nominal_snr_db, device
        )
        val_metrics["epoch"] = epoch
        val_metrics["sigma_q"] = sigma
        if wandb is not None:
            wandb.log({
                "validation/mse": val_metrics["mse"],
                "validation/psnr_db": val_metrics["psnr_db_mean_per_image"],
                "validation/ssim": val_metrics["ssim_mean"],
                "train/sigma_q": sigma,
            }, step=step)
        if val_metrics["psnr_db_mean_per_image"] > best_val_psnr:
            best_val_psnr = val_metrics["psnr_db_mean_per_image"]
            best_epoch = epoch
            _save_checkpoint(
                best_path, encoder, decoder, cfg, split_info, epoch, val_metrics
            )
        print(
            f"epoch {epoch:3d} | sigma_q {sigma:.4f} | "
            f"validation PSNR {val_metrics['psnr_db_mean_per_image']:.2f} dB | "
            f"SSIM {val_metrics['ssim_mean']:.4f}"
        )

    encoder.quantizer.set_sigma(cfg["quantizer"]["sigma_end"])
    final_val_metrics = evaluate_clean(
        encoder, decoder, val_loader, nominal_snr_db, device
    )
    _save_checkpoint(
        final_path, encoder, decoder, cfg, split_info, cfg["train"]["epochs"],
        final_val_metrics,
    )

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    encoder.load_state_dict(best_checkpoint["encoder"])
    decoder.load_state_dict(best_checkpoint["decoder"])
    encoder.eval()
    decoder.eval()

    train_clean = evaluate_clean(encoder, decoder, train_loader, nominal_snr_db, device,
                                 oracle_max_images)
    val_clean = evaluate_clean(encoder, decoder, val_loader, nominal_snr_db, device)
    test_clean = evaluate_clean(encoder, decoder, test_loader, nominal_snr_db, device,
                               eval_max_images)
    oracle = compute_oracle_ranking(
        encoder, decoder, train_loader, nominal_snr_db, device, oracle_max_images
    )
    oracle_path = os.path.join(output_dir, "oracle_ranking.json")
    with open(oracle_path, "w") as handle:
        json.dump({
            "checkpoint": best_path,
            "source_split": "train",
            "split": split_info,
            **oracle,
        }, handle, indent=2)

    rates = tuple(cfg.get("rs_uep", {}).get(
        "erasure_rates", [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]
    ))
    seeds = tuple(cfg.get("rs_uep", {}).get("seeds", [0, 1, 2]))
    rs_rows = evaluate_rs(
        encoder, decoder, test_loader, nominal_snr_db, device,
        oracle["ranking_channels_most_to_least_important"], rates, seeds,
        eval_max_images,
    )
    rs_validation_rows = evaluate_rs(
        encoder, decoder, val_loader, nominal_snr_db, device,
        oracle["ranking_channels_most_to_least_important"], rates, seeds,
        eval_max_images,
    )
    metrics = {
        "checkpoint_best": best_path,
        "checkpoint_final": final_path,
        "oracle_ranking": oracle_path,
        "best_epoch": best_epoch,
        "split": split_info,
        "clean_train": train_clean,
        "clean_validation": val_clean,
        "clean_test": test_clean,
        "oracle": oracle,
        "rs_uep_validation": rs_validation_rows,
        "rs_uep_test": rs_rows,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)

    if wandb is not None:
        oracle_rows = [
            {
                "channel": channel,
                "psnr_drop_db": oracle["psnr_drop_db_by_channel"][channel],
                "mse_increase": oracle["mse_increase_by_channel"][channel],
                "ssim_drop": oracle["ssim_drop_by_channel"][channel],
            }
            for channel in range(oracle["channel_count"])
        ]
        wandb.summary.update({
            "best_epoch": best_epoch,
            "train/mse": train_clean["mse"],
            "train/psnr_db": train_clean["psnr_db_mean_per_image"],
            "train/ssim": train_clean["ssim_mean"],
            "validation/final_mse": val_clean["mse"],
            "validation/final_psnr_db": val_clean["psnr_db_mean_per_image"],
            "validation/final_ssim": val_clean["ssim_mean"],
            "test/mse": test_clean["mse"],
            "test/psnr_db": test_clean["psnr_db_mean_per_image"],
            "test/ssim": test_clean["ssim_mean"],
            "oracle/ranking": oracle["ranking_channels_most_to_least_important"],
        })
        _log_table(wandb, "oracle/channel_scores", oracle_rows)
        _log_table(wandb, "rs_uep/validation_sweep", rs_validation_rows)
        _log_table(wandb, "rs_uep/test_sweep", rs_rows)
        artifact = wandb.Artifact("adjsccq-train-split-checkpoints", type="model")
        artifact.add_file(best_path)
        artifact.add_file(final_path)
        artifact.add_file(oracle_path)
        run.log_artifact(artifact)
        wandb.finish()
    print(json.dumps({
        "best_checkpoint": best_path,
        "final_checkpoint": final_path,
        "test": test_clean,
        "oracle_ranking": oracle["ranking_channels_most_to_least_important"],
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data_dir", default="./cifar10_data")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--oracle_max_images", type=int, default=None)
    parser.add_argument("--eval_max_images", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb_name", default=None)
    args = parser.parse_args()
    main(
        args.config, args.data_dir, args.output_dir, args.val_size,
        args.oracle_max_images, args.eval_max_images, args.wandb,
        args.wandb_project, args.wandb_name,
    )
