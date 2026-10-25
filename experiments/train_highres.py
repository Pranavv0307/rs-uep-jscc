"""Step C.1 fallback: fine-tune the same architecture on high-resolution patches.

Use this only if ``eval_kodak`` reports a clean PSNR well below the CIFAR level.
The architecture is unchanged (fully convolutional), so the CIFAR checkpoint is
a valid initialisation; training sees random ``--crop`` x ``--crop`` patches
from any folder of PNG/JPG images (e.g. COCO val2017, DIV2K, CLIC, ImageNet
subset -- you provide the folder; it must not contain Kodak images).

Holds out 2% of the images (by sorted path, deterministic) as validation for
checkpoint selection, and computes the oracle ranking on the training patches.
Output: ``best_val.pth`` + ``oracle_ranking.json`` (``source_split`` =
``highres_train``), in the same format every other script reads.

    python -m experiments.train_highres --init_checkpoint <cifar ckpt> --train_dir D:/coco/val2017 \\
        --output_dir results/phase3/highres --epochs 20 --crop 256
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.datasets import FolderImageDataset
from experiments import plotting, wandb_utils
from experiments.metrics import psnr_per_image
from experiments.model_io import build_models, load_model_bundle
from experiments.reproducibility import build_provenance, seed_everything, seed_worker, write_json
from experiments.sensitivity import damage_for_tensor, ranking_from_damage


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument("--oracle_images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    base = load_model_bundle(args.init_checkpoint, device)
    cfg = copy.deepcopy(base.config)
    cfg["model"]["image_size"] = args.crop  # decoder spatial grid for the crop size
    encoder, decoder = build_models(cfg, device)
    encoder.load_state_dict(base.encoder.state_dict())
    decoder.load_state_dict(base.decoder.state_dict())
    snr_db = base.snr_db
    encoder.quantizer.set_sigma(cfg["quantizer"]["sigma_end"])

    dataset = FolderImageDataset(args.train_dir, args.crop, train=True)
    val_dataset = FolderImageDataset(args.train_dir, args.crop, train=False)
    n_val = max(1, int(len(dataset) * args.val_fraction))
    train_idx, val_idx = list(range(len(dataset) - n_val)), list(range(len(dataset) - n_val, len(dataset)))
    generator = torch.Generator().manual_seed(args.seed + 1)
    loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, generator=generator,
                        num_workers=args.num_workers, drop_last=True, worker_init_fn=seed_worker)
    val_images = torch.stack([val_dataset[i][0] for i in val_idx])
    print(f"{len(train_idx)} train images, {len(val_idx)} validation images, crop {args.crop}")

    run = wandb_utils.init_run(args.wandb, args.wandb_project, "train-highres", vars(args), job_type="train",
                               group="phase3")
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    criterion = nn.MSELoss()

    @torch.no_grad()
    def validate():
        encoder.eval()
        decoder.eval()
        values = []
        for start in range(0, len(val_images), 8):
            x = val_images[start:start + 8].to(device)
            _, idx = encoder(x, snr_db, return_indices=True)
            recon = decoder(encoder.quantizer.dequantize_indices(idx), snr_db)
            values.append(psnr_per_image(recon, x).cpu())
        return float(torch.cat(values).mean())

    history = [{"epoch": 0, "val_psnr_db": validate()}]
    best = {"psnr": history[0]["val_psnr_db"], "epoch": 0, "state": copy.deepcopy(
        {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()})}
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        total, batches = 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            loss = criterion(decoder(encoder(x, snr_db), snr_db), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        value = validate()
        history.append({"epoch": epoch, "val_psnr_db": value, "train_mse": total / batches})
        wandb_utils.log_metrics(run, {"epoch": epoch, "train/mse": total / batches, "val/psnr_db": value})
        print(f"epoch {epoch:2d} | train mse {total / batches:.5f} | val PSNR {value:.2f} dB")
        if value > best["psnr"]:
            best = {"psnr": value, "epoch": epoch, "state": copy.deepcopy(
                {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()})}

    path = os.path.join(args.output_dir, "best_val.pth")
    torch.save({**best["state"], "config": cfg, "epoch": best["epoch"],
                "split": {"seed": base.split_seed, "highres_train_dir": args.train_dir},
                "validation_metrics": {"psnr_db_mean_per_image": best["psnr"]},
                "init_checkpoint": args.init_checkpoint}, path)

    selected = load_model_bundle(path, device)
    patches = torch.stack([dataset[i][0] for i in train_idx[:args.oracle_images]])
    labels = damage_for_tensor(selected, patches, 4)
    ranking = ranking_from_damage(labels["damage"])
    write_json(os.path.join(args.output_dir, "oracle_ranking.json"), {
        "checkpoint": path, "source_split": "highres_train", "method": "oracle_leave_one_channel_out",
        "ranking_channels_most_to_least_important": ranking,
        "psnr_drop_db_by_channel": labels["damage"].mean(axis=0).tolist(), "num_images": len(patches),
        "channel_count": selected.channels, "crop": args.crop})
    write_json(os.path.join(args.output_dir, "metrics.json"), {
        "metadata": build_provenance(args=args, checkpoint_path=path, global_seed=args.seed, device=device),
        "history": history, "best_epoch": best["epoch"], "best_val_psnr_db": best["psnr"], "ranking": ranking})
    figures = plotting.plot_lines({"validation PSNR": [h["val_psnr_db"] for h in history]},
                                  list(range(len(history))), os.path.join(args.output_dir, "val_psnr"),
                                  xlabel="Epoch", ylabel="Validation PSNR (dB)")
    wandb_utils.log_images(run, "figures", figures)
    wandb_utils.set_summary(run, {"best_val_psnr_db": best["psnr"], "best_epoch": best["epoch"]})
    wandb_utils.finish(run)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
