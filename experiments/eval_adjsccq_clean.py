"""
Clean (no-erasure) reference evaluation of the trained ADJSCC-Q checkpoint.

Every packet-erasure result is compared against this number, so it is
measured on the exact path the RS pipeline uses:

    image -> ADJSCCQEncoder -> byte indices -> dequantize_indices -> decoder

rather than decoding the encoder's float z_q directly. The script also
checks that both paths give the same reconstruction, which is the index
contract the RS pipeline depends on.

Usage:
    python -m experiments.eval_adjsccq_clean
    python -m experiments.eval_adjsccq_clean --max_images 1000
"""
import argparse
import json
import os

import torch

from data.cifar10 import get_cifar10_loaders
from experiments.metrics import psnr_per_image, ssim
from models.adjsccq import NOMINAL_SNR_DB_IDENTITY, ADJSCCQDecoder, ADJSCCQEncoder

DEFAULT_CHECKPOINT = "results/checkpoints/adjsccq_identity_train/adjsccq_identity_final.pth"


def load_adjsccq_checkpoint(path: str, device: torch.device):
    """Rebuild encoder/decoder from a train_adjsccq_identity checkpoint.

    Returns ``(encoder, decoder, config)`` in eval mode.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    encoder = ADJSCCQEncoder(
        k_over_n=cfg["model"]["k_over_n"],
        image_size=cfg["model"]["image_size"],
        num_levels=cfg["quantizer"]["num_levels"],
    ).to(device)
    decoder = ADJSCCQDecoder(k=encoder.k, image_size=cfg["model"]["image_size"]).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, cfg


@torch.no_grad()
def evaluate_clean(encoder, decoder, loader, snr_db: float, device, max_images: int = None):
    psnr_values, ssim_values = [], []
    squared_error_sum, pixel_count = 0.0, 0
    level_counts = torch.zeros(encoder.quantizer.num_levels, dtype=torch.long)
    max_path_gap = 0.0

    for x, _ in loader:
        if max_images is not None:
            remaining = max_images - len(psnr_values)
            if remaining <= 0:
                break
            x = x[:remaining]
        x = x.to(device)

        z_q, idx = encoder(x, snr_db, return_indices=True)
        x_hat = decoder(encoder.quantizer.dequantize_indices(idx), snr_db)
        max_path_gap = max(max_path_gap, (decoder(z_q, snr_db) - x_hat).abs().max().item())

        psnr_values.extend(psnr_per_image(x_hat, x).tolist())
        ssim_values.extend(ssim(x_hat, x).tolist())
        squared_error_sum += ((x_hat - x) ** 2).sum().item()
        pixel_count += x.numel()
        level_counts += torch.bincount(idx.flatten().cpu(), minlength=level_counts.numel())

    psnr_tensor = torch.tensor(psnr_values)
    ssim_tensor = torch.tensor(ssim_values)
    dataset_mse = squared_error_sum / pixel_count
    return {
        "num_images": len(psnr_values),
        "snr_db_af_conditioning": snr_db,
        "psnr_db_mean_per_image": psnr_tensor.mean().item(),
        "psnr_db_std_per_image": psnr_tensor.std().item(),
        "psnr_db_from_dataset_mse": (10 * torch.log10(torch.tensor(1.0 / dataset_mse))).item(),
        "ssim_mean": ssim_tensor.mean().item(),
        "ssim_std": ssim_tensor.std().item(),
        "codebook_levels_used": int((level_counts > 0).sum()),
        "codebook_levels_total": level_counts.numel(),
        "index_path_max_abs_gap": max_path_gap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_dir", default="./cifar10_data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=None, help="default: full 10,000-image test set")
    parser.add_argument("--output", default="results/eval/adjsccq_clean.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, cfg = load_adjsccq_checkpoint(args.checkpoint, device)
    snr_db = cfg.get("af_conditioning", {}).get("nominal_snr_db", NOMINAL_SNR_DB_IDENTITY)
    _, test_loader = get_cifar10_loaders(args.data_dir, batch_size=args.batch_size, num_workers=0)

    results = evaluate_clean(encoder, decoder, test_loader, snr_db, device, args.max_images)
    results.update({"checkpoint": args.checkpoint, "device": str(device), "torch": torch.__version__})

    for key, value in results.items():
        print(f"{key:28s} {value}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
