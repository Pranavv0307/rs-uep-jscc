"""Side-by-side visual inspection tool for Deep JSCC and RS-UEP.

This script strictly calls existing models and pipeline modules. It does NOT
implement or re-define any model architecture, quantizer, Reed-Solomon,
packetization, or channel erasure logic.

Outputs a multi-column visual comparison grid:
  Col 1: Original Ground Truth
  Col 2: Clean Reconstruction (0% Loss)
  Col 3: No RS Protection (at target loss)
  Col 4: Uniform RS (Equal Error Protection at target loss)
  Col 5: Sensitivity-Aware UEP (Proposed at target loss)

Usage::

    # Visualize 5 test images under 40% packet loss
    python -m experiments.visualize_reconstructions --erasure_rate 0.40 --num_images 5

    # Customize output path and seed
    python -m experiments.visualize_reconstructions --erasure_rate 0.35 --seed 100 --output results/eval/visual_check.png
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt

from data.cifar10 import get_cifar10_loaders
from experiments.eval_adjsccq_clean import DEFAULT_CHECKPOINT, load_adjsccq_checkpoint
from experiments.metrics import psnr_per_image, ssim
from coding.erasure_channel import sample_erasure_mask, apply_erasure_mask
from coding.rs_pipeline import (
    NO_RS,
    SCHEME_CANDIDATES,
    encode_blocks,
    decode_blocks,
    dequantize_with_fallback,
)

DEFAULT_ORACLE_RANKING = [3, 7, 4, 6, 11, 0, 12, 8, 1, 15, 14, 13, 9, 10, 5, 2]


@torch.no_grad()
def run_visual_comparison(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load validated model
    encoder, decoder, config = load_adjsccq_checkpoint(args.checkpoint, device)
    encoder.eval()
    decoder.eval()
    snr_db = config["af_conditioning"]["nominal_snr_db"]

    # 2. Load test images
    _, test_loader = get_cifar10_loaders(args.data_dir, batch_size=args.num_images, num_workers=0)
    x, _ = next(iter(test_loader))
    x = x[:args.num_images].to(device)
    batch_size = x.shape[0]

    # 3. Clean quantization and forward pass
    _, idx = encoder(x, snr_db, return_attn=False, return_indices=True)
    clean_latent = encoder.quantizer.dequantize_indices(idx)
    x_clean = decoder(clean_latent, snr_db).clamp(0.0, 1.0)

    # 4. Define channel order and loss masks
    oracle_order = torch.tensor(DEFAULT_ORACLE_RANKING, device=device).unsqueeze(0).expand(batch_size, -1)
    
    mask_gen = torch.Generator().manual_seed(args.seed)
    mask_96 = sample_erasure_mask(batch_size, 96, args.erasure_rate, mask_gen)
    mask_64 = sample_erasure_mask(batch_size, 64, args.erasure_rate, mask_gen)

    # 5. Method A: No RS
    no_rs_packets = encode_blocks(idx, NO_RS, None)
    no_rs_received = apply_erasure_mask(no_rs_packets, mask_64).received
    no_rs_decoded = decode_blocks(no_rs_received, NO_RS, None)
    no_rs_latent = dequantize_with_fallback(encoder.quantizer, no_rs_decoded.indices, no_rs_decoded.failed)
    x_no_rs = decoder(no_rs_latent, snr_db).clamp(0.0, 1.0)

    # 6. Method B: Uniform RS
    uniform_scheme = SCHEME_CANDIDATES["uniform"]
    uniform_packets = encode_blocks(idx, uniform_scheme, None)
    uniform_received = apply_erasure_mask(uniform_packets, mask_96).received
    uniform_decoded = decode_blocks(uniform_received, uniform_scheme, None)
    uniform_latent = dequantize_with_fallback(encoder.quantizer, uniform_decoded.indices, uniform_decoded.failed)
    x_uniform = decoder(uniform_latent, snr_db).clamp(0.0, 1.0)

    # 7. Method C: Sensitivity-Aware UEP (best scheme: three_tier_25_50_25)
    uep_scheme = SCHEME_CANDIDATES[args.uep_scheme]
    uep_packets = encode_blocks(idx, uep_scheme, oracle_order)
    uep_received = apply_erasure_mask(uep_packets, mask_96).received
    uep_decoded = decode_blocks(uep_received, uep_scheme, oracle_order)
    uep_latent = dequantize_with_fallback(encoder.quantizer, uep_decoded.indices, uep_decoded.failed)
    x_uep = decoder(uep_latent, snr_db).clamp(0.0, 1.0)

    # 8. Compute per-image metrics
    psnr_clean = psnr_per_image(x_clean, x)
    ssim_clean = ssim(x_clean, x)

    psnr_no_rs = psnr_per_image(x_no_rs, x)
    ssim_no_rs = ssim(x_no_rs, x)

    psnr_uniform = psnr_per_image(x_uniform, x)
    ssim_uniform = ssim(x_uniform, x)

    psnr_uep = psnr_per_image(x_uep, x)
    ssim_uep = ssim(x_uep, x)

    # 9. Plotting side-by-side comparison
    columns = [
        ("Original", x, None, None),
        ("Clean (0% Loss)", x_clean, psnr_clean, ssim_clean),
        (f"No RS ({int(args.erasure_rate*100)}% Loss)", x_no_rs, psnr_no_rs, ssim_no_rs),
        (f"Uniform RS ({int(args.erasure_rate*100)}% Loss)", x_uniform, psnr_uniform, ssim_uniform),
        (f"Sensitivity UEP ({int(args.erasure_rate*100)}% Loss)", x_uep, psnr_uep, ssim_uep),
    ]

    fig, axes = plt.subplots(batch_size, len(columns), figsize=(3.2 * len(columns), 3.4 * batch_size))
    if batch_size == 1:
        axes = [axes]

    for row in range(batch_size):
        for col, (title, img_tensor, psnr_vals, ssim_vals) in enumerate(columns):
            ax = axes[row][col]
            img = img_tensor[row].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

            if psnr_vals is not None:
                p = psnr_vals[row].item()
                s = ssim_vals[row].item()
                ax.set_xlabel(f"{p:.2f} dB | {s:.3f}", fontsize=11, fontweight="semibold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"\nVisual comparison figure successfully saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Side-by-side visual inspection of JSCC reconstructions.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to trained ADJSCC-Q checkpoint.")
    parser.add_argument("--data_dir", default="./cifar10_data", help="Path to CIFAR-10 data directory.")
    parser.add_argument("--num_images", type=int, default=5, help="Number of sample images to inspect side-by-side.")
    parser.add_argument("--erasure_rate", type=float, default=0.40, help="Packet erasure rate (e.g. 0.35, 0.40, 0.50).")
    parser.add_argument("--uep_scheme", default="three_tier_25_50_25", help="Candidate UEP scheme name.")
    parser.add_argument("--seed", type=int, default=42, help="Channel erasure random seed.")
    parser.add_argument("--output", default="results/eval/visual_comparison_side_by_side.png", help="Output figure path.")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution DPI for saved figure.")
    args = parser.parse_args()

    run_visual_comparison(args)


if __name__ == "__main__":
    main()
