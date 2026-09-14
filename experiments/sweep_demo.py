import argparse
import os

import matplotlib.pyplot as plt
import torch
import numpy as np

from data.cifar10 import get_cifar10_loaders
from models.channel import AWGNChannel
from experiments.toy_demo import build_model, load_checkpoint_if_available, run_pipeline, psnr_per_image, ssim

def main():
    parser = argparse.ArgumentParser(description="Sweep SNR and Erasure Rate to evaluate PSNR and SSIM.")
    parser.add_argument("--model", choices=["adjscc", "deepjscc"], default="adjscc")
    parser.add_argument("--rs_protection", choices=["none", "uniform"], default="none")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_images", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out", type=str, default="results/toy_demo/sweep_results1.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Use a larger batch of N images to get a reliable average for the sweep
    _, test_loader = get_cifar10_loaders(data_dir=args.data_dir, batch_size=args.n_images, num_workers=0)
    x, _ = next(iter(test_loader))
    x = x[: args.n_images].to(device)

    encoder, decoder = build_model(args.model, device)
    encoder.eval()
    decoder.eval()
    
    # Load checkpoint
    has_checkpoint = load_checkpoint_if_available(encoder, decoder, args.model, args.checkpoint, device)
    
    channel = AWGNChannel().to(device)

    # Define sweep values
    snr_vals = [0.0, 5.0, 10.0, 15.0, 20.0]
    erasure_vals = [0.0, 0.1, 0.2, 0.3, 0.4]

    results_psnr = np.zeros((len(erasure_vals), len(snr_vals)))
    results_ssim = np.zeros((len(erasure_vals), len(snr_vals)))

    print(f"\nRunning sweep: model={args.model} | rs_protection={args.rs_protection} | n_images={args.n_images}\n")

    for i, er in enumerate(erasure_vals):
        for j, snr in enumerate(snr_vals):
            # Reset generator per run so random erasures are reproducible
            erasure_generator = torch.Generator().manual_seed(args.seed + j)
            
            x_hat, _ = run_pipeline(
                encoder, decoder, channel, x, snr, er, args.rs_protection, args.model, erasure_generator
            )
            
            psnr_vals = psnr_per_image(x_hat, x)
            ssim_vals = ssim(x_hat, x)
            
            results_psnr[i, j] = psnr_vals.mean().item()
            results_ssim[i, j] = ssim_vals.mean().item()
            
            print(f"SNR {snr:4.1f}dB, Erasure {er:.1f} -> PSNR: {results_psnr[i, j]:5.2f}dB, SSIM: {results_ssim[i, j]:.3f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i, er in enumerate(erasure_vals):
        ax1.plot(snr_vals, results_psnr[i, :], marker='o', label=f"Erasure {er}")
        ax2.plot(snr_vals, results_ssim[i, :], marker='o', label=f"Erasure {er}")

    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("PSNR vs SNR")
    ax1.grid(True)
    ax1.legend(title="Erasure Rate")

    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("SSIM")
    ax2.set_title("SSIM vs SNR")
    ax2.grid(True)
    ax2.legend(title="Erasure Rate")

    fig.suptitle(f"Sweep Results (Model: {args.model}, Protection: {args.rs_protection})", fontsize=14)
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved sweep plot to {args.out}")

if __name__ == "__main__":
    main()
