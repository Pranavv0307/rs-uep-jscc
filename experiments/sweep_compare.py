import argparse
import os

import matplotlib.pyplot as plt
import torch
import numpy as np

from data.cifar10 import get_cifar10_loaders
from models.channel import AWGNChannel
from experiments.toy_demo import build_model, load_checkpoint_if_available, run_pipeline, psnr_per_image, ssim

def main():
    parser = argparse.ArgumentParser(description="Compare 'none' and 'uniform' RS protection.")
    parser.add_argument("--model", choices=["adjscc", "deepjscc"], default="adjscc")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_images", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out", type=str, default="results/toy_demo/sweep_compare.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    _, test_loader = get_cifar10_loaders(data_dir=args.data_dir, batch_size=args.n_images, num_workers=0)
    x, _ = next(iter(test_loader))
    x = x[: args.n_images].to(device)

    encoder, decoder = build_model(args.model, device)
    encoder.eval()
    decoder.eval()
    load_checkpoint_if_available(encoder, decoder, args.model, args.checkpoint, device)
    
    channel = AWGNChannel().to(device)

    snr_vals = [0.0, 5.0, 10.0, 15.0, 20.0]
    # Restrict to two erasure rates to keep the plot readable
    erasure_vals = [0.1, 0.3] 

    results_psnr = {"none": np.zeros((len(erasure_vals), len(snr_vals))),
                    "uniform": np.zeros((len(erasure_vals), len(snr_vals)))}
    results_ssim = {"none": np.zeros((len(erasure_vals), len(snr_vals))),
                    "uniform": np.zeros((len(erasure_vals), len(snr_vals)))}

    print(f"\nRunning comparison sweep: model={args.model} | n_images={args.n_images}\n")

    for prot in ["none", "uniform"]:
        for i, er in enumerate(erasure_vals):
            for j, snr in enumerate(snr_vals):
                # Reset generator to ensure same erasures happen for both schemes
                erasure_generator = torch.Generator().manual_seed(args.seed + j)
                x_hat, _ = run_pipeline(
                    encoder, decoder, channel, x, snr, er, prot, args.model, erasure_generator
                )
                
                psnr_vals = psnr_per_image(x_hat, x)
                ssim_vals = ssim(x_hat, x)
                
                results_psnr[prot][i, j] = psnr_vals.mean().item()
                results_ssim[prot][i, j] = ssim_vals.mean().item()
                
                print(f"[{prot:7}] SNR {snr:4.1f}dB, Erasure {er:.1f} -> PSNR: {results_psnr[prot][i, j]:5.2f}dB")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = ['tab:blue', 'tab:orange']
    for i, er in enumerate(erasure_vals):
        ax1.plot(snr_vals, results_psnr["none"][i, :], marker='o', linestyle='-', color=colors[i], label=f"none (ER={er})")
        ax1.plot(snr_vals, results_psnr["uniform"][i, :], marker='x', linestyle='--', color=colors[i], label=f"uniform (ER={er})")
        
        ax2.plot(snr_vals, results_ssim["none"][i, :], marker='o', linestyle='-', color=colors[i], label=f"none (ER={er})")
        ax2.plot(snr_vals, results_ssim["uniform"][i, :], marker='x', linestyle='--', color=colors[i], label=f"uniform (ER={er})")

    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("PSNR vs SNR")
    ax1.grid(True)
    ax1.legend(loc="lower right")

    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("SSIM")
    ax2.set_title("SSIM vs SNR")
    ax2.grid(True)
    ax2.legend(loc="lower right")

    fig.suptitle(f"Comparison: None vs Uniform Protection", fontsize=14)
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to {args.out}")

if __name__ == "__main__":
    main()
