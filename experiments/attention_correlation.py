import argparse
import os

import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import pearsonr

from data.cifar10 import get_cifar10_loaders
from models.channel import AWGNChannel
from experiments.toy_demo import build_model, load_checkpoint_if_available, psnr_per_image, NUM_ERASURE_CHUNKS

def main():
    parser = argparse.ArgumentParser(description="Correlate Attention with PSNR drop.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_images", type=int, default=128)
    parser.add_argument("--snr_db", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out", type=str, default="results/toy_demo/attention_correlation.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    _, test_loader = get_cifar10_loaders(data_dir=args.data_dir, batch_size=args.n_images, num_workers=0)
    x, _ = next(iter(test_loader))
    x = x[: args.n_images].to(device)

    # Setup Model
    encoder, decoder = build_model("adjscc", device)
    encoder.eval()
    decoder.eval()
    load_checkpoint_if_available(encoder, decoder, "adjscc", args.checkpoint, device)
    
    # We fix the generator for the channel so noise is identical across passes
    # This isolates the effect of the erasure!
    channel_generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # The AWGNChannel in toy_demo might not take a generator. Let's subclass or override briefly
    # Or simply run the channel with a large fixed seed so the noise added is the same
    # But actually AWGN channel just adds randn. If we don't control the seed, the PSNR drop
    # will have variance due to channel noise. We can freeze the seed before channel calls.

    print("Running baseline (no erasures)...")
    with torch.no_grad():
        z, attn = encoder(x, args.snr_db, return_attn=True)
        
        torch.manual_seed(args.seed)
        z_noisy_base = z + torch.randn_like(z) * (10 ** (-args.snr_db / 20.0))
        
        x_hat_base = decoder(z_noisy_base, args.snr_db)
        psnr_base = psnr_per_image(x_hat_base, x) # Shape: (B,)
    
    gate = attn["af5_bottleneck"].squeeze(-1).squeeze(-1) # Shape: (B, c_last)
    
    B, L = z.shape
    spatial_sq = L // gate.shape[1] 
    num_chunks = min(NUM_ERASURE_CHUNKS, L)
    bounds = torch.linspace(0, L, num_chunks + 1).round().long()

    attention_scores = []
    psnr_drops = []

    print(f"Evaluating {num_chunks} chunks to compute PSNR drop...")
    for c in range(num_chunks):
        lo, hi = bounds[c].item(), bounds[c+1].item()
        if hi <= lo: continue
        
        # Calculate the average attention score for this chunk for each image
        # Each index `i` from lo to hi-1 corresponds to channel `i // spatial_sq`
        channels_for_chunk = torch.arange(lo, hi) // spatial_sq
        chunk_attn = gate[:, channels_for_chunk].mean(dim=1) # Shape: (B,)
        
        # Zero out the chunk
        z_erased = z.clone()
        z_erased[:, lo:hi] = 0.0
        
        # Decode with SAME noise
        with torch.no_grad():
            torch.manual_seed(args.seed)
            z_noisy = z_erased + torch.randn_like(z_erased) * (10 ** (-args.snr_db / 20.0))
            
            x_hat_erased = decoder(z_noisy, args.snr_db)
            psnr_erased = psnr_per_image(x_hat_erased, x)
            
        drop = psnr_base - psnr_erased
        
        attention_scores.extend(chunk_attn.cpu().numpy())
        psnr_drops.extend(drop.cpu().numpy())

    attention_scores = np.array(attention_scores)
    psnr_drops = np.array(psnr_drops)
    
    rho, pval = pearsonr(attention_scores, psnr_drops)
    print(f"\nCorrelation between Attention and PSNR Drop: rho = {rho:.4f} (p-value: {pval:.4e})")

    # Plot
    plt.figure(figsize=(9, 6))
    plt.scatter(attention_scores, psnr_drops, alpha=0.4, edgecolors='none', color='tab:blue', s=25)
    plt.xlabel("Attention Gate Value (Averaged over chunk)")
    plt.ylabel("Reconstruction Degradation / PSNR Drop (dB)")
    plt.title(f"Reconstruction Damage vs Attention (SNR={args.snr_db}dB)\n"
              f"Pearson Correlation: $\\rho$ = {rho:.3f}", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add a trend line
    m, b = np.polyfit(attention_scores, psnr_drops, 1)
    x_range = np.linspace(attention_scores.min(), attention_scores.max(), 100)
    plt.plot(x_range, m*x_range + b, color='tab:red', linestyle='--', linewidth=2, label="Linear Trend")
    plt.legend()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved correlation plot to {args.out}")

if __name__ == "__main__":
    main()
