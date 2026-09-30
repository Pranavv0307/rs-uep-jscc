import argparse
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict
import json

from data.cifar10 import get_cifar10_loaders
from experiments.metrics import psnr_per_image
from experiments.eval_adjsccq_clean import load_adjsccq_checkpoint, DEFAULT_CHECKPOINT
from coding.rs_pipeline import dequantize_with_fallback
from tqdm import tqdm


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Correlate Attention with True PSNR drop (Oracle Importance).")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max_images", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out", type=str, default="results/eval/oracle_attention_correlation.png")
    parser.add_argument("--ranking_out", type=str, default="results/eval/oracle_ranking.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ADJSCC-Q
    encoder, decoder, config = load_adjsccq_checkpoint(args.checkpoint, device)
    snr_db = config["af_conditioning"]["nominal_snr_db"]
    num_channels = encoder.k  # Should be 16

    _, loader = get_cifar10_loaders(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=0)

    attention_scores = []
    psnr_drops = []
    
    # Track the global drop for each channel across all images to compute the true average Oracle Rank
    channel_total_drops = defaultdict(float)
    channel_total_attention = defaultdict(float)
    total_images_processed = 0

    print("Running Leave-One-Out Sensitivity Analysis on the Latent Channels...")

    for batch_index, (x, _) in enumerate(tqdm(loader, desc="Batches")):
        if args.max_images is not None:
            remaining = args.max_images - total_images_processed
            if remaining <= 0:
                break
            x = x[:remaining]
            
        x = x.to(device)
        B = x.shape[0]
        total_images_processed += B

        # 1. Clean Encode/Decode (Baseline)
        _, attn, indices = encoder(x, snr_db, return_attn=True, return_indices=True)
        # Attention shape: (B, C, 1, 1) -> (B, C)
        gate = attn["af5_bottleneck"].squeeze(-1).squeeze(-1)
        
        num_channels = gate.shape[1]
        
        # Dequantize with no erasures
        latent_clean = dequantize_with_fallback(
            encoder.quantizer, indices, torch.zeros_like(indices, dtype=torch.bool)
        )
        x_hat_clean = decoder(latent_clean, snr_db)
        psnr_clean = psnr_per_image(x_hat_clean, x) # (B,)

        # 2. Leave-One-Out Evaluation
        B, L = latent_clean.shape
        spatial_sq = L // num_channels
        
        for c in range(num_channels):
            # Clone clean latent and erase only channel 'c'
            latent_erased = latent_clean.clone()
            latent_erased[:, c * spatial_sq : (c+1) * spatial_sq] = 0.0  # Fallback value for erasures
            
            x_hat_erased = decoder(latent_erased, snr_db)
            psnr_erased = psnr_per_image(x_hat_erased, x) # (B,)
            
            drop = psnr_clean - psnr_erased # (B,) positive value means PSNR dropped (damage)
            
            attention_scores.extend(gate[:, c].cpu().numpy())
            psnr_drops.extend(drop.cpu().numpy())
            
            channel_total_drops[c] += drop.sum().item()
            channel_total_attention[c] += gate[:, c].sum().item()

    # Calculate correlation
    attention_scores = np.array(attention_scores)
    psnr_drops = np.array(psnr_drops)
    
    rho, pval = pearsonr(attention_scores, psnr_drops)
    print(f"\nImage-level Correlation between Attention and PSNR Drop: rho = {rho:.4f} (p-value: {pval:.4e})")
    
    # Calculate global channel rankings
    avg_drops = {c: channel_total_drops[c] / total_images_processed for c in range(num_channels)}
    avg_attns = {c: channel_total_attention[c] / total_images_processed for c in range(num_channels)}
    
    # Sort channels by actual damage caused when erased (Oracle Importance)
    oracle_ranking = sorted(range(num_channels), key=lambda c: avg_drops[c], reverse=True)
    # Sort channels by their attention values
    attention_ranking = sorted(range(num_channels), key=lambda c: avg_attns[c], reverse=True)

    print("\n================== RANKINGS ==================")
    print("Channel | Oracle Rank | Attention Rank | Avg PSNR Drop | Avg Attention")
    print("-" * 75)
    for c in range(num_channels):
        o_rank = oracle_ranking.index(c) + 1
        a_rank = attention_ranking.index(c) + 1
        print(f"   {c:02d}   |      {o_rank:02d}     |       {a_rank:02d}       |    {avg_drops[c]:.4f} dB   |    {avg_attns[c]:.4f}")

    # Plot Image-level scatter
    plt.figure(figsize=(9, 6))
    plt.scatter(attention_scores, psnr_drops, alpha=0.15, edgecolors='none', color='tab:blue', s=25)
    plt.xlabel("Attention Gate Value")
    plt.ylabel("Reconstruction Damage / PSNR Drop (dB)")
    plt.title(f"Reconstruction Damage vs Attention\nPearson Correlation: $\\rho$ = {rho:.3f}", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add a trend line
    m, b = np.polyfit(attention_scores, psnr_drops, 1)
    x_range = np.linspace(attention_scores.min(), attention_scores.max(), 100)
    plt.plot(x_range, m*x_range + b, color='tab:red', linestyle='--', linewidth=2, label="Linear Trend")
    plt.legend()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved correlation plot to {args.out}")
    
    # Save Oracle Ranking explicitly so it can be imported by the evaluation script
    os.makedirs(os.path.dirname(args.ranking_out), exist_ok=True)
    with open(args.ranking_out, "w") as f:
        json.dump({
            "oracle_ranking_channels": oracle_ranking,
            "average_psnr_drops": avg_drops
        }, f, indent=4)
    print(f"Saved exact Oracle Ranking to {args.ranking_out}")

if __name__ == "__main__":
    main()
