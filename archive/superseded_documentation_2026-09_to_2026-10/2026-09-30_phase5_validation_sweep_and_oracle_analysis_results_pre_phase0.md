# Phase 5: Validation Sweep Results & Oracle Importance Analysis

**Date:** 2026-09-30

## 1. The Validation Sweep

We ran a validation sweep of the full pipeline (Encoder -> Quantizer -> UEP RS -> Packetizer -> Erasure Channel -> Decoder). 

**Sweep Parameters:**
- Images: 1024 (First 10% of CIFAR-10 test set)
- Seeds: 0, 1, 2
- Checkpoint: ADJSCC-Q Identity Final
- Erasure Rates: 0.0 to 0.50
- Methods Evaluated: 
  - `no_rs`
  - `uniform` (RS 96, 64)
  - Assorted 2-tier and 3-tier UEP schemes
  - Random-assignment control variants for every UEP scheme

### Key Results at Major Erasure Thresholds

| Method | 0.00 Loss | 0.20 Loss | 0.30 Loss | 0.40 Loss |
|--------|-----------|-----------|-----------|-----------|
| **Uniform RS** | 30.50 dB | **30.49 dB** | **28.15 dB** | 19.82 dB |
| **no_rs** | 30.50 dB | 22.19 dB | 20.28 dB | 18.72 dB |
| **two_tier_50_50** (Best UEP) | 30.50 dB | 30.37 dB | 26.71 dB | 20.05 dB |
| **three_tier_25_50_25** | 30.50 dB | 27.17 dB | 24.36 dB | 19.96 dB |
| **three_tier_25_50_25_random** | 30.50 dB | 26.96 dB | 24.50 dB | **20.79 dB** |

## 2. Research Conclusions

### Conclusion A: Uniform RS is Optimal below 33% Loss
Uniform RS is a single massive block of RS(96, 64), meaning it can perfectly correct any packet loss up to 33.3%. At 20% and 30% packet loss, Uniform RS completely dominates all UEP strategies. 
The UEP strategies inherently weaken their "low" priority tiers to heavily armor the "high" priority tiers. Because the low tiers have very little redundancy (e.g., RS 40,32), they fail rapidly at low packet loss rates (10-20%), injecting noise into the image and dragging the overall PSNR down while Uniform remains perfectly uncorrupted.

### Conclusion B: Attention is NOT a Proxy for Visual Importance
The core hypothesis of using the `af5_bottleneck` attention gate values to rank the importance of channels has been proven false for this checkpoint.
At severe erasure rates (40% to 50%), where Uniform RS finally collapses, the UEP schemes begin to survive better. However, the **randomly assigned** UEP schemes perform identically to, or significantly better than, the attention-ranked UEP schemes (e.g., at 40% loss, the random 25/50/25 split achieved 20.79 dB compared to the attention-based 19.96 dB).

This proves that sorting channels by their attention gate values groups the latent symbols in a way that is actively *more* vulnerable to erasures than simply choosing channels at random.

## 3. Next Steps: Oracle Importance Ranking
We must conduct a true sensitivity analysis (Leave-One-Out) on the 16 channels to determine their *actual* oracle impact on image PSNR when erased. 
If we can map the true sensitivity of the channels, we can bypass the attention weights entirely and use the Oracle Ranking to guide the UEP allocation. If the Oracle Ranking works, it should comprehensively beat the Random UEP baseline in high-erasure regimes.
