# Insights (generated from the result files, not hand-written)

## Main erasure sweep

- Based on 5 channel seed(s); 10000 images per seed. All gains below are paired per seed against the baseline (same packet losses), so they carry no luck-of-the-draw advantage.
- Break-even: the best protected method overtakes `uniform` from about erasure_rate = 0.32.
- Low-loss penalty: even the best method is 0.87 dB below `uniform` at erasure_rate = 0.3 (three_tier_25_50_25@predicted). This is the block-length effect of a single large RS block, which tiered coding gives up.
- Peak gain: +3.00 dB at erasure_rate = 0.4 by three_tier_25_50_25@predicted (95% CI ±0.03, significant).
- Importance sources ranked by their best scheme's mean gain over erasure_rate >= 0.35: predicted +2.31 dB, energy +2.24 dB, l1 +2.22 dB, oracle +2.19 dB, random +0.50 dB, attention +0.31 dB.
- Share of the reference ranking's gain captured: predicted 106%, energy 102%, l1 101%, random 23%, attention 14%. A share above 100% means that source allocated protection better than the reference.
- Random-order control gains +0.50 dB versus oracle +2.19 dB, so the benefit comes from *which* channels are protected, not merely from tiering.
- Worst-case images at erasure_rate = 0.4: 5th-percentile PSNR 18.74 dB (three_tier_25_50_25@predicted) vs 16.66 dB (uniform).
- Worst-case images at erasure_rate = 0.5: 5th-percentile PSNR 15.90 dB (three_tier_25_25_50@predicted) vs 15.40 dB (uniform).
- Seed sensitivity: std across channel seeds stays below 0.3 dB everywhere.
- `uniform` shows its steepest fall (-4.27 dB) between erasure_rate = 0.3 and 0.35 -- its cliff.

## Model comparison (low-loss penalty, peak gain)

- base: worst best-method gain -0.87 dB, peak gain +3.00 dB, break-even at loss 0.32.
- robust_channel: worst best-method gain -0.73 dB, peak gain +2.51 dB, break-even at loss 0.32.
- robust_packet: worst best-method gain -0.64 dB, peak gain +1.17 dB, break-even at loss 0.34.
- robust_inpaint: worst best-method gain -0.41 dB, peak gain +1.22 dB, break-even at loss 0.32.
- Low-loss penalty change base -> robust_channel: -0.87 dB -> -0.73 dB (+0.14 dB).
- Low-loss penalty change base -> robust_packet: -0.87 dB -> -0.64 dB (+0.24 dB).
- Low-loss penalty change base -> robust_inpaint: -0.87 dB -> -0.41 dB (+0.46 dB).

## Digital separation baseline

- Clean: jpeg+rs 36.76 dB vs oracle UEP (three_tier_25_50_25) 30.88 dB.
- jpeg+rs falls below oracle UEP (three_tier_25_50_25) from loss = 0.35.
- At loss 0.35: jpeg+rs 21.64 dB vs oracle UEP (three_tier_25_50_25) 25.72 dB.
- Clean: jpeg2000+rs 32.91 dB vs oracle UEP (three_tier_25_50_25) 30.88 dB.
- jpeg2000+rs falls below oracle UEP (three_tier_25_50_25) from loss = 0.35.
- At loss 0.35: jpeg2000+rs 20.74 dB vs oracle UEP (three_tier_25_50_25) 25.72 dB.
- Clean: webp+rs 38.14 dB vs oracle UEP (three_tier_25_50_25) 30.88 dB.
- webp+rs falls below oracle UEP (three_tier_25_50_25) from loss = 0.35.
- At loss 0.35: webp+rs 23.04 dB vs oracle UEP (three_tier_25_50_25) 25.72 dB.

## Bandwidth-ratio sweep

- Largest gain +2.86 dB at k/n = 0.1667, loss 0.4.
- Gain is positive at every k/n in the high-loss region.

## multi-seed oracle

- Most damaging channel per model: seed42 8.7 dB (channel 10), seed100 13.2 dB (channel 3), seed7 10.5 dB (channel 4), seed13 17.1 dB (channel 1), seed21 15.9 dB (channel 11).
- Index-aligned Spearman between models averages -0.13: channel labels do permute between initialisations, as expected.
- Sorted damage profiles agree: mean Pearson 0.922, mean absolute difference 0.83 dB per channel.
- NOT every model has a >10 dB channel: the 'one catastrophic channel' claim needs softening.
- Share of total damage carried by the top-4 channels: seed42 50%, seed100 45%, seed7 47%, seed13 53%, seed21 52%.

## attention vs damage

- Pooled Pearson between attention and damage: +0.118 (2000 images x 16 channels).
- Within an image, attention ranks the channels with Spearman -0.015 (95% CI -0.020...-0.011).
- Attention's top-4 overlaps the true top-4 by 0.23 (random guessing: 0.25).
- Best cheap latent statistic by within-image Spearman: energy.

## proxy quality

- Proxies ranked by within-image Spearman with the true damage: static_train_ranking +0.89, energy +0.84, l1 +0.84, taylor +0.63, variance +0.44, random +0.00, attention -0.02.
- Static train ranking reaches +0.89: how much of the per-image structure a single global table already explains.
- Best proxy beyond the static table: energy (top-4 overlap 0.68 vs chance 0.25).

## sensitivity head

- Head top-4 overlap on validation: 0.920 ± 0.001 over 5 head seeds (random 0.25, static table 0.88).
- Protection regret: head 0.21 dB vs static table 0.60 dB vs random 15.87 dB per image.
- The head costs 2048 MACs = 0.039% of the encoder.
- The head beats the static table on top-4 overlap (+0.039); if images share one ranking the online predictor is unnecessary.

