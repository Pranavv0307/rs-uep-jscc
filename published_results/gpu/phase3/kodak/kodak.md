# Kodak (768x512) erasure sweep

- commit `931ed24146ae277b5618b0d1d26097fc250e903b` (dirty: True), 2026-10-18T20:41:18+00:00
- checkpoint `results/gpu/models/seed42/best_val.pth` sha256 `2d3b1616950625f1a7c63ab89eeafcdc5241d3c881700b008d091acc96e0b343`
- clean_psnr_mean: 24.314178466796875
- images_per_seed: 24
- packet_size: 128
- channel_seeds: (0, 1, 2, 3, 4)
- erasure_rates: (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5)
- lpips_available: False
- gf256_note: tiers split into sub-blocks of <= 255 packets (coding/scaling.py)
- PSNR: mean ± sample std over channel seeds (95% t-interval half-width in brackets)

| Method | erasure_rate | PSNR (dB) | median | p5 | SSIM | block-fail | Coding OH | Side-info B | Total OH |
|---|---|---|---|---|---|---|---|---|---|
| no_rs | 0 | 24.31 ± 0.00 (CI ±0.00) | 24.47 | 23.03 | 0.6015 | 0.000 | 0.00% | 0 | 0.00% |
| three_tier_25_25_50@attention | 0 | 24.31 ± 0.00 (CI ±0.00) | 24.47 | 23.03 | 0.6015 | 0.000 | 50.00% | 16 | 50.00% |
| three_tier_25_25_50@oracle | 0 | 24.31 ± 0.00 (CI ±0.00) | 24.47 | 23.03 | 0.6015 | 0.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@attention | 0 | 24.31 ± 0.00 (CI ±0.00) | 24.47 | 23.03 | 0.6015 | 0.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@oracle | 0 | 24.31 ± 0.00 (CI ±0.00) | 24.47 | 23.03 | 0.6015 | 0.000 | 50.00% | 16 | 50.00% |
| uniform | 0 | 24.31 ± 0.00 (CI ±0.00) | 24.47 | 23.03 | 0.6015 | 0.000 | 50.00% | 0 | 50.00% |
| no_rs | 0.1 | 23.35 ± 0.03 (CI ±0.04) | 23.57 | 21.90 | 0.5614 | 1.000 | 0.00% | 0 | 0.00% |
| three_tier_25_25_50@attention | 0.1 | 24.31 ± 0.00 (CI ±0.00) | 24.47 | 23.03 | 0.6015 | 0.000 | 50.00% | 16 | 50.00% |
| three_tier_25_25_50@oracle | 0.1 | 24.31 ± 0.00 (CI ±0.00) | 24.47 | 23.03 | 0.6015 | 0.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@attention | 0.1 | 23.76 ± 0.02 (CI ±0.03) | 24.03 | 22.53 | 0.5864 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@oracle | 0.1 | 24.32 ± 0.01 (CI ±0.01) | 24.51 | 22.97 | 0.5995 | 1.000 | 50.00% | 16 | 50.00% |
| uniform | 0.1 | 24.31 ± 0.00 (CI ±0.00) | 24.47 | 23.03 | 0.6015 | 0.000 | 50.00% | 0 | 50.00% |
| no_rs | 0.2 | 22.33 ± 0.06 (CI ±0.08) | 22.55 | 20.83 | 0.5196 | 1.000 | 0.00% | 0 | 0.00% |
| three_tier_25_25_50@attention | 0.2 | 23.62 ± 0.12 (CI ±0.15) | 23.73 | 21.96 | 0.5807 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_25_50@oracle | 0.2 | 24.18 ± 0.06 (CI ±0.08) | 24.35 | 22.66 | 0.5912 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@attention | 0.2 | 23.22 ± 0.04 (CI ±0.05) | 23.46 | 22.01 | 0.5713 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@oracle | 0.2 | 24.33 ± 0.01 (CI ±0.01) | 24.51 | 22.86 | 0.5975 | 1.000 | 50.00% | 16 | 50.00% |
| uniform | 0.2 | 24.31 ± 0.00 (CI ±0.00) | 24.47 | 23.03 | 0.6015 | 0.000 | 50.00% | 0 | 50.00% |
| no_rs | 0.3 | 21.31 ± 0.04 (CI ±0.05) | 21.48 | 19.89 | 0.4762 | 1.000 | 0.00% | 0 | 0.00% |
| three_tier_25_25_50@attention | 0.3 | 22.23 ± 0.03 (CI ±0.03) | 22.35 | 20.98 | 0.5339 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_25_50@oracle | 0.3 | 23.70 ± 0.05 (CI ±0.06) | 23.84 | 22.18 | 0.5679 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@attention | 0.3 | 22.55 ± 0.08 (CI ±0.10) | 22.76 | 21.33 | 0.5510 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@oracle | 0.3 | 24.04 ± 0.06 (CI ±0.08) | 24.24 | 22.66 | 0.5877 | 1.000 | 50.00% | 16 | 50.00% |
| uniform | 0.3 | 23.88 ± 0.10 (CI ±0.12) | 24.04 | 22.51 | 0.5868 | 0.942 | 50.00% | 0 | 50.00% |
| no_rs | 0.35 | 20.79 ± 0.04 (CI ±0.05) | 20.93 | 19.41 | 0.4534 | 1.000 | 0.00% | 0 | 0.00% |
| three_tier_25_25_50@attention | 0.35 | 21.27 ± 0.10 (CI ±0.13) | 21.39 | 19.62 | 0.5011 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_25_50@oracle | 0.35 | 22.52 ± 0.03 (CI ±0.04) | 22.70 | 20.95 | 0.5420 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@attention | 0.35 | 21.46 ± 0.13 (CI ±0.16) | 21.59 | 19.85 | 0.5085 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@oracle | 0.35 | 22.65 ± 0.04 (CI ±0.05) | 22.89 | 21.13 | 0.5502 | 1.000 | 50.00% | 16 | 50.00% |
| uniform | 0.35 | 21.73 ± 0.09 (CI ±0.11) | 21.73 | 20.10 | 0.5047 | 1.000 | 50.00% | 0 | 50.00% |
| no_rs | 0.4 | 20.24 ± 0.04 (CI ±0.05) | 20.37 | 18.92 | 0.4294 | 1.000 | 0.00% | 0 | 0.00% |
| three_tier_25_25_50@attention | 0.4 | 20.45 ± 0.05 (CI ±0.07) | 20.56 | 18.76 | 0.4702 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_25_50@oracle | 0.4 | 21.64 ± 0.04 (CI ±0.05) | 21.90 | 20.03 | 0.5203 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@attention | 0.4 | 20.50 ± 0.07 (CI ±0.09) | 20.54 | 18.76 | 0.4715 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@oracle | 0.4 | 21.66 ± 0.05 (CI ±0.06) | 21.92 | 20.06 | 0.5213 | 1.000 | 50.00% | 16 | 50.00% |
| uniform | 0.4 | 20.24 ± 0.04 (CI ±0.05) | 20.31 | 18.93 | 0.4313 | 1.000 | 50.00% | 0 | 50.00% |
| no_rs | 0.45 | 19.68 ± 0.02 (CI ±0.03) | 19.80 | 18.37 | 0.4048 | 1.000 | 0.00% | 0 | 0.00% |
| three_tier_25_25_50@attention | 0.45 | 19.88 ± 0.06 (CI ±0.07) | 19.92 | 18.26 | 0.4479 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_25_50@oracle | 0.45 | 21.09 ± 0.10 (CI ±0.12) | 21.19 | 19.17 | 0.5039 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@attention | 0.45 | 19.92 ± 0.03 (CI ±0.04) | 19.95 | 18.18 | 0.4480 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@oracle | 0.45 | 21.11 ± 0.13 (CI ±0.16) | 21.21 | 19.20 | 0.5042 | 1.000 | 50.00% | 16 | 50.00% |
| uniform | 0.45 | 19.64 ± 0.02 (CI ±0.02) | 19.74 | 18.29 | 0.4044 | 1.000 | 50.00% | 0 | 50.00% |
| no_rs | 0.5 | 19.11 ± 0.03 (CI ±0.04) | 19.24 | 17.83 | 0.3801 | 1.000 | 0.00% | 0 | 0.00% |
| three_tier_25_25_50@attention | 0.5 | 19.32 ± 0.06 (CI ±0.08) | 19.32 | 17.65 | 0.4101 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_25_50@oracle | 0.5 | 19.80 ± 0.29 (CI ±0.36) | 19.87 | 17.75 | 0.4453 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@attention | 0.5 | 19.34 ± 0.06 (CI ±0.07) | 19.34 | 17.58 | 0.4102 | 1.000 | 50.00% | 16 | 50.00% |
| three_tier_25_50_25@oracle | 0.5 | 19.82 ± 0.29 (CI ±0.36) | 19.84 | 17.79 | 0.4456 | 1.000 | 50.00% | 16 | 50.00% |
| uniform | 0.5 | 19.05 ± 0.02 (CI ±0.02) | 19.16 | 17.70 | 0.3789 | 1.000 | 50.00% | 0 | 50.00% |

# Paired difference to `uniform` (same loss mask per seed)

| Method | erasure_rate | mean gain (dB) | std over seeds | 95% CI ± | wins | significant |
|---|---|---|---|---|---|---|
| no_rs | 0 | +0.00 | 0.00 | 0.00 | 0% | no |
| three_tier_25_25_50@attention | 0 | +0.00 | 0.00 | 0.00 | 0% | no |
| three_tier_25_25_50@oracle | 0 | +0.00 | 0.00 | 0.00 | 0% | no |
| three_tier_25_50_25@attention | 0 | +0.00 | 0.00 | 0.00 | 0% | no |
| three_tier_25_50_25@oracle | 0 | +0.00 | 0.00 | 0.00 | 0% | no |
| no_rs | 0.1 | -0.96 | 0.03 | 0.04 | 0% | yes |
| three_tier_25_25_50@attention | 0.1 | +0.00 | 0.00 | 0.00 | 0% | no |
| three_tier_25_25_50@oracle | 0.1 | +0.00 | 0.00 | 0.00 | 0% | no |
| three_tier_25_50_25@attention | 0.1 | -0.55 | 0.02 | 0.03 | 0% | yes |
| three_tier_25_50_25@oracle | 0.1 | +0.01 | 0.01 | 0.01 | 80% | no |
| no_rs | 0.2 | -1.99 | 0.06 | 0.08 | 0% | yes |
| three_tier_25_25_50@attention | 0.2 | -0.69 | 0.12 | 0.15 | 0% | yes |
| three_tier_25_25_50@oracle | 0.2 | -0.14 | 0.06 | 0.08 | 0% | yes |
| three_tier_25_50_25@attention | 0.2 | -1.09 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_50_25@oracle | 0.2 | +0.01 | 0.01 | 0.01 | 80% | no |
| no_rs | 0.3 | -2.57 | 0.12 | 0.15 | 0% | yes |
| three_tier_25_25_50@attention | 0.3 | -1.64 | 0.11 | 0.14 | 0% | yes |
| three_tier_25_25_50@oracle | 0.3 | -0.17 | 0.11 | 0.13 | 0% | yes |
| three_tier_25_50_25@attention | 0.3 | -1.33 | 0.09 | 0.11 | 0% | yes |
| three_tier_25_50_25@oracle | 0.3 | +0.16 | 0.10 | 0.13 | 100% | yes |
| no_rs | 0.35 | -0.94 | 0.11 | 0.14 | 0% | yes |
| three_tier_25_25_50@attention | 0.35 | -0.46 | 0.11 | 0.14 | 0% | yes |
| three_tier_25_25_50@oracle | 0.35 | +0.79 | 0.08 | 0.10 | 100% | yes |
| three_tier_25_50_25@attention | 0.35 | -0.27 | 0.14 | 0.17 | 0% | yes |
| three_tier_25_50_25@oracle | 0.35 | +0.92 | 0.06 | 0.07 | 100% | yes |
| no_rs | 0.4 | +0.00 | 0.05 | 0.07 | 60% | no |
| three_tier_25_25_50@attention | 0.4 | +0.21 | 0.04 | 0.05 | 100% | yes |
| three_tier_25_25_50@oracle | 0.4 | +1.40 | 0.06 | 0.08 | 100% | yes |
| three_tier_25_50_25@attention | 0.4 | +0.27 | 0.05 | 0.06 | 100% | yes |
| three_tier_25_50_25@oracle | 0.4 | +1.43 | 0.05 | 0.07 | 100% | yes |
| no_rs | 0.45 | +0.03 | 0.04 | 0.04 | 100% | no |
| three_tier_25_25_50@attention | 0.45 | +0.24 | 0.05 | 0.06 | 100% | yes |
| three_tier_25_25_50@oracle | 0.45 | +1.45 | 0.09 | 0.11 | 100% | yes |
| three_tier_25_50_25@attention | 0.45 | +0.28 | 0.02 | 0.03 | 100% | yes |
| three_tier_25_50_25@oracle | 0.45 | +1.46 | 0.12 | 0.15 | 100% | yes |
| no_rs | 0.5 | +0.06 | 0.04 | 0.05 | 100% | yes |
| three_tier_25_25_50@attention | 0.5 | +0.27 | 0.07 | 0.08 | 100% | yes |
| three_tier_25_25_50@oracle | 0.5 | +0.75 | 0.30 | 0.37 | 100% | yes |
| three_tier_25_50_25@attention | 0.5 | +0.29 | 0.06 | 0.07 | 100% | yes |
| three_tier_25_50_25@oracle | 0.5 | +0.76 | 0.29 | 0.36 | 100% | yes |

# Insights

- Based on 5 channel seed(s); 24 images per seed. All gains below are paired per seed against the baseline (same packet losses), so they carry no luck-of-the-draw advantage.
- No low-loss penalty: the best method is never more than 0.05 dB below `uniform` at any erasure_rate in this sweep.
- Peak gain: +1.46 dB at erasure_rate = 0.45 by three_tier_25_50_25@oracle (95% CI ±0.15, significant).
- Importance sources ranked by their best scheme's mean gain over erasure_rate >= 0.35: oracle +1.14 dB, attention +0.14 dB.
- Share of the reference ranking's gain captured: attention 12%. A share above 100% means that source allocated protection better than the reference.
- Worst-case images at erasure_rate = 0.4: 5th-percentile PSNR 20.06 dB (three_tier_25_50_25@oracle) vs 18.93 dB (uniform).
- Worst-case images at erasure_rate = 0.5: 5th-percentile PSNR 17.79 dB (three_tier_25_50_25@oracle) vs 17.70 dB (uniform).
- Seed sensitivity: std across channel seeds stays below 0.3 dB everywhere.
- `uniform` shows its steepest fall (-2.14 dB) between erasure_rate = 0.3 and 0.35 -- its cliff.

# Perceptual metrics

- ms_ssim_mean at loss 0.4: best three_tier_25_50_25@oracle +0.1540 vs uniform (roadmap target: ΔMS-SSIM ≥ +0.04).
