# Burst-loss study, average loss 0.35

- commit `2fd2d1f0c25c038d84f2e21e3528ba53b37d938e` (dirty: False), 2026-10-18T09:16:56+00:00
- checkpoint `/kaggle/input/datasets/vaishakprasadbhat/rs-uep-jscc-models/seed42/best_val.pth` sha256 `2d3b1616950625f1a7c63ab89eeafcdc5241d3c881700b008d091acc96e0b343`
- images_per_seed: 10000
- is_full_test_set: True
- channels: 16
- packet_size: 16
- overhead: 0.5
- clean_psnr_mean: 30.883869171142578
- side_info_rule: 0 B for uniform/no_rs, one byte per channel for sorted methods
- channel_seeds: [0, 1, 2, 3, 4]
- good_loss: 0.01
- bad_loss: 0.85
- mask_protocol: Gilbert-Elliott timeline mask shared by all methods; burst_length 1 = Bernoulli
- PSNR: mean ± sample std over channel seeds (95% t-interval half-width in brackets)

| Method | burst_length | PSNR (dB) | median | p5 | SSIM | block-fail | Coding OH | Side-info B | Total OH | timeline_mean_burst |
|---|---|---|---|---|---|---|---|---|---|---|
| no_rs | 1 | 20.36 ± 0.01 (CI ±0.01) | 20.30 | 17.27 | 0.7947 | 1.000 | 0.00% | 0 | 0.00% | 1.529 |
| three_tier_25_25_50@oracle | 1 | 24.92 ± 0.02 (CI ±0.03) | 24.91 | 20.23 | 0.8889 | 0.985 | 50.00% | 16 | 51.56% | 1.529 |
| three_tier_25_25_50@oracle#rr | 1 | 24.91 ± 0.02 (CI ±0.03) | 24.90 | 20.19 | 0.8889 | 0.985 | 50.00% | 16 | 51.56% | 1.529 |
| three_tier_25_50_25@oracle | 1 | 25.72 ± 0.05 (CI ±0.06) | 25.31 | 20.24 | 0.8974 | 0.999 | 50.00% | 16 | 51.56% | 1.529 |
| three_tier_25_50_25@oracle#rr | 1 | 25.71 ± 0.03 (CI ±0.04) | 25.31 | 20.25 | 0.8973 | 1.000 | 50.00% | 16 | 51.56% | 1.529 |
| two_tier_50_50@oracle | 1 | 24.12 ± 0.04 (CI ±0.05) | 22.98 | 17.76 | 0.8617 | 0.777 | 50.00% | 16 | 51.56% | 1.529 |
| two_tier_50_50@oracle#rr | 1 | 24.13 ± 0.04 (CI ±0.05) | 23.03 | 17.71 | 0.8618 | 0.778 | 50.00% | 16 | 51.56% | 1.529 |
| uniform | 1 | 24.40 ± 0.06 (CI ±0.07) | 21.71 | 17.44 | 0.8544 | 0.589 | 50.00% | 0 | 50.00% | 1.529 |
| no_rs | 2 | 20.32 ± 0.02 (CI ±0.02) | 20.25 | 17.11 | 0.7945 | 1.000 | 0.00% | 0 | 0.00% | 1.736 |
| three_tier_25_25_50@oracle | 2 | 24.80 ± 0.03 (CI ±0.04) | 24.81 | 19.78 | 0.8861 | 0.976 | 50.00% | 16 | 51.56% | 1.736 |
| three_tier_25_25_50@oracle#rr | 2 | 24.94 ± 0.01 (CI ±0.02) | 24.93 | 20.17 | 0.8888 | 0.979 | 50.00% | 16 | 51.56% | 1.736 |
| three_tier_25_50_25@oracle | 2 | 25.60 ± 0.03 (CI ±0.03) | 25.25 | 19.86 | 0.8948 | 0.999 | 50.00% | 16 | 51.56% | 1.736 |
| three_tier_25_50_25@oracle#rr | 2 | 25.74 ± 0.03 (CI ±0.03) | 25.42 | 20.16 | 0.8972 | 0.999 | 50.00% | 16 | 51.56% | 1.736 |
| two_tier_50_50@oracle | 2 | 24.07 ± 0.04 (CI ±0.05) | 23.04 | 17.56 | 0.8604 | 0.775 | 50.00% | 16 | 51.56% | 1.736 |
| two_tier_50_50@oracle#rr | 2 | 24.23 ± 0.05 (CI ±0.07) | 23.04 | 17.57 | 0.8611 | 0.741 | 50.00% | 16 | 51.56% | 1.736 |
| uniform | 2 | 24.39 ± 0.05 (CI ±0.06) | 21.74 | 17.27 | 0.8541 | 0.582 | 50.00% | 0 | 50.00% | 1.736 |
| no_rs | 4 | 20.32 ± 0.01 (CI ±0.02) | 20.16 | 16.49 | 0.7947 | 1.000 | 0.00% | 0 | 0.00% | 2.664 |
| three_tier_25_25_50@oracle | 4 | 24.37 ± 0.03 (CI ±0.04) | 24.38 | 18.04 | 0.8741 | 0.930 | 50.00% | 16 | 51.56% | 2.664 |
| three_tier_25_25_50@oracle#rr | 4 | 25.08 ± 0.05 (CI ±0.06) | 25.13 | 19.39 | 0.8870 | 0.938 | 50.00% | 16 | 51.56% | 2.664 |
| three_tier_25_50_25@oracle | 4 | 25.05 ± 0.04 (CI ±0.06) | 24.79 | 18.13 | 0.8817 | 0.983 | 50.00% | 16 | 51.56% | 2.664 |
| three_tier_25_50_25@oracle#rr | 4 | 25.76 ± 0.03 (CI ±0.04) | 25.74 | 19.27 | 0.8942 | 0.999 | 50.00% | 16 | 51.56% | 2.664 |
| two_tier_50_50@oracle | 4 | 23.95 ± 0.05 (CI ±0.07) | 23.36 | 16.72 | 0.8554 | 0.754 | 50.00% | 16 | 51.56% | 2.664 |
| two_tier_50_50@oracle#rr | 4 | 24.53 ± 0.05 (CI ±0.06) | 23.34 | 16.88 | 0.8585 | 0.633 | 50.00% | 16 | 51.56% | 2.664 |
| uniform | 4 | 24.60 ± 0.05 (CI ±0.07) | 22.40 | 16.62 | 0.8548 | 0.544 | 50.00% | 0 | 50.00% | 2.664 |
| no_rs | 8 | 20.45 ± 0.02 (CI ±0.02) | 20.10 | 15.85 | 0.7937 | 0.998 | 0.00% | 0 | 0.00% | 3.650 |
| three_tier_25_25_50@oracle | 8 | 23.87 ± 0.04 (CI ±0.05) | 23.78 | 16.82 | 0.8588 | 0.886 | 50.00% | 16 | 51.56% | 3.650 |
| three_tier_25_25_50@oracle#rr | 8 | 25.10 ± 0.01 (CI ±0.02) | 25.31 | 17.49 | 0.8793 | 0.858 | 50.00% | 16 | 51.56% | 3.650 |
| three_tier_25_50_25@oracle | 8 | 24.46 ± 0.04 (CI ±0.05) | 24.21 | 16.89 | 0.8657 | 0.935 | 50.00% | 16 | 51.56% | 3.650 |
| three_tier_25_50_25@oracle#rr | 8 | 25.59 ± 0.02 (CI ±0.03) | 26.02 | 17.37 | 0.8848 | 0.996 | 50.00% | 16 | 51.56% | 3.650 |
| two_tier_50_50@oracle | 8 | 23.78 ± 0.04 (CI ±0.06) | 23.54 | 15.78 | 0.8466 | 0.737 | 50.00% | 16 | 51.56% | 3.650 |
| two_tier_50_50@oracle#rr | 8 | 24.58 ± 0.04 (CI ±0.05) | 24.19 | 15.91 | 0.8526 | 0.572 | 50.00% | 16 | 51.56% | 3.650 |
| uniform | 8 | 24.64 ± 0.04 (CI ±0.04) | 23.47 | 15.93 | 0.8506 | 0.521 | 50.00% | 0 | 50.00% | 3.650 |
| no_rs | 16 | 20.75 ± 0.06 (CI ±0.07) | 19.96 | 14.94 | 0.7882 | 0.978 | 0.00% | 0 | 0.00% | 4.477 |
| three_tier_25_25_50@oracle | 16 | 23.38 ± 0.06 (CI ±0.08) | 22.97 | 15.70 | 0.8398 | 0.839 | 50.00% | 16 | 51.56% | 4.477 |
| three_tier_25_25_50@oracle#rr | 16 | 25.00 ± 0.07 (CI ±0.08) | 25.55 | 15.90 | 0.8654 | 0.766 | 50.00% | 16 | 51.56% | 4.477 |
| three_tier_25_50_25@oracle | 16 | 23.89 ± 0.06 (CI ±0.07) | 23.50 | 15.66 | 0.8461 | 0.875 | 50.00% | 16 | 51.56% | 4.477 |
| three_tier_25_50_25@oracle#rr | 16 | 25.25 ± 0.07 (CI ±0.09) | 26.31 | 15.80 | 0.8692 | 0.975 | 50.00% | 16 | 51.56% | 4.477 |
| two_tier_50_50@oracle | 16 | 23.59 ± 0.10 (CI ±0.12) | 23.67 | 14.80 | 0.8328 | 0.718 | 50.00% | 16 | 51.56% | 4.477 |
| two_tier_50_50@oracle#rr | 16 | 24.45 ± 0.07 (CI ±0.09) | 25.18 | 14.79 | 0.8405 | 0.541 | 50.00% | 16 | 51.56% | 4.477 |
| uniform | 16 | 24.45 ± 0.08 (CI ±0.11) | 24.99 | 14.97 | 0.8376 | 0.509 | 50.00% | 0 | 50.00% | 4.477 |

# Paired difference to `uniform` (same loss mask per seed)

| Method | burst_length | mean gain (dB) | std over seeds | 95% CI ± | wins | significant |
|---|---|---|---|---|---|---|
| no_rs | 1 | -4.04 | 0.05 | 0.06 | 0% | yes |
| three_tier_25_25_50@oracle | 1 | +0.52 | 0.05 | 0.07 | 100% | yes |
| three_tier_25_25_50@oracle#rr | 1 | +0.51 | 0.07 | 0.09 | 100% | yes |
| three_tier_25_50_25@oracle | 1 | +1.31 | 0.06 | 0.07 | 100% | yes |
| three_tier_25_50_25@oracle#rr | 1 | +1.31 | 0.03 | 0.04 | 100% | yes |
| two_tier_50_50@oracle | 1 | -0.28 | 0.05 | 0.06 | 0% | yes |
| two_tier_50_50@oracle#rr | 1 | -0.27 | 0.05 | 0.06 | 0% | yes |
| no_rs | 2 | -4.07 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_25_50@oracle | 2 | +0.41 | 0.03 | 0.03 | 100% | yes |
| three_tier_25_25_50@oracle#rr | 2 | +0.55 | 0.05 | 0.06 | 100% | yes |
| three_tier_25_50_25@oracle | 2 | +1.21 | 0.03 | 0.04 | 100% | yes |
| three_tier_25_50_25@oracle#rr | 2 | +1.35 | 0.04 | 0.05 | 100% | yes |
| two_tier_50_50@oracle | 2 | -0.32 | 0.05 | 0.06 | 0% | yes |
| two_tier_50_50@oracle#rr | 2 | -0.16 | 0.03 | 0.04 | 0% | yes |
| no_rs | 4 | -4.28 | 0.05 | 0.06 | 0% | yes |
| three_tier_25_25_50@oracle | 4 | -0.23 | 0.03 | 0.04 | 0% | yes |
| three_tier_25_25_50@oracle#rr | 4 | +0.48 | 0.02 | 0.03 | 100% | yes |
| three_tier_25_50_25@oracle | 4 | +0.46 | 0.03 | 0.04 | 100% | yes |
| three_tier_25_50_25@oracle#rr | 4 | +1.17 | 0.03 | 0.04 | 100% | yes |
| two_tier_50_50@oracle | 4 | -0.65 | 0.04 | 0.05 | 0% | yes |
| two_tier_50_50@oracle#rr | 4 | -0.07 | 0.02 | 0.02 | 0% | yes |
| no_rs | 8 | -4.19 | 0.02 | 0.03 | 0% | yes |
| three_tier_25_25_50@oracle | 8 | -0.77 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_25_50@oracle#rr | 8 | +0.46 | 0.03 | 0.04 | 100% | yes |
| three_tier_25_50_25@oracle | 8 | -0.18 | 0.03 | 0.03 | 0% | yes |
| three_tier_25_50_25@oracle#rr | 8 | +0.95 | 0.04 | 0.05 | 100% | yes |
| two_tier_50_50@oracle | 8 | -0.87 | 0.05 | 0.07 | 0% | yes |
| two_tier_50_50@oracle#rr | 8 | -0.06 | 0.03 | 0.03 | 0% | yes |
| no_rs | 16 | -3.70 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_25_50@oracle | 16 | -1.07 | 0.06 | 0.07 | 0% | yes |
| three_tier_25_25_50@oracle#rr | 16 | +0.55 | 0.03 | 0.03 | 100% | yes |
| three_tier_25_50_25@oracle | 16 | -0.56 | 0.05 | 0.07 | 0% | yes |
| three_tier_25_50_25@oracle#rr | 16 | +0.80 | 0.02 | 0.02 | 100% | yes |
| two_tier_50_50@oracle | 16 | -0.87 | 0.04 | 0.04 | 0% | yes |
| two_tier_50_50@oracle#rr | 16 | -0.00 | 0.02 | 0.02 | 40% | no |

# Insights

- Based on 5 channel seed(s); 10000 images per seed. All gains below are paired per seed against the baseline (same packet losses), so they carry no luck-of-the-draw advantage.
- No low-loss penalty: the best method is never more than 0.05 dB below `uniform` at any burst_length in this sweep.
- Peak gain: +1.35 dB at burst_length = 2 by three_tier_25_50_25@oracle#rr (95% CI ±0.05, significant).
- Importance sources ranked by their best scheme's mean gain over burst_length >= 0: oracle#rr +1.11 dB, oracle +0.45 dB.
- Worst-case images at burst_length = 1: 5th-percentile PSNR 20.24 dB (three_tier_25_50_25@oracle) vs 17.44 dB (uniform).
- Worst-case images at burst_length = 16: 5th-percentile PSNR 15.80 dB (three_tier_25_50_25@oracle#rr) vs 14.97 dB (uniform).
- Seed sensitivity: std across channel seeds stays below 0.3 dB everywhere.
- `uniform` shows its steepest fall (-0.19 dB) between burst_length = 8 and 16 -- its cliff.

# Burst-specific findings

- Burst length 1: best method three_tier_25_50_25@oracle at +1.31 dB over uniform
- Burst length 2: best method three_tier_25_50_25@oracle#rr at +1.35 dB over uniform
- Burst length 4: best method three_tier_25_50_25@oracle#rr at +1.17 dB over uniform
- Burst length 8: best method three_tier_25_50_25@oracle#rr at +0.95 dB over uniform
- Burst length 16: best method three_tier_25_50_25@oracle#rr at +0.80 dB over uniform
- Interleaving three_tier_25_25_50@oracle: +0.71 dB at burst length 4
- Interleaving three_tier_25_25_50@oracle: +1.23 dB at burst length 8
- Interleaving three_tier_25_25_50@oracle: +1.62 dB at burst length 16
- Interleaving three_tier_25_50_25@oracle: +0.71 dB at burst length 4
- Interleaving three_tier_25_50_25@oracle: +1.13 dB at burst length 8
- Interleaving three_tier_25_50_25@oracle: +1.35 dB at burst length 16
- Interleaving two_tier_50_50@oracle: +0.58 dB at burst length 4
- Interleaving two_tier_50_50@oracle: +0.81 dB at burst length 8
- Interleaving two_tier_50_50@oracle: +0.86 dB at burst length 16
