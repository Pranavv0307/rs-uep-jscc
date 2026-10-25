# Burst-loss study, average loss 0.45

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
| no_rs | 1 | 18.94 ± 0.01 (CI ±0.01) | 18.90 | 16.01 | 0.7452 | 1.000 | 0.00% | 0 | 0.00% | 1.802 |
| three_tier_25_25_50@oracle | 1 | 21.88 ± 0.03 (CI ±0.04) | 21.80 | 17.11 | 0.8230 | 1.000 | 50.00% | 16 | 51.56% | 1.802 |
| three_tier_25_25_50@oracle#rr | 1 | 21.87 ± 0.03 (CI ±0.03) | 21.83 | 17.07 | 0.8227 | 1.000 | 50.00% | 16 | 51.56% | 1.802 |
| three_tier_25_50_25@oracle | 1 | 21.77 ± 0.03 (CI ±0.04) | 21.71 | 17.07 | 0.8201 | 1.000 | 50.00% | 16 | 51.56% | 1.802 |
| three_tier_25_50_25@oracle#rr | 1 | 21.77 ± 0.04 (CI ±0.05) | 21.72 | 17.04 | 0.8200 | 1.000 | 50.00% | 16 | 51.56% | 1.802 |
| two_tier_50_50@oracle | 1 | 19.40 ± 0.03 (CI ±0.03) | 19.04 | 16.01 | 0.7558 | 0.996 | 50.00% | 16 | 51.56% | 1.802 |
| two_tier_50_50@oracle#rr | 1 | 19.41 ± 0.03 (CI ±0.03) | 19.04 | 16.00 | 0.7561 | 0.995 | 50.00% | 16 | 51.56% | 1.802 |
| uniform | 1 | 19.07 ± 0.01 (CI ±0.02) | 18.91 | 16.01 | 0.7472 | 0.987 | 50.00% | 0 | 50.00% | 1.802 |
| no_rs | 2 | 18.96 ± 0.02 (CI ±0.03) | 18.94 | 16.01 | 0.7454 | 1.000 | 0.00% | 0 | 0.00% | 1.741 |
| three_tier_25_25_50@oracle | 2 | 21.91 ± 0.03 (CI ±0.03) | 21.85 | 17.22 | 0.8237 | 1.000 | 50.00% | 16 | 51.56% | 1.741 |
| three_tier_25_25_50@oracle#rr | 2 | 21.87 ± 0.02 (CI ±0.03) | 21.81 | 17.13 | 0.8230 | 1.000 | 50.00% | 16 | 51.56% | 1.741 |
| three_tier_25_50_25@oracle | 2 | 21.80 ± 0.03 (CI ±0.04) | 21.76 | 17.15 | 0.8210 | 1.000 | 50.00% | 16 | 51.56% | 1.741 |
| three_tier_25_50_25@oracle#rr | 2 | 21.77 ± 0.03 (CI ±0.03) | 21.71 | 17.06 | 0.8203 | 1.000 | 50.00% | 16 | 51.56% | 1.741 |
| two_tier_50_50@oracle | 2 | 19.39 ± 0.01 (CI ±0.02) | 19.04 | 16.07 | 0.7552 | 0.996 | 50.00% | 16 | 51.56% | 1.741 |
| two_tier_50_50@oracle#rr | 2 | 19.42 ± 0.01 (CI ±0.02) | 19.04 | 16.07 | 0.7565 | 0.996 | 50.00% | 16 | 51.56% | 1.741 |
| uniform | 2 | 19.08 ± 0.03 (CI ±0.03) | 18.94 | 16.01 | 0.7473 | 0.988 | 50.00% | 0 | 50.00% | 1.741 |
| no_rs | 4 | 18.87 ± 0.02 (CI ±0.03) | 18.78 | 15.54 | 0.7458 | 1.000 | 0.00% | 0 | 0.00% | 2.697 |
| three_tier_25_25_50@oracle | 4 | 21.64 ± 0.02 (CI ±0.02) | 21.48 | 16.29 | 0.8139 | 0.996 | 50.00% | 16 | 51.56% | 2.697 |
| three_tier_25_25_50@oracle#rr | 4 | 21.98 ± 0.03 (CI ±0.04) | 21.94 | 16.51 | 0.8209 | 0.998 | 50.00% | 16 | 51.56% | 2.697 |
| three_tier_25_50_25@oracle | 4 | 21.70 ± 0.03 (CI ±0.03) | 21.42 | 16.29 | 0.8135 | 0.999 | 50.00% | 16 | 51.56% | 2.697 |
| three_tier_25_50_25@oracle#rr | 4 | 22.02 ± 0.03 (CI ±0.04) | 21.85 | 16.48 | 0.8202 | 1.000 | 50.00% | 16 | 51.56% | 2.697 |
| two_tier_50_50@oracle | 4 | 19.97 ± 0.03 (CI ±0.04) | 19.06 | 15.52 | 0.7688 | 0.973 | 50.00% | 16 | 51.56% | 2.697 |
| two_tier_50_50@oracle#rr | 4 | 19.76 ± 0.03 (CI ±0.04) | 18.98 | 15.67 | 0.7600 | 0.957 | 50.00% | 16 | 51.56% | 2.697 |
| uniform | 4 | 19.57 ± 0.04 (CI ±0.04) | 18.84 | 15.54 | 0.7560 | 0.930 | 50.00% | 0 | 50.00% | 2.697 |
| no_rs | 8 | 18.91 ± 0.03 (CI ±0.04) | 18.68 | 15.01 | 0.7452 | 1.000 | 0.00% | 0 | 0.00% | 3.715 |
| three_tier_25_25_50@oracle | 8 | 21.45 ± 0.05 (CI ±0.06) | 21.06 | 15.55 | 0.8047 | 0.979 | 50.00% | 16 | 51.56% | 3.715 |
| three_tier_25_25_50@oracle#rr | 8 | 22.11 ± 0.04 (CI ±0.05) | 22.05 | 15.76 | 0.8164 | 0.980 | 50.00% | 16 | 51.56% | 3.715 |
| three_tier_25_50_25@oracle | 8 | 21.67 ± 0.05 (CI ±0.06) | 21.05 | 15.52 | 0.8068 | 0.990 | 50.00% | 16 | 51.56% | 3.715 |
| three_tier_25_50_25@oracle#rr | 8 | 22.29 ± 0.04 (CI ±0.05) | 21.95 | 15.71 | 0.8175 | 1.000 | 50.00% | 16 | 51.56% | 3.715 |
| two_tier_50_50@oracle | 8 | 20.54 ± 0.06 (CI ±0.08) | 19.13 | 14.79 | 0.7768 | 0.933 | 50.00% | 16 | 51.56% | 3.715 |
| two_tier_50_50@oracle#rr | 8 | 20.48 ± 0.06 (CI ±0.07) | 18.96 | 15.02 | 0.7691 | 0.872 | 50.00% | 16 | 51.56% | 3.715 |
| uniform | 8 | 20.45 ± 0.09 (CI ±0.11) | 18.85 | 15.02 | 0.7671 | 0.837 | 50.00% | 0 | 50.00% | 3.715 |
| no_rs | 16 | 19.01 ± 0.04 (CI ±0.06) | 18.40 | 14.29 | 0.7399 | 0.997 | 0.00% | 0 | 0.00% | 4.583 |
| three_tier_25_25_50@oracle | 16 | 21.25 ± 0.05 (CI ±0.06) | 20.52 | 14.71 | 0.7915 | 0.948 | 50.00% | 16 | 51.56% | 4.583 |
| three_tier_25_25_50@oracle#rr | 16 | 22.27 ± 0.08 (CI ±0.10) | 22.07 | 14.80 | 0.8077 | 0.927 | 50.00% | 16 | 51.56% | 4.583 |
| three_tier_25_50_25@oracle | 16 | 21.57 ± 0.05 (CI ±0.06) | 20.58 | 14.62 | 0.7951 | 0.961 | 50.00% | 16 | 51.56% | 4.583 |
| three_tier_25_50_25@oracle#rr | 16 | 22.46 ± 0.08 (CI ±0.09) | 21.91 | 14.75 | 0.8099 | 0.997 | 50.00% | 16 | 51.56% | 4.583 |
| two_tier_50_50@oracle | 16 | 20.96 ± 0.06 (CI ±0.08) | 19.16 | 14.04 | 0.7752 | 0.885 | 50.00% | 16 | 51.56% | 4.583 |
| two_tier_50_50@oracle#rr | 16 | 21.14 ± 0.07 (CI ±0.08) | 18.89 | 14.11 | 0.7715 | 0.781 | 50.00% | 16 | 51.56% | 4.583 |
| uniform | 16 | 21.14 ± 0.06 (CI ±0.07) | 18.72 | 14.30 | 0.7690 | 0.753 | 50.00% | 0 | 50.00% | 4.583 |

# Paired difference to `uniform` (same loss mask per seed)

| Method | burst_length | mean gain (dB) | std over seeds | 95% CI ± | wins | significant |
|---|---|---|---|---|---|---|
| no_rs | 1 | -0.13 | 0.02 | 0.02 | 0% | yes |
| three_tier_25_25_50@oracle | 1 | +2.81 | 0.03 | 0.04 | 100% | yes |
| three_tier_25_25_50@oracle#rr | 1 | +2.80 | 0.02 | 0.03 | 100% | yes |
| three_tier_25_50_25@oracle | 1 | +2.69 | 0.03 | 0.04 | 100% | yes |
| three_tier_25_50_25@oracle#rr | 1 | +2.70 | 0.03 | 0.04 | 100% | yes |
| two_tier_50_50@oracle | 1 | +0.33 | 0.02 | 0.03 | 100% | yes |
| two_tier_50_50@oracle#rr | 1 | +0.34 | 0.02 | 0.02 | 100% | yes |
| no_rs | 2 | -0.12 | 0.01 | 0.01 | 0% | yes |
| three_tier_25_25_50@oracle | 2 | +2.83 | 0.02 | 0.02 | 100% | yes |
| three_tier_25_25_50@oracle#rr | 2 | +2.79 | 0.03 | 0.03 | 100% | yes |
| three_tier_25_50_25@oracle | 2 | +2.72 | 0.02 | 0.02 | 100% | yes |
| three_tier_25_50_25@oracle#rr | 2 | +2.69 | 0.03 | 0.04 | 100% | yes |
| two_tier_50_50@oracle | 2 | +0.31 | 0.02 | 0.02 | 100% | yes |
| two_tier_50_50@oracle#rr | 2 | +0.34 | 0.03 | 0.03 | 100% | yes |
| no_rs | 4 | -0.70 | 0.01 | 0.02 | 0% | yes |
| three_tier_25_25_50@oracle | 4 | +2.06 | 0.02 | 0.02 | 100% | yes |
| three_tier_25_25_50@oracle#rr | 4 | +2.41 | 0.02 | 0.02 | 100% | yes |
| three_tier_25_50_25@oracle | 4 | +2.13 | 0.01 | 0.02 | 100% | yes |
| three_tier_25_50_25@oracle#rr | 4 | +2.45 | 0.02 | 0.02 | 100% | yes |
| two_tier_50_50@oracle | 4 | +0.39 | 0.03 | 0.04 | 100% | yes |
| two_tier_50_50@oracle#rr | 4 | +0.19 | 0.02 | 0.02 | 100% | yes |
| no_rs | 8 | -1.54 | 0.06 | 0.07 | 0% | yes |
| three_tier_25_25_50@oracle | 8 | +1.00 | 0.06 | 0.08 | 100% | yes |
| three_tier_25_25_50@oracle#rr | 8 | +1.66 | 0.05 | 0.06 | 100% | yes |
| three_tier_25_50_25@oracle | 8 | +1.22 | 0.06 | 0.08 | 100% | yes |
| three_tier_25_50_25@oracle#rr | 8 | +1.84 | 0.05 | 0.06 | 100% | yes |
| two_tier_50_50@oracle | 8 | +0.09 | 0.05 | 0.06 | 100% | yes |
| two_tier_50_50@oracle#rr | 8 | +0.02 | 0.05 | 0.06 | 60% | no |
| no_rs | 16 | -2.13 | 0.03 | 0.04 | 0% | yes |
| three_tier_25_25_50@oracle | 16 | +0.11 | 0.02 | 0.02 | 100% | yes |
| three_tier_25_25_50@oracle#rr | 16 | +1.13 | 0.03 | 0.03 | 100% | yes |
| three_tier_25_50_25@oracle | 16 | +0.42 | 0.02 | 0.02 | 100% | yes |
| three_tier_25_50_25@oracle#rr | 16 | +1.32 | 0.02 | 0.03 | 100% | yes |
| two_tier_50_50@oracle | 16 | -0.18 | 0.02 | 0.02 | 0% | yes |
| two_tier_50_50@oracle#rr | 16 | -0.00 | 0.02 | 0.03 | 60% | no |

# Insights

- Based on 5 channel seed(s); 10000 images per seed. All gains below are paired per seed against the baseline (same packet losses), so they carry no luck-of-the-draw advantage.
- No low-loss penalty: the best method is never more than 0.05 dB below `uniform` at any burst_length in this sweep.
- Peak gain: +2.83 dB at burst_length = 2 by three_tier_25_25_50@oracle (95% CI ±0.02, significant).
- Importance sources ranked by their best scheme's mean gain over burst_length >= 0: oracle#rr +2.20 dB, oracle +1.84 dB.
- Worst-case images at burst_length = 1: 5th-percentile PSNR 17.11 dB (three_tier_25_25_50@oracle) vs 16.01 dB (uniform).
- Worst-case images at burst_length = 16: 5th-percentile PSNR 14.75 dB (three_tier_25_50_25@oracle#rr) vs 14.30 dB (uniform).
- Seed sensitivity: std across channel seeds stays below 0.3 dB everywhere.
- `uniform` shows its steepest fall (0.01 dB) between burst_length = 1 and 2 -- its cliff.

# Burst-specific findings

- Burst length 1: best method three_tier_25_25_50@oracle at +2.81 dB over uniform
- Burst length 2: best method three_tier_25_25_50@oracle at +2.83 dB over uniform
- Burst length 4: best method three_tier_25_50_25@oracle#rr at +2.45 dB over uniform
- Burst length 8: best method three_tier_25_50_25@oracle#rr at +1.84 dB over uniform
- Burst length 16: best method three_tier_25_50_25@oracle#rr at +1.32 dB over uniform
- Interleaving three_tier_25_25_50@oracle: +0.35 dB at burst length 4
- Interleaving three_tier_25_25_50@oracle: +0.66 dB at burst length 8
- Interleaving three_tier_25_25_50@oracle: +1.02 dB at burst length 16
- Interleaving three_tier_25_50_25@oracle: +0.31 dB at burst length 4
- Interleaving three_tier_25_50_25@oracle: +0.62 dB at burst length 8
- Interleaving three_tier_25_50_25@oracle: +0.89 dB at burst length 16
- Interleaving two_tier_50_50@oracle: -0.21 dB at burst length 4
- Interleaving two_tier_50_50@oracle: -0.06 dB at burst length 8
- Interleaving two_tier_50_50@oracle: +0.18 dB at burst length 16
