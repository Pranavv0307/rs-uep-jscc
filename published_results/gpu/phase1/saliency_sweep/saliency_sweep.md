# cifar10/val erasure sweep

- commit `19b8298bbce83bcd17a0c403d75be4a229a1160d` (dirty: True), 2026-10-18T00:52:41+00:00
- checkpoint `results/gpu/models/seed42/best_val.pth` sha256 `2d3b1616950625f1a7c63ab89eeafcdc5241d3c881700b008d091acc96e0b343`
- images_per_seed: 10000
- is_full_test_set: False
- channels: 16
- packet_size: 16
- overhead: 0.5
- clean_psnr_mean: 30.84453773498535
- side_info_rule: 0 B for uniform/no_rs, one byte per channel for sorted methods
- channel_seeds: [0, 1, 2, 3, 4]
- mask_protocol: one master mask per (batch, seed, rate) shared by all methods; smaller schemes use its first columns
- PSNR: mean ± sample std over channel seeds (95% t-interval half-width in brackets)

| Method | erasure_rate | PSNR (dB) | median | p5 | SSIM | block-fail | Coding OH | Side-info B | Total OH |
|---|---|---|---|---|---|---|---|---|---|
| no_rs | 0.3 | 21.14 ± 0.01 (CI ±0.02) | 21.07 | 17.93 | 0.8172 | 1.000 | 0.00% | 0 | 0.00% |
| three_tier_25_50_25@attention | 0.3 | 25.55 ± 0.02 (CI ±0.02) | 25.88 | 19.37 | 0.9067 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@energy | 0.3 | 27.71 ± 0.02 (CI ±0.02) | 28.16 | 21.60 | 0.9283 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@l1 | 0.3 | 27.69 ± 0.02 (CI ±0.02) | 28.13 | 21.63 | 0.9284 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@oracle | 0.3 | 27.69 ± 0.01 (CI ±0.02) | 28.17 | 21.55 | 0.9264 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@oracle_perimage | 0.3 | 27.86 ± 0.02 (CI ±0.02) | 28.34 | 21.83 | 0.9288 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@predicted | 0.3 | 27.77 ± 0.01 (CI ±0.02) | 28.22 | 21.76 | 0.9277 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@random | 0.3 | 25.33 ± 0.03 (CI ±0.04) | 25.45 | 19.77 | 0.9012 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@taylor | 0.3 | 26.89 ± 0.02 (CI ±0.02) | 27.20 | 20.98 | 0.9191 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@variance | 0.3 | 25.84 ± 0.04 (CI ±0.04) | 26.00 | 20.22 | 0.9101 | 0.997 | 50.00% | 16 | 51.56% |
| uniform | 0.3 | 28.63 ± 0.05 (CI ±0.07) | 30.02 | 18.85 | 0.9254 | 0.205 | 50.00% | 0 | 50.00% |
| no_rs | 0.4 | 19.62 ± 0.02 (CI ±0.02) | 19.56 | 16.60 | 0.7685 | 1.000 | 0.00% | 0 | 0.00% |
| three_tier_25_50_25@attention | 0.4 | 21.45 ± 0.03 (CI ±0.03) | 20.90 | 17.14 | 0.8247 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@energy | 0.4 | 23.63 ± 0.03 (CI ±0.03) | 23.16 | 18.42 | 0.8622 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@l1 | 0.4 | 23.60 ± 0.02 (CI ±0.03) | 23.13 | 18.44 | 0.8627 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@oracle | 0.4 | 23.56 ± 0.03 (CI ±0.04) | 22.98 | 18.65 | 0.8590 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@oracle_perimage | 0.4 | 23.78 ± 0.03 (CI ±0.03) | 23.22 | 18.77 | 0.8634 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@predicted | 0.4 | 23.70 ± 0.03 (CI ±0.04) | 23.15 | 18.74 | 0.8620 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@random | 0.4 | 21.65 ± 0.01 (CI ±0.02) | 21.20 | 17.59 | 0.8263 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@taylor | 0.4 | 23.02 ± 0.03 (CI ±0.03) | 22.58 | 18.22 | 0.8522 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@variance | 0.4 | 22.11 ± 0.02 (CI ±0.02) | 21.75 | 17.68 | 0.8380 | 1.000 | 50.00% | 16 | 51.56% |
| uniform | 0.4 | 20.72 ± 0.04 (CI ±0.04) | 19.69 | 16.63 | 0.7852 | 0.890 | 50.00% | 0 | 50.00% |
| no_rs | 0.5 | 18.29 ± 0.02 (CI ±0.03) | 18.29 | 15.36 | 0.7180 | 1.000 | 0.00% | 0 | 0.00% |
| three_tier_25_50_25@attention | 0.5 | 18.85 ± 0.01 (CI ±0.01) | 18.73 | 15.60 | 0.7459 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@energy | 0.5 | 20.25 ± 0.02 (CI ±0.03) | 20.22 | 15.75 | 0.7762 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@l1 | 0.5 | 20.23 ± 0.02 (CI ±0.03) | 20.22 | 15.74 | 0.7767 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@oracle | 0.5 | 20.19 ± 0.02 (CI ±0.03) | 20.15 | 15.91 | 0.7741 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@oracle_perimage | 0.5 | 20.35 ± 0.02 (CI ±0.03) | 20.39 | 15.84 | 0.7776 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@predicted | 0.5 | 20.30 ± 0.02 (CI ±0.02) | 20.33 | 15.86 | 0.7764 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@random | 0.5 | 19.12 ± 0.02 (CI ±0.02) | 19.03 | 15.75 | 0.7502 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@taylor | 0.5 | 19.91 ± 0.02 (CI ±0.03) | 19.74 | 15.83 | 0.7700 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@variance | 0.5 | 19.31 ± 0.02 (CI ±0.02) | 19.17 | 15.65 | 0.7570 | 1.000 | 50.00% | 16 | 51.56% |
| uniform | 0.5 | 18.30 ± 0.02 (CI ±0.03) | 18.29 | 15.36 | 0.7181 | 0.999 | 50.00% | 0 | 50.00% |

# Paired difference to `uniform` (same loss mask per seed)

| Method | erasure_rate | mean gain (dB) | std over seeds | 95% CI ± | wins | significant |
|---|---|---|---|---|---|---|
| no_rs | 0.3 | -7.49 | 0.06 | 0.08 | 0% | yes |
| three_tier_25_50_25@attention | 0.3 | -3.08 | 0.05 | 0.06 | 0% | yes |
| three_tier_25_50_25@energy | 0.3 | -0.92 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_50_25@l1 | 0.3 | -0.94 | 0.05 | 0.06 | 0% | yes |
| three_tier_25_50_25@oracle | 0.3 | -0.94 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_50_25@oracle_perimage | 0.3 | -0.77 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_50_25@predicted | 0.3 | -0.86 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_50_25@random | 0.3 | -3.30 | 0.05 | 0.06 | 0% | yes |
| three_tier_25_50_25@taylor | 0.3 | -1.74 | 0.06 | 0.07 | 0% | yes |
| three_tier_25_50_25@variance | 0.3 | -2.79 | 0.04 | 0.05 | 0% | yes |
| no_rs | 0.4 | -1.10 | 0.03 | 0.03 | 0% | yes |
| three_tier_25_50_25@attention | 0.4 | +0.74 | 0.03 | 0.03 | 100% | yes |
| three_tier_25_50_25@energy | 0.4 | +2.91 | 0.02 | 0.02 | 100% | yes |
| three_tier_25_50_25@l1 | 0.4 | +2.89 | 0.03 | 0.03 | 100% | yes |
| three_tier_25_50_25@oracle | 0.4 | +2.85 | 0.02 | 0.03 | 100% | yes |
| three_tier_25_50_25@oracle_perimage | 0.4 | +3.06 | 0.02 | 0.03 | 100% | yes |
| three_tier_25_50_25@predicted | 0.4 | +2.99 | 0.02 | 0.03 | 100% | yes |
| three_tier_25_50_25@random | 0.4 | +0.94 | 0.03 | 0.04 | 100% | yes |
| three_tier_25_50_25@taylor | 0.4 | +2.30 | 0.03 | 0.03 | 100% | yes |
| three_tier_25_50_25@variance | 0.4 | +1.39 | 0.03 | 0.04 | 100% | yes |
| no_rs | 0.5 | -0.01 | 0.00 | 0.00 | 0% | yes |
| three_tier_25_50_25@attention | 0.5 | +0.56 | 0.02 | 0.02 | 100% | yes |
| three_tier_25_50_25@energy | 0.5 | +1.95 | 0.03 | 0.04 | 100% | yes |
| three_tier_25_50_25@l1 | 0.5 | +1.93 | 0.03 | 0.03 | 100% | yes |
| three_tier_25_50_25@oracle | 0.5 | +1.90 | 0.03 | 0.04 | 100% | yes |
| three_tier_25_50_25@oracle_perimage | 0.5 | +2.05 | 0.03 | 0.03 | 100% | yes |
| three_tier_25_50_25@predicted | 0.5 | +2.00 | 0.02 | 0.03 | 100% | yes |
| three_tier_25_50_25@random | 0.5 | +0.82 | 0.02 | 0.03 | 100% | yes |
| three_tier_25_50_25@taylor | 0.5 | +1.62 | 0.03 | 0.04 | 100% | yes |
| three_tier_25_50_25@variance | 0.5 | +1.01 | 0.03 | 0.04 | 100% | yes |

# Insights

- Based on 5 channel seed(s); 10000 images per seed. All gains below are paired per seed against the baseline (same packet losses), so they carry no luck-of-the-draw advantage.
- Break-even: the best protected method overtakes `uniform` from about erasure_rate = 0.32.
- Low-loss penalty: even the best method is 0.77 dB below `uniform` at erasure_rate = 0.3 (three_tier_25_50_25@oracle_perimage). This is the block-length effect of a single large RS block, which tiered coding gives up.
- Peak gain: +3.06 dB at erasure_rate = 0.4 by three_tier_25_50_25@oracle_perimage (95% CI ±0.03, significant).
- Importance sources ranked by their best scheme's mean gain over erasure_rate >= 0.35: oracle_perimage +2.56 dB, predicted +2.50 dB, energy +2.43 dB, l1 +2.41 dB, oracle +2.37 dB, taylor +1.96 dB, variance +1.20 dB, random +0.88 dB, attention +0.65 dB.
- Share of the reference ranking's gain captured: predicted 98%, energy 95%, l1 94%, oracle 93%, taylor 77%, variance 47%, random 34%, attention 25%. A share above 100% means that source allocated protection better than the reference.
- Random-order control gains +0.88 dB versus oracle +2.37 dB, so the benefit comes from *which* channels are protected, not merely from tiering.
- Worst-case images at erasure_rate = 0.4: 5th-percentile PSNR 18.77 dB (three_tier_25_50_25@oracle_perimage) vs 16.63 dB (uniform).
- Worst-case images at erasure_rate = 0.5: 5th-percentile PSNR 15.84 dB (three_tier_25_50_25@oracle_perimage) vs 15.36 dB (uniform).
- Seed sensitivity: std across channel seeds stays below 0.3 dB everywhere.
- `uniform` shows its steepest fall (-7.92 dB) between erasure_rate = 0.3 and 0.4 -- its cliff.
