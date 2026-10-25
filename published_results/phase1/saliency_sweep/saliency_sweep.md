# cifar10/val erasure sweep

- commit `7fe6f706fbf5abef0ea54f8c8081dcf2075d3a26` (dirty: False), 2026-10-17T20:02:18+00:00
- checkpoint `results/local_models/seed42/best_val.pth` sha256 `35ee3f09d6643f527e65ed6ffda70037ea761d049d089cd321308835abe9c09f`
- images_per_seed: 10000
- is_full_test_set: False
- channels: 16
- packet_size: 16
- overhead: 0.5
- clean_psnr_mean: 30.524097442626953
- side_info_rule: 0 B for uniform/no_rs, one byte per channel for sorted methods
- channel_seeds: [0, 1, 2, 3, 4]
- mask_protocol: one master mask per (batch, seed, rate) shared by all methods; smaller schemes use its first columns
- PSNR: mean ± sample std over channel seeds (95% t-interval half-width in brackets)

| Method | erasure_rate | PSNR (dB) | median | p5 | SSIM | block-fail | Coding OH | Side-info B | Total OH |
|---|---|---|---|---|---|---|---|---|---|
| no_rs | 0.3 | 21.68 ± 0.02 (CI ±0.02) | 21.61 | 18.52 | 0.8190 | 1.000 | 0.00% | 0 | 0.00% |
| three_tier_25_50_25@attention | 0.3 | 25.66 ± 0.03 (CI ±0.03) | 25.85 | 20.29 | 0.9040 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@energy | 0.3 | 27.02 ± 0.02 (CI ±0.02) | 27.35 | 21.34 | 0.9213 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@l1 | 0.3 | 26.99 ± 0.02 (CI ±0.02) | 27.30 | 21.32 | 0.9209 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@oracle | 0.3 | 26.62 ± 0.02 (CI ±0.02) | 27.10 | 20.38 | 0.9141 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@oracle_perimage | 0.3 | 27.06 ± 0.02 (CI ±0.02) | 27.43 | 21.16 | 0.9202 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@predicted | 0.3 | 26.91 ± 0.02 (CI ±0.02) | 27.25 | 21.07 | 0.9188 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@random | 0.3 | 25.52 ± 0.03 (CI ±0.04) | 25.75 | 20.09 | 0.8982 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@taylor | 0.3 | 26.36 ± 0.02 (CI ±0.02) | 26.58 | 20.94 | 0.9117 | 0.997 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@variance | 0.3 | 25.87 ± 0.03 (CI ±0.04) | 26.19 | 20.26 | 0.9038 | 0.997 | 50.00% | 16 | 51.56% |
| uniform | 0.3 | 28.50 ± 0.05 (CI ±0.06) | 29.67 | 19.43 | 0.9240 | 0.205 | 50.00% | 0 | 50.00% |
| no_rs | 0.4 | 20.24 ± 0.02 (CI ±0.02) | 20.19 | 17.24 | 0.7732 | 1.000 | 0.00% | 0 | 0.00% |
| three_tier_25_50_25@attention | 0.4 | 22.20 ± 0.02 (CI ±0.02) | 21.86 | 18.21 | 0.8309 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@energy | 0.4 | 23.19 ± 0.03 (CI ±0.03) | 22.69 | 18.78 | 0.8539 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@l1 | 0.4 | 23.15 ± 0.03 (CI ±0.03) | 22.63 | 18.77 | 0.8531 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@oracle | 0.4 | 22.44 ± 0.02 (CI ±0.03) | 21.77 | 18.06 | 0.8412 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@oracle_perimage | 0.4 | 23.08 ± 0.02 (CI ±0.02) | 22.52 | 18.69 | 0.8510 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@predicted | 0.4 | 22.96 ± 0.03 (CI ±0.03) | 22.39 | 18.60 | 0.8489 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@random | 0.4 | 22.10 ± 0.02 (CI ±0.03) | 21.68 | 18.12 | 0.8264 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@taylor | 0.4 | 22.78 ± 0.02 (CI ±0.03) | 22.38 | 18.50 | 0.8432 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@variance | 0.4 | 22.47 ± 0.03 (CI ±0.03) | 22.12 | 18.26 | 0.8350 | 1.000 | 50.00% | 16 | 51.56% |
| uniform | 0.4 | 21.24 ± 0.03 (CI ±0.03) | 20.32 | 17.26 | 0.7894 | 0.890 | 50.00% | 0 | 50.00% |
| no_rs | 0.5 | 18.98 ± 0.02 (CI ±0.02) | 18.99 | 15.93 | 0.7258 | 1.000 | 0.00% | 0 | 0.00% |
| three_tier_25_50_25@attention | 0.5 | 19.81 ± 0.01 (CI ±0.02) | 19.75 | 16.40 | 0.7589 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@energy | 0.5 | 20.25 ± 0.03 (CI ±0.03) | 20.15 | 16.51 | 0.7730 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@l1 | 0.5 | 20.23 ± 0.01 (CI ±0.02) | 20.13 | 16.49 | 0.7723 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@oracle | 0.5 | 19.67 ± 0.01 (CI ±0.02) | 19.49 | 16.37 | 0.7625 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@oracle_perimage | 0.5 | 20.14 ± 0.02 (CI ±0.02) | 20.00 | 16.54 | 0.7704 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@predicted | 0.5 | 20.07 ± 0.02 (CI ±0.03) | 19.92 | 16.51 | 0.7688 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@random | 0.5 | 19.73 ± 0.02 (CI ±0.03) | 19.65 | 16.36 | 0.7551 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@taylor | 0.5 | 20.06 ± 0.02 (CI ±0.03) | 19.95 | 16.44 | 0.7657 | 1.000 | 50.00% | 16 | 51.56% |
| three_tier_25_50_25@variance | 0.5 | 19.95 ± 0.01 (CI ±0.02) | 19.87 | 16.34 | 0.7617 | 1.000 | 50.00% | 16 | 51.56% |
| uniform | 0.5 | 18.98 ± 0.02 (CI ±0.02) | 18.99 | 15.93 | 0.7259 | 0.999 | 50.00% | 0 | 50.00% |

# Paired difference to `uniform` (same loss mask per seed)

| Method | erasure_rate | mean gain (dB) | std over seeds | 95% CI ± | wins | significant |
|---|---|---|---|---|---|---|
| no_rs | 0.3 | -6.82 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_50_25@attention | 0.3 | -2.84 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_50_25@energy | 0.3 | -1.48 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_50_25@l1 | 0.3 | -1.51 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_50_25@oracle | 0.3 | -1.88 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_50_25@oracle_perimage | 0.3 | -1.45 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_50_25@predicted | 0.3 | -1.60 | 0.04 | 0.04 | 0% | yes |
| three_tier_25_50_25@random | 0.3 | -2.98 | 0.05 | 0.06 | 0% | yes |
| three_tier_25_50_25@taylor | 0.3 | -2.14 | 0.04 | 0.05 | 0% | yes |
| three_tier_25_50_25@variance | 0.3 | -2.63 | 0.02 | 0.03 | 0% | yes |
| no_rs | 0.4 | -1.00 | 0.02 | 0.02 | 0% | yes |
| three_tier_25_50_25@attention | 0.4 | +0.95 | 0.02 | 0.03 | 100% | yes |
| three_tier_25_50_25@energy | 0.4 | +1.94 | 0.02 | 0.02 | 100% | yes |
| three_tier_25_50_25@l1 | 0.4 | +1.90 | 0.02 | 0.03 | 100% | yes |
| three_tier_25_50_25@oracle | 0.4 | +1.20 | 0.03 | 0.03 | 100% | yes |
| three_tier_25_50_25@oracle_perimage | 0.4 | +1.83 | 0.02 | 0.03 | 100% | yes |
| three_tier_25_50_25@predicted | 0.4 | +1.71 | 0.02 | 0.03 | 100% | yes |
| three_tier_25_50_25@random | 0.4 | +0.86 | 0.03 | 0.04 | 100% | yes |
| three_tier_25_50_25@taylor | 0.4 | +1.53 | 0.03 | 0.04 | 100% | yes |
| three_tier_25_50_25@variance | 0.4 | +1.22 | 0.03 | 0.04 | 100% | yes |
| no_rs | 0.5 | -0.01 | 0.00 | 0.00 | 0% | yes |
| three_tier_25_50_25@attention | 0.5 | +0.82 | 0.01 | 0.02 | 100% | yes |
| three_tier_25_50_25@energy | 0.5 | +1.27 | 0.02 | 0.02 | 100% | yes |
| three_tier_25_50_25@l1 | 0.5 | +1.24 | 0.01 | 0.01 | 100% | yes |
| three_tier_25_50_25@oracle | 0.5 | +0.69 | 0.01 | 0.01 | 100% | yes |
| three_tier_25_50_25@oracle_perimage | 0.5 | +1.16 | 0.01 | 0.01 | 100% | yes |
| three_tier_25_50_25@predicted | 0.5 | +1.08 | 0.02 | 0.02 | 100% | yes |
| three_tier_25_50_25@random | 0.5 | +0.75 | 0.02 | 0.02 | 100% | yes |
| three_tier_25_50_25@taylor | 0.5 | +1.08 | 0.02 | 0.02 | 100% | yes |
| three_tier_25_50_25@variance | 0.5 | +0.96 | 0.02 | 0.02 | 100% | yes |

# Insights

- Based on 5 channel seed(s); 10000 images per seed. All gains below are paired per seed against the baseline (same packet losses), so they carry no luck-of-the-draw advantage.
- Break-even: the best protected method overtakes `uniform` from about erasure_rate = 0.34.
- Low-loss penalty: even the best method is 1.45 dB below `uniform` at erasure_rate = 0.3 (three_tier_25_50_25@oracle_perimage). This is the block-length effect of a single large RS block, which tiered coding gives up.
- Peak gain: +1.94 dB at erasure_rate = 0.4 by three_tier_25_50_25@energy (95% CI ±0.02, significant).
- Importance sources ranked by their best scheme's mean gain over erasure_rate >= 0.35: energy +1.61 dB, l1 +1.57 dB, oracle_perimage +1.50 dB, predicted +1.40 dB, taylor +1.31 dB, variance +1.09 dB, oracle +0.94 dB, attention +0.89 dB, random +0.80 dB.
- Share of the oracle gain captured: energy 170%, l1 167%, predicted 148%, taylor 139%, variance 116%, attention 94%, random 85%.
- Random-order control gains +0.80 dB versus oracle +0.94 dB, so the benefit comes from *which* channels are protected, not merely from tiering.
- Worst-case images at erasure_rate = 0.4: 5th-percentile PSNR 18.78 dB (three_tier_25_50_25@energy) vs 17.26 dB (uniform).
- Worst-case images at erasure_rate = 0.5: 5th-percentile PSNR 16.51 dB (three_tier_25_50_25@energy) vs 15.93 dB (uniform).
- Seed sensitivity: std across channel seeds stays below 0.3 dB everywhere.
- `uniform` shows its steepest fall (-7.26 dB) between erasure_rate = 0.3 and 0.4 -- its cliff.
