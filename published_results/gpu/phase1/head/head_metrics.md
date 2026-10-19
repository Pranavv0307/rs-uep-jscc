# Sensitivity head (validation split)

| ranking | top-4 overlap | top-1 hit | within-image Spearman | regret (dB) |
|---|---|---|---|---|
| head (mean over seeds) | 0.920 | 0.873 | +0.946 | 0.21 |
| random | 0.250 | 0.060 | -0.001 | 15.87 |
| static_train_ranking | 0.881 | 0.463 | +0.892 | 0.60 |
| energy | 0.666 | 0.416 | +0.840 | 3.45 |
| variance | 0.452 | 0.079 | +0.436 | 9.38 |
| attention | 0.231 | 0.000 | -0.013 | 16.76 |

## Insights

- Head top-4 overlap on validation: 0.920 ± 0.001 over 5 head seeds (random 0.25, static table 0.88).
- Protection regret: head 0.21 dB vs static table 0.60 dB vs random 15.87 dB per image.
- The head costs 2048 MACs = 0.039% of the encoder.
- The head beats the static table on top-4 overlap (+0.039); if images share one ranking the online predictor is unnecessary.
