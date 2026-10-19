# Sensitivity head (validation split)

| ranking | top-4 overlap | top-1 hit | within-image Spearman | regret (dB) |
|---|---|---|---|---|
| head (mean over seeds) | 0.883 | 0.985 | +0.890 | 0.27 |
| random | 0.249 | 0.057 | -0.004 | 13.02 |
| static_train_ranking | 0.755 | 0.977 | +0.746 | 1.68 |
| energy | 0.729 | 0.581 | +0.818 | 1.73 |
| variance | 0.412 | 0.032 | +0.350 | 10.51 |
| attention | 0.331 | 0.001 | +0.183 | 12.83 |

## Insights

- Head top-4 overlap on validation: 0.883 ± 0.001 over 5 head seeds (random 0.25, static table 0.75).
- Protection regret: head 0.27 dB vs static table 1.68 dB vs random 13.02 dB per image.
- The head costs 2048 MACs = 0.039% of the encoder.
- The head beats the static table on top-4 overlap (+0.128); if images share one ranking the online predictor is unnecessary.
