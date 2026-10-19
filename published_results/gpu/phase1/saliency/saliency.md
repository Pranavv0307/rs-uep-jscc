# Feature-saliency proxies vs true leave-one-out damage

| proxy | within-image Spearman (95% CI) | top-1 hit | top-4 overlap | verdict |
|---|---|---|---|---|
| attention | -0.015 (-0.02..-0.01) | 0.00 | 0.23 | no better than chance |
| energy | +0.843 (+0.84..+0.85) | 0.40 | 0.67 | useful |
| variance | +0.438 (+0.43..+0.45) | 0.07 | 0.46 | weak |
| l1 | +0.840 (+0.84..+0.84) | 0.45 | 0.68 | useful |
| taylor | +0.626 (+0.62..+0.63) | 0.31 | 0.64 | useful |
| random | +0.002 (-0.01..+0.01) | 0.06 | 0.25 | no better than chance |
| static_train_ranking | +0.893 (+0.89..+0.90) | 0.46 | 0.88 | useful |

## Insights

- Proxies ranked by within-image Spearman with the true damage: static_train_ranking +0.89, energy +0.84, l1 +0.84, taylor +0.63, variance +0.44, random +0.00, attention -0.02.
- Static train ranking reaches +0.89: how much of the per-image structure a single global table already explains.
- Best proxy beyond the static table: energy (top-4 overlap 0.68 vs chance 0.25).
