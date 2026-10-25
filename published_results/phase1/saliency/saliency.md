# Feature-saliency proxies vs true leave-one-out damage

| proxy | within-image Spearman (95% CI) | top-1 hit | top-4 overlap | verdict |
|---|---|---|---|---|
| attention | +0.180 (+0.17..+0.19) | 0.00 | 0.33 | no better than chance |
| energy | +0.819 (+0.81..+0.82) | 0.59 | 0.73 | useful |
| variance | +0.348 (+0.34..+0.36) | 0.04 | 0.41 | weak |
| l1 | +0.809 (+0.80..+0.81) | 0.61 | 0.72 | useful |
| taylor | +0.463 (+0.45..+0.47) | 0.77 | 0.54 | weak |
| random | -0.003 (-0.01..+0.01) | 0.06 | 0.25 | no better than chance |
| static_train_ranking | +0.748 (+0.74..+0.75) | 0.98 | 0.76 | useful |

## Insights

- Proxies ranked by within-image Spearman with the true damage: energy +0.82, l1 +0.81, static_train_ranking +0.75, taylor +0.46, variance +0.35, attention +0.18, random -0.00.
- Static train ranking reaches +0.75: how much of the per-image structure a single global table already explains.
- Best proxy beyond the static table: energy (top-4 overlap 0.73 vs chance 0.25).
