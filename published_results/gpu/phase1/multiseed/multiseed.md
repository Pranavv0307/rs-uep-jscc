# Multi-seed oracle comparison

| model | clean val PSNR | top-4 channels | worst (dB) | least (dB) | top-4 share | bands |
|---|---|---|---|---|---|---|
| seed42 | 30.84 | [10, 8, 1, 7] | 8.67 | 0.92 | 50% | {'catastrophic (>10 dB)': 0, 'high (5-10 dB)': 5, 'medium (2-5 dB)': 8, 'low (<2 dB)': 3} |
| seed100 | 31.30 | [3, 7, 1, 14] | 13.24 | 1.84 | 45% | {'catastrophic (>10 dB)': 1, 'high (5-10 dB)': 3, 'medium (2-5 dB)': 11, 'low (<2 dB)': 1} |
| seed7 | 31.40 | [4, 15, 2, 6] | 10.47 | 2.16 | 47% | {'catastrophic (>10 dB)': 1, 'high (5-10 dB)': 2, 'medium (2-5 dB)': 13, 'low (<2 dB)': 0} |
| seed13 | 31.71 | [1, 15, 7, 11] | 17.10 | 1.58 | 53% | {'catastrophic (>10 dB)': 1, 'high (5-10 dB)': 4, 'medium (2-5 dB)': 10, 'low (<2 dB)': 1} |
| seed21 | 31.04 | [11, 2, 0, 9] | 15.91 | 1.13 | 52% | {'catastrophic (>10 dB)': 1, 'high (5-10 dB)': 2, 'medium (2-5 dB)': 11, 'low (<2 dB)': 2} |

## Insights

- Most damaging channel per model: seed42 8.7 dB (channel 10), seed100 13.2 dB (channel 3), seed7 10.5 dB (channel 4), seed13 17.1 dB (channel 1), seed21 15.9 dB (channel 11).
- Index-aligned Spearman between models averages -0.13: channel labels do permute between initialisations, as expected.
- Sorted damage profiles agree: mean Pearson 0.922, mean absolute difference 0.83 dB per channel.
- NOT every model has a >10 dB channel: the 'one catastrophic channel' claim needs softening.
- Share of total damage carried by the top-4 channels: seed42 50%, seed100 45%, seed7 47%, seed13 53%, seed21 52%.
