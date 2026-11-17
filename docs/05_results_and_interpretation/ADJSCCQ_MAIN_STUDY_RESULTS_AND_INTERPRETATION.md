​
# Consolidated Results and What They Mean

## Executive conclusion

This project studies sensitivity-aware unequal error protection (UEP) for
quantized Deep JSCC over packet-erasure channels.

The final evidence supports this conclusion:

> Latent channels have strongly unequal erasure sensitivity. Protecting the
> most damaging channels more strongly improves reconstruction after severe
> packet loss. ADJSCC attention gates are not a reliable erasure-sensitivity
> signal; latent energy, a checkpoint-specific leave-one-out ranking, and a
> small sensitivity head are much more useful.

The benefit is regime-dependent:

- Uniform RS is usually better at low loss because one large code block is
  more efficient.
- Ranked UEP becomes better at roughly 32-35% loss, after the uniform-RS
  correction cliff.
- The best primary CIFAR-10 test gain is about **+3.00 dB at 40% loss**.
- The ranking transfers strongly to CIFAR-100 and remains useful on Kodak,
  although Kodak uses a different resolution and RS block regime.

Primary sources:

- [`README.md`](../../README.md)
- [`published_results/gpu/report/INSIGHTS.md`](../../published_results/gpu/report/INSIGHTS.md)
- [`published_results/gpu/phase1/transfer/transfer.md`](../../published_results/gpu/phase1/transfer/transfer.md)
- [`published_results/gpu/phase3/kodak/kodak.md`](../../published_results/gpu/phase3/kodak/kodak.md)

---

## 1. Experimental protocol

The evaluated pipeline is:

```text
image -> ADJSCC encoder -> 256-level quantizer -> byte latent symbols
      -> channel ordering / UEP tiers -> RS packetization
      -> packet-erasure channel -> RS decoding and zero fallback
      -> ADJSCC decoder -> reconstruction
```

The final CIFAR protocol uses 40,000 training images, 10,000 validation
images, 10,000 test images, validation-only checkpoint/scheme selection,
five channel seeds, paired packet masks, and mean per-image PSNR.

The official test set is reserved for final evaluation. Failed RS blocks are
filled with `0.0` in latent space; no ground truth reaches the receiver.

Importance sources evaluated:

- `uniform`: equal protection;
- `random`: random channel-to-tier assignment;
- `attention`: ADJSCC bottleneck attention ranking;
- `oracle`: one frozen training-derived LOO ranking;
- `oracle_perimage`: true per-image LOO upper bound;
- `energy`, `l1`, `variance`, `taylor`: inexpensive latent proxies;
- `predicted`: learned sensitivity head.

---

## 2. Leave-one-out (LOO) sensitivity

For every image and latent channel \(c\):

1. reconstruct from the clean hard-quantized latent;
2. set channel \(c\) to zero;
3. reconstruct again;
4. record:

```text
damage(c) = clean PSNR - PSNR after erasing channel c
```

The implementation is in
[`experiments/sensitivity.py`](../../experiments/sensitivity.py).

The quantized path is intentional: it measures sensitivity in the same
representation that is packetized and transmitted.

### Results

- The 16 channels are not equally important.
- Across final model seeds, the top four channels carry approximately
  **45%-53% of total channel damage**.
- The largest channel damage varies by model, approximately **8.7-17.1 dB**.
- Sorted damage profiles agree across seeds with mean Pearson correlation
  approximately **0.922**.
- Exact channel indices differ between independently trained checkpoints.

The correct interpretation is:

```text
importance concentration is repeatable;
the numerical channel label is checkpoint-specific.
```

Channel labels can permute because a channel index is only an internal tensor
position. Reordering the outputs of one layer and the corresponding inputs of
the next layer can preserve the end-to-end function.

Therefore, a ranking must be tied to the exact checkpoint that produced it.

---

## 3. Attention versus true damage

Final validation results from 2,000 images:

| Metric | Result |
|---|---:|
| Pooled Pearson | +0.118 |
| Within-image Spearman | **-0.015** |
| 95% CI for within-image Spearman | about -0.020 to -0.011 |
| Attention top-4 overlap with true top-4 | **0.23** |
| Random top-4 overlap | 0.25 |

Implementation:
[`experiments/attention_vs_damage.py`](../../experiments/attention_vs_damage.py).

Meaning: the attention gate was trained for the model's end-to-end
rate/distortion objective, not for the question “what happens if this whole
channel is erased?” A channel useful for continuous noise or texture
allocation need not be the channel whose packet loss causes the largest
structural reconstruction failure.

---

## 4. Primary CIFAR-10 RS-UEP results

Best validation-selected tiered gain over uniform RS on the CIFAR-10 test
set:

| Model | 20% loss | 30% loss | 40% loss | 50% loss |
|---|---:|---:|---:|---:|
| seed 42 | -1.15 dB | -0.87 dB | **+3.00 dB** | +2.10 dB |
| seed 100 | -1.66 dB | -1.46 dB | +2.83 dB | +2.04 dB |
| seed 7 | -1.78 dB | -1.60 dB | +2.59 dB | +1.86 dB |
| seed 13 | -1.65 dB | -1.32 dB | +3.23 dB | +2.23 dB |
| seed 21 | -1.41 dB | -1.09 dB | +3.11 dB | +2.18 dB |
| seed 2024 (not converged) | -0.87 dB | -0.74 dB | +2.09 dB | +1.42 dB |

The best tiered method overtakes uniform RS at approximately **32% loss**.

### Why UEP loses at low loss

Uniform RS uses a large code block and therefore uses parity efficiently below
its correction cliff. UEP divides data into smaller blocks and gives some
tiers less parity; a low-priority tier may fail while uniform RS still
recovers everything.

### Why UEP wins at high loss

After uniform RS collapses, UEP can preserve high-priority channels while
allowing low-priority channels to fail. The decoder then loses less
reconstruction-critical information.

### Importance-source comparison

Mean gain over uniform for loss rates at or above 35%:

| Source | Mean gain |
|---|---:|
| Predicted sensitivity head | **+2.31 dB** |
| Energy | +2.24 dB |
| L1 | +2.22 dB |
| Static oracle | +2.19 dB |
| Random order | +0.50 dB |
| Attention | +0.31 dB |

Random UEP is the control proving that the benefit is not merely caused by
splitting the code into tiers. The identity of the protected channels
matters.

---

## 5. CIFAR-10 to CIFAR-100 transfer

The model was frozen. LOO damage was measured on CIFAR-10 validation images
and 2,000 CIFAR-100 test images. The model was not retrained on CIFAR-100.

| Metric | Result |
|---|---:|
| Spearman | **+0.997** |
| Spearman 95% CI | +0.991 to +1.000 |
| Pearson | +0.997 |
| Kendall | +0.983 |
| Top-4 overlap | **1.00** |
| Worst channel in both datasets | channel 10 |
| Mean target damage | 3.785 dB |

Shared top four:

```text
{10, 8, 1, 7}
```

Channel damage:

| Channel | CIFAR-10 (dB) | CIFAR-100 (dB) |
|---:|---:|---:|
| 10 | 8.665 | 7.974 |
| 8 | 8.295 | 7.813 |
| 1 | 7.282 | 7.509 |
| 7 | 6.629 | 6.733 |
| 4 | 5.037 | 4.975 |
| 9 | 4.553 | 4.523 |
| 12 | 3.242 | 3.226 |
| 15 | 2.998 | 2.954 |
| 5 | 2.854 | 2.953 |
| 13 | 2.514 | 2.385 |
| 3 | 2.076 | 1.992 |
| 6 | 2.018 | 1.959 |
| 14 | 2.009 | 1.880 |
| 2 | 1.521 | 1.414 |
| 0 | 1.508 | 1.419 |
| 11 | 0.924 | 0.858 |

Meaning: the absolute damage changed somewhat, but the ordering was almost
unchanged. For this checkpoint and related datasets, a fixed ranking can be
reused with zero per-image side-information. This should not be generalized
to arbitrary domains without testing.

Implementation:
[`experiments/transfer_sensitivity.py`](../../experiments/transfer_sensitivity.py).

---

## 6. Practical importance predictors

### Cheap proxies

| Source | Within-image Spearman with true damage |
|---|---:|
| Static training ranking | +0.89 |
| Energy | +0.84 |
| L1 | +0.84 |
| Taylor | +0.63 |
| Variance | +0.44 |
| Random | approximately 0.00 |
| Attention | -0.02 |

Energy and L1 are attractive because they require no extra decoder passes.

### Sensitivity head

The sensitivity head maps the current latent to predicted damage for each
channel:

```text
latent -> head -> predicted damage vector -> UEP ordering
```

| Metric | Result |
|---|---:|
| Top-4 overlap | 0.920 ± 0.001 |
| Static-ranking top-4 overlap | 0.88 |
| Random top-4 overlap | 0.25 |
| Protection regret | 0.21 dB |
| Extra computation | 2,048 MACs |
| Fraction of encoder MACs | 0.039% |

The head is defined in
[`models/adjsccq.py`](../../models/adjsccq.py) and trained by
[`experiments/train_sensitivity_head.py`](../../experiments/train_sensitivity_head.py).

Meaning: LOO can provide offline training labels while deployment uses a
cheap one-pass predictor.

---

## 7. Kodak high-resolution evaluation

### What changed

The main Kodak run used the CIFAR-trained seed-42 checkpoint without
retraining it for Kodak:

| Quantity | CIFAR | Kodak |
|---|---:|---:|
| Image size | 32 x 32 | 768 x 512 |
| Latent channels | 16 | 16 |
| Latent spatial size | small | 128 x 192 |
| Packet size | CIFAR setting | 128 bytes |
| Data packets | small | 3,072 |
| Transmitted packets | small | 4,608 |
| Coding overhead | 50% | 50% |
| Images | 10,000 in primary test | 24 Kodak images |
| Channel seeds | 5 | 5 |

Portrait Kodak images were rotated to landscape without cropping or resizing.
The model is fully convolutional and ran at native resolution.

GF(256) RS codewords have at most 255 symbols, so large Kodak tiers were
split into legal sub-blocks of at most 255 packets. This changes finite-length
coding behavior compared with the small CIFAR system.

Implementation:
[`experiments/eval_kodak.py`](../../experiments/eval_kodak.py).

### Kodak results

| Packet loss | Uniform RS | Best oracle UEP | Gain |
|---:|---:|---:|---:|
| 0% | 24.31 dB | 24.31 dB | +0.00 dB |
| 10% | 24.31 dB | 24.32 dB | +0.01 dB |
| 20% | 24.31 dB | 24.33 dB | +0.01 dB |
| 30% | 23.88 dB | 24.04 dB | +0.16 dB |
| 35% | 21.73 dB | 22.65 dB | **+0.92 dB** |
| 40% | 20.24 dB | 21.66 dB | **+1.43 dB** |
| 45% | 19.64 dB | 21.11 dB | **+1.46 dB** |
| 50% | 19.05 dB | 19.82 dB | **+0.76 dB** |

Clean Kodak quality:

```text
PSNR: 24.31 dB
SSIM: 0.6015
```

High-loss mean gains:

| Source | Mean gain over uniform for loss >= 35% |
|---|---:|
| Frozen oracle | **+1.14 dB** |
| Attention | +0.14 dB |

Kodak showed essentially no low-loss penalty: the best method was never more
than about 0.05 dB below uniform. The large packet population and legal RS
sub-block splitting soften the sharp small-block threshold.

Important limitation: the published Kodak experiment shows behavioral
transfer of the frozen ranking through UEP performance. It does not publish a
direct CIFAR-ranking-versus-Kodak-ranking LOO correlation coefficient.

---

## 8. Robust decoder and inpainting experiments

| Model | Worst low-loss penalty | Peak high-loss gain |
|---|---:|---:|
| Base | -0.87 dB | +3.00 dB |
| Robust channel | -0.73 dB | +2.51 dB |
| Robust packet | -0.64 dB | +1.17 dB |
| Robust inpainting | -0.41 dB | +1.22 dB |

Meaning: erasure-aware training reduced the low-loss penalty but also reduced
the peak UEP advantage. The desired outcome—remove the penalty while keeping
the full high-loss gain—was not achieved.

---

## 9. Burst losses

Gilbert-Elliott burst evaluation found:

- interleaving tiers across the packet timeline preserves more UEP gain for
  long bursts;
- sequential tier transmission can let one burst destroy one tier;
- the target of more than 3 dB at long burst lengths was not met;
- observed gains were approximately 0.5-2.5 dB.

Meaning: importance ranking alone is insufficient for burst channels; packet
layout and temporal interleaving also matter.

---

## 10. Digital compression baselines

Clean quality:

| Method | Clean PSNR |
|---|---:|
| JPEG + RS | 36.76 dB |
| JPEG 2000 + RS | 32.91 dB |
| WebP + RS | 38.14 dB |
| Oracle Deep JSCC UEP | 30.88 dB |

At 35% packet loss:

| Method | PSNR |
|---|---:|
| JPEG + RS | 21.64 dB |
| JPEG 2000 + RS | 20.74 dB |
| WebP + RS | 23.04 dB |
| Oracle Deep JSCC UEP | 25.72 dB |

Meaning: digital codecs are better in clean conditions, while ranked Deep
JSCC is more graceful under severe packet loss.

---

## 11. Overall project meaning

### Supported conclusions

1. Latent importance is unequal.
2. The importance profile is structurally repeatable, but channel indices are
   checkpoint-specific.
3. Attention does not predict erasure damage for the evaluated model family.
4. UEP is most useful after the uniform-RS cliff.
5. The identity of protected channels matters; random tiering is much weaker.
6. A frozen ranking transfers strongly to CIFAR-100 and remains useful on
   Kodak.
7. A tiny sensitivity head provides a practical adaptive alternative.
8. RS finite-field limits, block size, and interleaving materially affect the
   results.

### Claims to avoid

- Do not say the same numerical channel index is important in every model.
- Do not say UEP always beats uniform RS.
- Do not say attention is universally meaningless for every architecture or
  channel model.
- Do not claim a Kodak LOO correlation coefficient was measured unless that
  separate analysis is run.
- Do not mix preliminary Phase 5 numbers with final tables without identifying
  the protocol.

---

## 12. Preliminary versus final results

Earlier Phase 5 documents reported preliminary values such as approximately
`+2.83 dB` at 40% loss and a fixed ranking beginning
`[3, 7, 4, 6, ...]`. Those runs preceded the final end-to-end protocol.

The final protocol standardized:

- shared packet-loss masks;
- validation-only scheme selection;
- five channel seeds;
- train/validation/test separation;
- test-access logging;
- corrected uniform and no-RS baselines;
- multi-seed analysis;
- paired gains and confidence intervals.

For final conclusions, prefer the published GPU report and README listed at
the top of this document.

---

## Recommended one-paragraph project summary

> Quantized Deep JSCC produces latent channels with strongly unequal
> erasure sensitivity. A leave-one-out analysis reveals which channels are
> reconstruction-critical, and protecting those channels with unequal
> Reed-Solomon redundancy improves PSNR by up to about 3 dB at 40% packet loss
> on the primary CIFAR-10 test protocol. Uniform RS remains better at lower
> loss because it uses one large code block more efficiently. ADJSCC attention
> gates do not predict erasure damage, while latent energy and a tiny learned
> sensitivity head provide effective practical alternatives. The
> model-specific ranking transfers strongly to CIFAR-100 and remains useful
> on full-resolution Kodak images, where packetization and finite-field
> effects produce smaller gains.
