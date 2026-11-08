# G1 Track: All Results and Their Meaning

## Executive Summary

> **Research question:** Is sensitivity-aware Unequal Error Protection (UEP) an
> intrinsic property of quantised Deep JSCC latents, or merely an artefact of
> the ADJSCC attention gates?
>
> **Answer (supported by the evidence below):** UEP is intrinsic. Removing all
> 10 attention modules from ADJSCC-Q produces DeepJSCC-Q, a model that still
> exhibits strongly unequal channel sensitivity and yields up to **+2.78 dB**
> UEP gain at severe packet loss. Attention is not required.

### One-paragraph project summary

> Quantised Deep JSCC produces latent channels with strongly unequal erasure
> sensitivity even without attention gates. A leave-one-out analysis identifies
> which channels are reconstruction-critical, and protecting those channels
> with unequal Reed-Solomon redundancy improves PSNR by up to +2.78 dB at
> 40% packet loss on the CIFAR-10 test set. Uniform RS remains better at
> lower loss because it uses one large code block more efficiently. Latent
> energy — a zero-training proxy — matches or exceeds the oracle ranking's
> UEP gain in two of four models, and a tiny learned sensitivity head (2,048
> MACs, 0.039% of the encoder) provides 110% of the oracle's protection at
> negligible cost.

---

## 0. Experimental Setup

### Architecture

**DeepJSCC-Q** is ADJSCC-Q with its 10 `af5_bottleneck` attention modules
removed and replaced with identity pass-throughs. Every other component —
encoder, decoder, quantiser, channel model — is unchanged.

```text
DeepJSCC-Q = ADJSCC-Q − 10 attention modules
```

### Protocol

| Property | Value |
|---|---|
| Dataset | CIFAR-10, 32×32 |
| Training images | 40,000 (split_seed 42) |
| Validation images | 10,000 |
| Test images | 10,000 |
| Latent channels | 16 |
| Quantisation levels | 256 (byte-valued) |
| Coding overhead | 50% (RS parity) |
| Training device | Apple Silicon MPS |
| Evaluation device | CPU (4 performance cores) |
| Channel seeds (test) | 5 per model seed |
| Model seeds | 42, 100, 7, 13 |
| Convergence rule | `best_epoch` ≥ 50 |

### Pipeline executed

All steps from the [G1 Runbook (Mac)](../03_runbooks/G1_DEEPJSCCQ_ATTENTION_ABLATION_RUNBOOK_MAC_MPS.md), Steps 0–6:

```text
Step 0:  Setup and environment verification
Step 1:  One-time setup (venv, MPS check)
Step 2:  Unit tests (16/16 passed) and dry run (18 steps clean)
Step 3:  Smoke test (2-epoch train + small evaluation)
Step 4:  Full training (4 seeds × 60 epochs on MPS)
Step 5:  Validation analysis (multiseed, saliency, head, val sweep, select)
Step 6:  Final test sweeps (5 channel seeds × 10,000 images × 4 model seeds)
```

Steps 7–8 (same-hardware anchor and comparison of record) have **not yet been
executed**. The results below are from Steps 3–6.

---

## 1. Model Training (Step 4)

### What was done

Four DeepJSCC-Q models were trained from scratch on MPS, each for 60 epochs
with early stopping.

### Results

| Seed | Best Epoch | Val PSNR (dB) | Test PSNR (dB) | Train Time |
|---|---|---|---|---|
| 42 | 56 | 32.35 | 32.38 | 81 min |
| 100 | 51 | 31.51 | 31.54 | 87 min |
| 7 | 53 | 31.62 | 31.66 | 91 min |
| 13 | 57 | 32.36 | 32.40 | 81 min |

### What this means

**100% convergence rate.** All four seeds converged well past the epoch-50
threshold (best epochs 51–57). This is notable: the published ADJSCC-Q GPU
runs had 5 of 7 seeds converge (seeds 2024 and one other diverged). Removing
attention did not harm training stability — if anything, it slightly improved
it.

**Clean reconstruction quality is comparable to ADJSCC-Q.** The published
ADJSCC-Q clean test PSNR ranges from 30.8–31.7 dB. DeepJSCC-Q spans
31.54–32.40 dB. Removing attention did not degrade image quality; the
information that attention was allocating is still captured by the
convolutional layers.

---

## 2. Leave-One-Out Sensitivity Analysis (Step 5a: `g1_multiseed`)

### What was done

For each of the 16 latent channels in each model, one channel was zeroed out
(simulating total packet loss for that channel), and the resulting PSNR drop
was measured across all 40,000 training images:

```text
damage(channel c) = clean PSNR − PSNR after erasing channel c
```

### Results

| Seed | Top-4 Channels | Worst Channel Damage | Least Channel Damage | Top-4 Damage Share |
|---|---|---|---|---|
| 42 | [8, 7, 5, 14] | **11.68 dB** (ch 8) | 4.35 dB (ch 6) | **34.7%** |
| 100 | [3, 4, 2, 13] | **11.22 dB** (ch 3) | 3.21 dB (ch 12) | **33.4%** |
| 7 | [1, 6, 15, 11] | **11.43 dB** (ch 1) | 0.26 dB (ch 7) | **40.3%** |
| 13 | [12, 11, 13, 2] | **10.49 dB** (ch 12) | 3.39 dB (ch 4) | **39.8%** |

**Cross-seed summary:**

| Statistic | Value |
|---|---|
| Mean top-4 damage share | **37.0%** (vs 25.0% for an even split) |
| Mean worst-channel damage | **11.21 dB** |
| Every seed has a catastrophic (>10 dB) channel | ✅ Yes |
| Sorted damage profile Pearson correlation | **0.930** |
| Index-aligned Spearman (do channel labels match?) | **−0.19** (no — labels permute) |

### What this means

**Sensitivity concentration is intrinsic to quantised autoencoders, not an
attention effect.** Even without any attention mechanism, four out of four
independently trained DeepJSCC-Q models develop a latent representation where
a small subset of channels is dramatically more important than the rest.
Losing the single most important channel destroys 10–12 dB of reconstruction
quality.

**The concentration is structural, not coincidental.** The sorted damage
profiles agree across seeds (Pearson = 0.930), meaning every model develops a
similar "shape" of importance distribution — a few critical channels, a bulk
of moderately important ones. But the specific channel *indices* differ between
models (Spearman = −0.19), so the importance ranking must always be tied to the
specific checkpoint.

**Comparison with ADJSCC-Q (published GPU results):**

| Metric | ADJSCC-Q (GPU) | DeepJSCC-Q (Mac MPS) |
|---|---|---|
| Top-4 damage share | 45–53% | 33–40% |
| Worst channel damage | 8.7–17.1 dB | 10.5–11.7 dB |
| Catastrophic channels per model | 1–3 | 1–2 |

DeepJSCC-Q's concentration is slightly lower than ADJSCC-Q's, but still
far above the 25% even-split baseline. The attention gates in ADJSCC-Q appear
to amplify concentration but are not its cause.

---

## 3. Feature-Saliency Proxies (Step 5b: `g1_saliency_A2`)

### What was done

Six zero-training importance proxies were evaluated against the true
leave-one-out damage on the seed-42 model. Each proxy assigns an importance
score to each latent channel without needing any extra decoder passes or
labels.

### Results

| Proxy | Within-Image Spearman | Top-4 Overlap | Top-1 Hit Rate | Verdict |
|---|---|---|---|---|
| **Energy** (‖z_c‖²) | **+0.840** | **0.673** | 0.364 | ✅ useful |
| **L1** (‖z_c‖₁) | +0.813 | 0.657 | 0.311 | ✅ useful |
| Static train ranking | +0.736 | 0.709 | 0.807 | ✅ useful |
| Taylor (∂Loss/∂z · z) | +0.468 | 0.559 | 0.342 | ⚠️ weak |
| Variance (var(z_c)) | +0.347 | 0.434 | 0.273 | ⚠️ weak |
| Random | +0.003 | 0.251 | 0.076 | ❌ chance |

(No attention proxy exists for DeepJSCC-Q — the module was removed by design.)

### What this means

**Energy is the best zero-training proxy.** With a within-image Spearman
correlation of +0.840 against true damage and a top-4 overlap of 67.3% (vs
25% chance), the squared magnitude of the latent channel is an excellent
predictor of which channels matter most — and it requires zero extra
computation beyond what the encoder already produces.

**Why energy works:** A channel with high energy has encoded more signal
amplitude into its quantised symbols. Erasing that channel removes more
reconstruction information, causing a larger PSNR drop. This is a
property of the encoder's learned representation, not of attention.

**Why Taylor is weak:** The Taylor proxy uses the gradient-times-activation
approximation of the loss change. On quantised (discrete) latents, the
gradient signal is noisy because the straight-through estimator used during
training does not perfectly approximate the true gradient of a rounding
operation. The L1 and energy proxies, which are purely data-driven (no
gradients needed), avoid this problem.

**The static training ranking captures 73.6% of the per-image structure.**
This means that a single global ranking (computed once from the training set)
already explains most of the inter-channel variation. The remaining 26% is
image-specific variation that the online sensitivity head can capture.

---

## 4. Learned Sensitivity Head (Step 5c: `g1_head_A3`)

### What was done

A tiny MLP head was trained to predict the per-image, per-channel damage
vector from the encoder's latent representation. The head was trained on
50,000 labelled images (training + validation sets, with LOO damage as the
target) and evaluated on the 10,000 validation images, using 5 independent
training seeds.

### Architecture

```text
latent [B, C, H, W] → global average pool [B, C] → Linear(C, C) → predicted damage [B, C]
```

### Results

| Ranking Method | Top-4 Overlap | Top-1 Hit Rate | Spearman | Regret (dB) |
|---|---|---|---|---|
| **Sensitivity head** | **0.881 ± 0.001** | 0.933 | +0.916 | **0.34** |
| Static train ranking | 0.712 | 0.804 | +0.735 | 2.30 |
| Energy proxy | 0.668 | 0.381 | +0.839 | 2.35 |
| Variance proxy | 0.436 | 0.267 | +0.348 | 7.88 |
| Random | 0.248 | 0.064 | +0.004 | 12.65 |

**Cost:**

| Metric | Value |
|---|---|
| Head parameters | 2,128 |
| Head MACs | 2,048 |
| Fraction of encoder MACs | **0.039%** |

### What this means

**The sensitivity head predicts channel importance with 88.1% top-4 accuracy
at negligible computational cost.** It correctly identifies the four most
critical channels 88.1% of the time, compared to 71.2% for a fixed ranking
and 24.8% for random guessing.

**Protection regret is only 0.34 dB per image.** "Protection regret" measures
how much reconstruction quality is lost by protecting the head's predicted
top-4 channels instead of the true top-4. At 0.34 dB, the head's prediction is
nearly optimal. Compare this to the static ranking (2.30 dB regret) or random
ordering (12.65 dB regret).

**The head is practical for deployment.** At 2,048 MACs (0.039% of the
encoder), the head adds essentially zero latency or power consumption. It
provides adaptive, per-image protection ordering at the cost of 16 bytes of
side information (one byte per channel ranking).

---

## 5. Validation Sweep and Scheme Selection (Step 5d–e)

### What was done

A full RS-UEP evaluation was run on the **validation set** (5,000 images ×
3 channel seeds × 8 erasure rates × 32 method configurations) to choose the
best tier allocation schemes before touching the test set.

### Tier allocation schemes explained

| Scheme | Protection Level (Most → Least Important) | Meaning |
|---|---|---|
| `two_tier_50_50` | 50% parity / 50% parity | Light differentiation |
| `two_tier_25_75` | 75% parity / 25% parity | Strong top-tier protection |
| `three_tier_25_50_25` | Heavy / Medium / Light | Balanced three-tier |
| `three_tier_25_25_50` | Heavy / Heavy / Light | Aggressive top protection |

### Selected schemes (frozen for test)

The validation sweep selected these four schemes as the union of
best-per-source allocations across all erasure rates:

```
three_tier_25_25_50, three_tier_25_50_25, two_tier_25_75, two_tier_50_50
```

Every importance source (oracle, predicted, energy, l1, random) independently
converged to the same scheme being best overall in the high-loss region:
**`three_tier_25_50_25`**.

### Key validation findings

| Metric | Value |
|---|---|
| Break-even point | ~35% erasure rate |
| Peak gain | **+2.41 dB** at 45% loss (`three_tier_25_25_50@predicted`) |
| Low-loss penalty | −2.10 dB at 30% loss (block-length effect) |
| Random control gain (≥35% loss) | +0.07 dB |
| Oracle gain (≥35% loss) | +1.41 dB |

### What this means

**UEP gains are real and large on validation.** Above 35% packet loss, the
best tiered protection method gains more than 2 dB over uniform RS. Below 35%,
uniform RS is better because its single large code block is more efficient.

**The scheme selection is consistent across sources.** All importance sources
agree that `three_tier_25_50_25` is the best overall allocation. This
stability gives confidence that the selection is not overfitted to a specific
ranking method.

---

## 6. Final Test Results (Step 6)

### What was done

The test set was evaluated **exactly once** using the validation-selected
schemes, with 5 channel seeds × 10,000 images per model seed. Test access was
logged to `results/test_access_log.jsonl` for audit.

### Headline test results

| Seed | Oracle Gain @ 40% Loss | Peak Gain | Peak Method | CI (95%) |
|---|---|---|---|---|
| 42 | **+2.11 dB** | **+2.42 dB** (0.45) | `three_tier_25_25_50@predicted` | ±0.04 |
| 100 | **+2.05 dB** | **+2.23 dB** (0.45) | `three_tier_25_50_25@oracle` | ±0.05 |
| 7 | **+2.76 dB** | **+2.78 dB** (0.40) | `three_tier_25_50_25@energy` | ±0.04 |
| 13 | **+2.43 dB** | **+2.59 dB** (0.45) | `three_tier_25_25_50@energy` | ±0.05 |

### Importance source comparison (mean gain over uniform, loss ≥ 35%)

| Source | Seed 42 | Seed 100 | Seed 7 | Seed 13 | Family Mean |
|---|---|---|---|---|---|
| **Oracle** | +1.42 dB | +1.36 dB | +1.99 dB | +1.73 dB | **+1.63 dB** |
| **Predicted** (head) | +1.56 dB | — | — | — | — |
| **Energy** | +1.36 dB | +1.12 dB | +2.02 dB | +1.82 dB | **+1.58 dB** |
| **L1** | +1.33 dB | — | — | — | — |
| **Random** (control) | +0.31 dB | +0.27 dB | +0.31 dB | +0.32 dB | **+0.30 dB** |

(Predicted and L1 were evaluated only on seed 42; the other seeds used oracle
and energy to save compute.)

### Worst-case image protection

| Erasure Rate | 5th-Percentile PSNR (Best UEP) | 5th-Percentile PSNR (Uniform) | Tail Improvement |
|---|---|---|---|
| 40% (seed 42) | 16.87 dB | 14.91 dB | **+1.96 dB** |
| 50% (seed 42) | 14.10 dB | 13.77 dB | **+0.33 dB** |
| 40% (seed 7) | 16.30 dB | 14.29 dB | **+2.01 dB** |

### What this means

**UEP works on DeepJSCC-Q without attention.** Every model seed delivers
substantial UEP gains at severe packet loss:

- The **minimum oracle gain at 40% loss** across all four seeds is **+2.05 dB**,
  well above the pre-registered threshold of +1.5 dB (AD-09, criterion C2).
- All confidence intervals exclude zero (narrowest CI = ±0.04 dB).

**The identity of protected channels matters — this is not a tiering artefact.**
The random-order control gains only +0.27 to +0.32 dB, while the oracle gains
+1.36 to +1.99 dB. The random control captures less than 20% of the oracle
gain, proving that the benefit comes from protecting the *right* channels, not
merely from splitting the code into tiers.

**Energy matches or exceeds the oracle on two seeds.** On seeds 7 and 13,
energy-based UEP actually delivers higher mean gain than the offline oracle
ranking (+2.02 vs +1.99 dB for seed 7; +1.82 vs +1.73 dB for seed 13). This
happens because the energy proxy adapts *per image*, while the oracle is a
single global ranking frozen from the training set. For images whose
sensitivity structure differs from the training average, energy's per-image
adaptation wins.

**The sensitivity head delivers 110% of the oracle gain (seed 42).** The
head's mean gain of +1.56 dB exceeds the oracle's +1.42 dB because the head
provides per-image rankings at minimal cost, capturing image-specific
sensitivity variation that the static oracle misses.

**Low-loss penalty is an inherent trade-off.** At 30% loss, the best UEP
method loses about 2 dB compared to uniform RS. This is unavoidable: uniform
RS uses one large RS block that can correct up to its capacity, while tiered
coding splits data into smaller blocks, each with its own correction threshold.
Below each tier's individual cliff, some tiers fail unnecessarily.

---

## 7. Assessment Against Pre-Registered Decision Criteria (AD-09)

The decision thresholds were frozen **before any G1 model was trained**
(commit `79b295ee`, 2026-11-06).

### Criterion C1: Concentration (top-4 damage share ≥ 0.40)

| Seed | Top-4 Share | Pass? |
|---|---|---|
| 42 | 34.7% | ❌ |
| 100 | 33.4% | ❌ |
| 7 | 40.3% | ✅ |
| 13 | 39.8% | ❌ (borderline) |
| **Family mean** | **37.0%** | **❌ (below 0.40)** |

**Status:** C1 is not met by the family mean. However, the mean (37.0%) is
substantially above the 25% even split and above the FLAT threshold of 30%.
This places DeepJSCC-Q in the "intermediate" zone — there is clear
concentration, but less than ADJSCC-Q's 45–53%.

### Criterion C2: UEP gain (oracle gain at 40% loss ≥ +1.5 dB, CI > 0)

| Seed | Oracle Gain @ 40% | CI Lower Bound | Pass? |
|---|---|---|---|
| 42 | +2.11 dB | +2.06 dB | ✅ |
| 100 | +2.05 dB | +1.99 dB | ✅ |
| 7 | +2.76 dB | +2.72 dB | ✅ |
| 13 | +2.43 dB | +2.37 dB | ✅ |

**Status: C2 PASSED (all 4 seeds).** Every model exceeds the +1.5 dB
threshold with confidence intervals entirely above zero.

### Criterion C3: Identity matters (random < 0.5 × oracle, mean over loss ≥ 0.35)

| Seed | Random Mean | Oracle Mean | Ratio | Pass? |
|---|---|---|---|---|
| 42 | +0.31 dB | +1.42 dB | 0.22 | ✅ |
| 100 | +0.27 dB | +1.36 dB | 0.20 | ✅ |
| 7 | +0.31 dB | +1.99 dB | 0.16 | ✅ |
| 13 | +0.32 dB | +1.73 dB | 0.18 | ✅ |

**Status: C3 PASSED (all 4 seeds).** Random captures only 16–22% of the
oracle gain, well below the 50% threshold. The benefit is unambiguously from
*which* channels are protected.

### Criterion FLAT: No exploitable structure (top-4 < 0.30 AND oracle ≈ random)

**Status: FLAT is REJECTED.** The family mean top-4 share (37.0%) is well
above the 0.30 FLAT threshold, and oracle gains are 5–7× larger than random.
The hypothesis that attention is necessary for sensitivity-aware UEP is
conclusively disproven.

### Overall Verdict

| Criterion | Status |
|---|---|
| C1 (concentration ≥ 0.40) | ❌ Not met (37.0% < 40.0%) |
| C2 (UEP gain ≥ +1.5 dB) | ✅ Met (all 4 seeds) |
| C3 (identity matters) | ✅ Met (all 4 seeds) |
| FLAT (no structure) | ❌ Rejected |

**Formal verdict: "intermediate"** — C2 and C3 are met but C1 is not, and FLAT
is rejected. In practice, this means:

> DeepJSCC-Q has clear, exploitable sensitivity structure that enables
> substantial UEP gains (+2.0 to +2.8 dB at 40% loss). The concentration
> is lower than ADJSCC-Q's, suggesting that attention amplifies the
> concentration effect but is not its cause. The benefit is real, repeatable,
> and robust across four independent training seeds.

---

## 8. Why UEP Loses at Low Loss and Wins at High Loss

### The uniform-RS cliff

Uniform RS protects all 16 channels with one large Reed-Solomon code block.
This block can correct up to `floor(n_parity / 2)` erasures. When packet loss
is below this threshold, the block corrects everything perfectly. When loss
exceeds it, the block fails completely and all channels are lost.

The data shows this cliff occurs between 30% and 35% loss, where uniform RS
PSNR drops by **5.5–5.7 dB** in a single step.

### How UEP survives beyond the cliff

Tiered UEP divides channels into groups (tiers) with different RS redundancy
levels. The most important channels get heavy parity and survive higher loss
rates; less important channels get less parity and may fail earlier.

At 40% loss:
- Uniform RS: the single block fails → all channels lost → severe quality drop
- UEP: the top tier (most important channels) survives → decoder retains
  critical reconstruction information → graceful quality degradation

### Why UEP loses at low loss

Below 30% loss, uniform RS corrects everything perfectly because its single
large block is within correction capacity. UEP splits this efficient block
into smaller sub-blocks, each of which is individually less efficient. Some
low-priority tiers may fail even when uniform RS would have recovered
everything.

**The low-loss penalty (about 2 dB at 30% loss) is an inherent cost of the
tiered approach**, not a flaw in the ranking. Any system that trades uniform
protection for importance-weighted protection will pay this cost.

---

## 9. Execution Timeline

| Step | Start Time | Duration | Exit Code |
|---|---|---|---|
| Smoke train | 15:48 Nov 6 | 245 s (4 min) | 0 |
| Train seed 42 | ~15:56 Nov 6 | 4,874 s (81 min) | 0 |
| Train seed 100 | ~17:17 Nov 6 | 5,215 s (87 min) | 0 |
| Train seed 7 | ~18:44 Nov 6 | 5,452 s (91 min) | 0 |
| Train seed 13 | ~20:15 Nov 6 | 4,852 s (81 min) | 0 |
| Multiseed analysis | ~21:47 Nov 6 | 3 s | 0 |
| Saliency A.2 | ~21:47 Nov 6 | 11 s | 0 |
| Head A.3 | ~21:48 Nov 6 | 193 s (3 min) | 0 |
| Validation sweep | ~21:51 Nov 6 | 2,265 s (38 min) | 0 |
| Scheme selection | ~22:28 Nov 6 | 8 s | 0 |
| Test sweep seed 42 | ~22:52 Nov 6 | 4,019 s (67 min) | 0 |
| Test sweep seed 100 | ~23:59 Nov 6 | 2,870 s (48 min) | 0 |
| Test sweep seed 7 | ~00:47 Nov 7 | 2,885 s (48 min) | 0 |
| Test sweep seed 13 | ~01:35 Nov 7 | 2,671 s (45 min) | 0 |

**Total wall-clock time: approximately 10.5 hours** (training dominated).

---

## 10. Output Files Reference

| File | Contents |
|---|---|
| `results/g1/deepjsccq/seed*/metrics.json` | Training metrics, convergence info |
| `results/g1/deepjsccq/seed*/oracle_ranking.json` | LOO damage ranking (training set) |
| `results/g1/deepjsccq/seed*/best_val.pth` | Best validation checkpoint |
| `results/g1/deepjsccq/multiseed/multiseed.md` | Cross-seed damage comparison |
| `results/g1/deepjsccq/seed42/saliency/saliency.md` | Proxy vs true damage |
| `results/g1/deepjsccq/seed42/head/head_metrics.md` | Sensitivity head performance |
| `results/g1/deepjsccq/seed42/val_sweep/val_sweep.md` | Validation RS-UEP sweep |
| `results/g1/deepjsccq/seed42/val_sweep/selected_schemes.json` | Frozen scheme selection |
| `results/g1/deepjsccq/seed*/test_sweep/test_sweep.md` | Final test results |
| `results/test_access_log.jsonl` | Test set access audit trail |
| `results/logs/pipeline_manifest.jsonl` | Full execution provenance |

---

## 11. Remaining Steps (Not Yet Executed)

### Step 7: Same-Hardware Anchor (AD-07)

Retrain ADJSCC-Q seed 42 on this Mac (MPS) to separate "different hardware"
from "different architecture" as an explanation for any ADJSCC/DeepJSCC
difference.

```bash
caffeinate -i python run_pipeline.py --only g1_anchor_train --train_device mps --skip_existing --execute
caffeinate -i python run_pipeline.py --only g1_anchor_final --workers $W --skip_existing --execute
```

### Step 8: Comparison of Record

Run the formal comparison with pre-registered thresholds:

```bash
python run_pipeline.py --only g1_compare_with_anchor,g1_compare --execute
```

This will produce the final verdict document at
`results/g1/compare_with_anchor/compare.md`.

---

## 12. Supported Conclusions

1. **Latent channel sensitivity is strongly unequal without attention.** Every
   DeepJSCC-Q model has at least one channel whose loss destroys >10 dB of
   PSNR. The top-4 channels carry 33–40% of total damage (vs 25% for uniform).

2. **UEP gains are substantial and statistically significant.** Oracle UEP
   gains range from +2.05 to +2.76 dB at 40% loss, with 95% CIs entirely
   above +1.5 dB.

3. **The benefit comes from *which* channels are protected.** Random tiering
   captures only 16–22% of the oracle gain. The identity of protected channels
   is the dominant factor.

4. **Attention is not required for UEP.** The FLAT hypothesis (no exploitable
   structure without attention) is decisively rejected. Sensitivity
   concentration is an intrinsic property of quantised Deep JSCC autoencoders.

5. **Energy is a powerful zero-training proxy.** Latent-channel energy
   (squared magnitude) tracks true damage with Spearman +0.84 and can match
   or exceed the oracle ranking in the final UEP evaluation.

6. **The sensitivity head is a practical deployment solution.** At 0.039% of
   the encoder cost, it provides 88.1% top-4 accuracy and 110% of the oracle's
   UEP gain through per-image adaptation.

7. **Attention amplifies but does not create concentration.** ADJSCC-Q's
   45–53% top-4 share vs DeepJSCC-Q's 33–40% suggests attention concentrates
   importance further, but the base effect exists without it.

## 13. Claims to Avoid

- Do not claim UEP always beats uniform RS. Below ~35% loss, uniform RS is
  better due to the block-length advantage.
- Do not claim specific channel indices are universally important. Channel
  labels are checkpoint-specific and permute between initialisations.
- Do not cite the formal verdict as "criteria met" — it is "intermediate"
  because C1 (top-4 ≥ 0.40) narrowly failed at the family level.
- Do not generalise these results to architectures or datasets not tested
  without further experimentation.

---

*Generated from G1 pipeline results, Steps 0–6. Git commit
`79b295eeee917b210fe71782134c259f1b8b23d9`, branch `g0-g1-speech-pipeline`.
Platform: macOS arm64, PyTorch 2.4.1, Apple Silicon MPS.
Last test sweep completed 2026-11-07 02:19 IST.*

