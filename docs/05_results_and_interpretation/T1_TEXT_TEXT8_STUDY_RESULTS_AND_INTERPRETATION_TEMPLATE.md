# Text Track (T1): All Results and Their Meaning

> **TEMPLATE: nothing has been run yet.** Every `—` is a number to fill in from the files named next
> to it. The structure mirrors the speech write-up (S1) so the two can be read side by side. How to
> produce the numbers: [`T1_TEXT_TEXT8_STUDY_RUNBOOK_MAC_MPS.md`](../03_runbooks/T1_TEXT_TEXT8_STUDY_RUNBOOK_MAC_MPS.md). Design decisions:
> [`ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md`](../02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md), AD-22 to AD-30.

**Units.** Text quality is **character accuracy (%)**. Damages and gains are **percentage points (pp)**
of character accuracy. In the shared result files they sit in fields called `psnr_db_*` /
`psnr_drop_db_by_channel`, and shared tables may label them "dB"; for text, read every such "dB" as pp
(AD-24). Only the **top-4 share** and the **random/oracle ratio** are unit-free and directly comparable
with images and speech.

## Executive Summary

> **Research question:** Does sensitivity-aware Unequal Error Protection (UEP) carry over to a
> **discrete** source (text), and is channel importance an intrinsic property of quantised Deep JSCC
> latents or an artefact of attention gates?
>
> **Answer:** — *(one or two sentences once Sections 1–2 are filled: e.g. "Concentration is intrinsic
> for text as well: top-4 share —% without attention vs —% with attention; the worst single channel
> costs — pp of character accuracy.")*

---

## 0. Experimental Setup

### Architectures

1. **`adjsccq_text` (attention-gated text JSCC):** character embedding (27 → 32), then the 1-D
   ADJSCC-Q backbone with an AF gate after every layer; logits over 27 symbols per position.
2. **`deepjsccq_text` (attention-free text JSCC):** identical backbone with all attention modules removed.

```text
deepjsccq_text = adjsccq_text − all attention gates
```

### Protocol & Geometry

| Property | Value | Meaning |
|---|---|---|
| Dataset | text8 (first 10^8 characters of English Wikipedia, `a-z` + space) | standard character-level benchmark |
| Splits | train [0, 90M), val [90M, 95M), test [95M, 100M) characters | the standard contiguous text8 split |
| Window | 2,048 characters, non-overlapping | 1 window = 1 transmission unit |
| Window count | train 40,000 (of 43,945); val 2,441; test 2,441 | deterministic uint8 cache |
| Latent geometry | 16 channels × 64 steps | **1,024 bytes per window** (identical to CIFAR 16×8×8 and speech 16×64) |
| Rate | 4 bits per character | below the 4.75 bits of a context-free 27-symbol code |
| Quantisation | 256 levels (byte-valued) | compatible with byte-level Reed-Solomon |
| Coding overhead | 50% RS parity budget | N = 2K baseline, as for images and speech |
| Loss | cross-entropy on per-character logits | the categorical counterpart of MSE (AD-25) |
| Training device | — | MPS / CUDA, same for all six models |
| Evaluation device | CPU (W = —) | deterministic multi-process RS evaluation |
| Convergence rule | `best_epoch` ≥ 50 (of 60) | as for images and speech |

### Verified Dataset Hashes (from `data/text8_windows/*.json`)

* **Source file `text8`:** SHA-256 `—`
* **Train:** 40,000 windows, SHA-256 `—`
* **Val:** 2,441 windows, SHA-256 `—`
* **Test:** 2,441 windows, SHA-256 `—`

### Pipeline Execution Status

```text
Step 2:  Unit tests (tests/test_text.py) and dry run            [PENDING]
Step 3:  Download text8 + build window caches                   [PENDING]
Step 4:  Smoke test + smoke evaluation                          [PENDING]
Step 5:  Full training of 6 text models                         [PENDING]
Step 6:  Validation analysis (multiseed, proxies, head, sweep)   [PENDING]
Step 7:  Test sweeps (5 channel seeds × 2,441 windows)           [PENDING AD-28 CONFIRMATION]
Step 8:  Word accuracy / bits-per-char / example passages        [PENDING]
Step 9:  Frozen comparison of record                             [PENDING]
```

---

## 1. Model Training Results (Step 5)

Source: `results/text/<arch>/seed*/metrics.json` (`best_epoch`, `clean_validation.*`, `clean_test.*`) and
`results/logs/pipeline_manifest.jsonl` (time, return code).

| Model | Seed | Training time | Best epoch | Val char acc | Val word acc | Val bits/char | Test char acc | Test word acc | Converged |
|---|---|---|---|---|---|---|---|---|---|
| `adjsccq_text` | 42 | — | — | — % | — % | — | — % | — % | — |
| `adjsccq_text` | 100 | — | — | — % | — % | — | — % | — % | — |
| `adjsccq_text` | 7 | — | — | — % | — % | — | — % | — % | — |
| **`adjsccq_text` mean** | — | — | — | — | — | — | — | — | —/3 |
| `deepjsccq_text` | 42 | — | — | — % | — % | — | — % | — % | — |
| `deepjsccq_text` | 100 | — | — | — % | — % | — | — % | — % | — |
| `deepjsccq_text` | 7 | — | — | — % | — % | — | — % | — % | — |
| **`deepjsccq_text` mean** | — | — | — | — | — | — | — | — | —/3 |

### How to read these numbers

* **Attention vs no attention on clean text:** the difference of the two family means of test character
  accuracy, in pp. In images and speech this gap was small (speech: 0.11 dB). A small gap here too means
  attention adds no capacity for text either.
* **Word vs character accuracy:** word accuracy is always lower (a word needs every letter right). A rough
  check: with independent errors at rate p, word accuracy ≈ (1 − p)^5.5. If word accuracy is much *higher*
  than that, the errors cluster in a few words (bursty), which is worth a sentence.
* **Bits per character:** below 4.75 means the decoder is more certain than a uniform guess; lower is
  better. It is the text analogue of the SI-SDR sanity check in speech.
* **Convergence:** any `best_epoch` < 50 is named here and excluded from the family means. Seed 13 is
  added for that architecture (runbook step 5).

---

## 2. Leave-One-Out (LOO) Sensitivity Analysis (Oracle Damage)

$$\text{damage}(c) = \text{clean char acc} - \text{char acc after zeroing channel } c \quad (\text{pp})$$

Source: `results/text/<arch>/seed*/oracle_ranking.json` (`psnr_drop_db_by_channel`, computed on all
40,000 training windows) and `results/text/<arch>/multiseed/multiseed.md`.

| Model | Seed | Top-4 channels | Worst channel (pp) | 4th channel (pp) | Least important (pp) | Top-4 sum | Total 16-channel | **Top-4 share** |
|---|---|---|---|---|---|---|---|---|
| `adjsccq_text` | 42 | — | — | — | — | — | — | **—%** |
| `adjsccq_text` | 100 | — | — | — | — | — | — | **—%** |
| `adjsccq_text` | 7 | — | — | — | — | — | — | **—%** |
| **`adjsccq_text` mean** | — | — | — | — | — | — | — | **—%** |
| `deepjsccq_text` | 42 | — | — | — | — | — | — | **—%** |
| `deepjsccq_text` | 100 | — | — | — | — | — | — | **—%** |
| `deepjsccq_text` | 7 | — | — | — | — | — | — | **—%** |
| **`deepjsccq_text` mean** | — | — | — | — | — | — | — | **—%** |

Top-4 share = top-4 sum / total. A perfectly even split gives 4/16 = 25%.

### Detailed channel drop profiles

* `adjsccq_text` seed 42: — *(the 16 damages, sorted, from `oracle_ranking.json`)*
* … one line per model …

### What to look for

1. **Is concentration intrinsic?** Compare the two family top-4 shares. Speech gave 43.0% (with attention)
   vs 46.6% (without). If `deepjsccq_text` is also ≥ 40%, the claim extends to discrete symbols.
2. **Two tiers?** In images and speech the 16 channels split into a critical core (1–4 channels) and a
   flat tail. Check whether the sorted profile has a clear knee, and at which rank.
3. **Index permutation across seeds:** the critical channel indices should differ between seeds (symmetry
   breaking by SGD), as in images and speech. Same indices across seeds would be surprising and worth checking.
4. **What do critical channels carry?** Optional but useful for the paper: zero the top channel on a few
   windows (`examples.md` style) and look at *which* characters break. Spaces (word boundaries) vs
   vowels vs rare letters is a text-specific story.

---

## 3. Cross-Modality Comparison

| Modality | Metric | With attention | Without attention | Reading |
|---|---|---|---|---|
| Images (2-D) | Top-4 share | 45–53% | 34–40% (mean 37.0%) | from G1 |
| Speech (1-D, continuous) | Top-4 share | 43.0% | 46.6% | from S1 |
| **Text (1-D, discrete)** | **Top-4 share** | **—%** | **—%** | — |
| Images | Worst channel | 8.7–17.1 dB | 10.5–11.7 dB | |
| Speech | Worst channel | 8.08 dB | 10.56 dB | |
| **Text** | **Worst channel** | **— pp** | **— pp** | pp, not dB: compare the *pattern* only |
| Images | Peak UEP gain | +2.8 to +3.1 dB | +2.78 dB | |
| Speech | Peak UEP gain | (pending S1 step 7) | (pending S1 step 7) | |
| **Text** | **Peak UEP gain** | **— pp** | **— pp** | from step 7 |

### Key takeaway

— *(Write once filled. If both text families clear 40%: "Unequal error sensitivity holds for images,
speech and text, with and without attention: it is a property of quantised autoencoder latents, not of
a modality or of attention.")*

---

## 4. Pre-Registered Decision Rule (AD-28)

| Criterion | Threshold | `adjsccq_text` | `deepjsccq_text` | Verdict |
|---|---|---|---|---|
| **C1** top-4 concentration | ≥ 0.40 | — | — | — |
| **FLAT** | top-4 < 0.30 and oracle within 0.5 pp of random | — | — | — |
| **C2** UEP gain at 40% loss | ≥ +2.0 pp, CI > 0 | step 7 | step 7 | — |
| **C3** random control | random < 0.5 × oracle | step 7 | step 7 | — |

C1 and FLAT (concentration part) can be filled right after step 5 from the oracle files. C2 and C3 need
the test sweeps (`results/text/compare/compare.md`).

---

## 5. Example passages (Step 8)

Paste one window from `results/text/<arch>/seed42/examples/examples.md` at 40% loss (original,
`uniform`, best tiered scheme @ oracle). Wrong characters are in UPPER CASE.

```text
—
```

---

## 6. Next steps

See [`T1_TEXT_TEXT8_STUDY_RUNBOOK_MAC_MPS.md`](../03_runbooks/T1_TEXT_TEXT8_STUDY_RUNBOOK_MAC_MPS.md) for the commands of every remaining step.
