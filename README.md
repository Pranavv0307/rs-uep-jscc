# rs-uep-jscc: sensitivity-aware unequal error protection for quantised Deep JSCC over packet-erasure channels

This repository holds a research project on **deep joint source-channel coding (Deep JSCC)** for images,
with **Reed-Solomon (RS) codes** placed over its quantised latent bytes and sent over a
**packet-erasure channel**. The central question is whether some latent channels matter far more than others.
If they do, protecting those channels more strongly (**unequal error protection, UEP**) may beat equal
protection at the same bandwidth. The project also asks how a transmitter can tell which channels matter.

* **Target venue:** IEEE GLOBECOM 2027 (main track), with an extended journal version (IEEE TCCN / TCOM / TWC) later.
* **Current state (2026-11-08):** the image study is complete for two model families (ADJSCC-Q and DeepJSCC-Q).
  The speech extension is coded but not run, and the manuscript is not started. Section 6 lists known issues
  that must be fixed before writing.

---

## Contents

1. [The original problem statement](#1-the-original-problem-statement)
2. [How the project evolved, and why](#2-how-the-project-evolved-and-why)
3. [The research question now](#3-the-research-question-now)
4. [System in one picture](#4-system-in-one-picture)
5. [Results so far](#5-results-so-far)
6. [Known issues to fix before the paper](#6-known-issues-to-fix-before-the-paper)
7. [What comes next](#7-what-comes-next)
8. [Repository structure](#8-repository-structure)
9. [Setup](#9-setup)
10. [How to run](#10-how-to-run)
11. [Scientific-integrity rules enforced by the code](#11-scientific-integrity-rules-enforced-by-the-code)
12. [Naming and organisation conventions](#12-naming-and-organisation-conventions)

---

## 1. The original problem statement

The project began with this brief (full text and a concept explainer:
[`docs/01_project_definition/ORIGINAL_PROBLEM_STATEMENT_AND_CONCEPT_EXPLAINER.md`](docs/01_project_definition/ORIGINAL_PROBLEM_STATEMENT_AND_CONCEPT_EXPLAINER.md)):

> **Importance-Aware Reed–Solomon Coding for Deep JSCC.** Investigate whether Reed–Solomon coding can be
> applied in an importance-aware manner over the latent representation of a Deep JSCC model (e.g., using
> ADJSCC channel-attention weights) and evaluate its effectiveness under packet-erasure channels.
>
> Key references: Bourtsoulatze, Kurka & Gündüz (2019), *Deep Joint Source-Channel Coding for Wireless Image
> Transmission* ([arXiv 1809.01733](https://arxiv.org/abs/1809.01733)); Xu et al. (2022), *Wireless Image
> Transmission Using Deep Source Channel Coding With Attention Modules* (ADJSCC).

The working hypothesis behind the brief:

1. Deep JSCC maps an image to a latent tensor of C channels. Some channels probably carry more of the image
   than others.
2. ADJSCC adds squeeze-and-excitation-style **attention gates** that scale each channel. These gates might
   already say which channels are important.
3. If so, the transmitter can sort channels by attention, put the important ones in an RS block with more
   parity and the rest in blocks with less, and survive packet losses better than giving every channel the
   same protection.

## 2. How the project evolved, and why

The question changed twice. Each change came from a result, not a plan.

| When (2026) | Stage | What was built or learned | Why the project moved on |
|---|---|---|---|
| 13 Aug to 6 Sep | Weeks 1-5: reproduce Deep JSCC | DeepJSCC (Bourtsoulatze et al.) and ADJSCC (Xu et al.) re-implemented over an **AWGN** channel, with overfit sanity checks and SNR sweeps | RS codes need discrete symbols; an analogue AWGN latent cannot be RS-coded. |
| 14 to 18 Sep | Phase 3: quantise | **ADJSCC-Q**: a learnable 256-level scalar quantiser after the encoder, so **one latent value = one byte = one GF(256) RS symbol**. Trained on an identity channel (no noise) | Packet loss acts on bytes, not on analogue noise. |
| 15 to 30 Sep | Phases 3-5: the RS pipeline | Byte packetiser, GF(256) systematic RS erasure codec, packet-erasure channel, tiered schemes at a fixed 50% overhead, attention-sorted channel order | **First surprise:** tiers sorted by attention barely helped. A **leave-one-out (LOO) oracle** was added: zero one channel, measure the PSNR drop. The damage was very unequal, but attention did not track it. |
| 4 to 6 Oct | Train/val/test protocol | 40k/10k/10k CIFAR-10 split; checkpoint and scheme selection on validation only | Earlier models trained on all 50k images and selected on test. |
| 17 Oct | **Phase 0 audit** + roadmap Phases 1-5 | Integrity fixes (one shared loss mask for all methods, unsorted `uniform`, side-information accounting, seeding, provenance, seed-level statistics). Code for multi-seed analysis, cheap importance proxies, a sensitivity head, erasure-aware decoders, Kodak, bandwidth sweep, bursts, digital baselines | The pre-audit numbers mixed runs and gave some methods different losses. Everything from before is archived. |
| 17 to 19 Oct | Full runs (Kaggle T4/CPU) | 6 ADJSCC-Q seeds, all tracks, test split used once per table | Training recipe v1 diverged on 2 of 8 runs. Recipes v2 (cosine lr + clipping) and v3 (lower lr) did not fix it and were rejected. |
| 6 Nov | Literature review | About 120 works across six literatures (see `docs/06_literature_review/`) | **Second surprise:** unequal RS over packets is 30 years old (PET 1996, ULP 2000) and Grad-FEC (ICIP 2023) already does importance-ranked RS on learned features. The novelty is in the **measurements**, not the coding machinery. |
| 6 to 7 Nov | **G1 ablation** (DeepJSCC-Q, Mac MPS) | ADJSCC-Q with all 10 attention modules removed, same protocol, 4 seeds | Tests whether the channel imbalance is an attention artefact. It is not: the imbalance and the UEP gain remain without attention. |
| 6 Nov | S1 speech study | 1-D models on LibriSpeech, same 1024-byte latent | Code only; not run. |

**In short:** the project started as "use attention to decide RS protection". It became "attention does not
tell you what matters. Here is what does (leave-one-out damage, latent energy, a 2,048-MAC head), the imbalance
exists without attention, and here is when sensitivity-aware RS actually pays off."

## 3. The research question now

> **Does a quantised Deep JSCC latent have strongly unequal per-channel erasure sensitivity? Can a transmitter
> estimate it cheaply? When does protecting the sensitive channels with stronger Reed-Solomon codes beat
> equal protection at the same total overhead?**

Sub-questions, each answered by one part of the code:

| Question | Where it is answered |
|---|---|
| Is per-channel damage unequal, and is it stable across training seeds? | `experiments/multiseed_analysis.py` |
| Do ADJSCC's attention gates predict damage? | `experiments/attention_vs_damage.py`, `experiments/feature_saliency.py` |
| Can cheap statistics (energy, L1, Taylor) or a tiny learned head predict damage? | `experiments/feature_saliency.py`, `experiments/train_sensitivity_head.py` |
| Does one ranking transfer across datasets? | `experiments/transfer_sensitivity.py` |
| How much PSNR does sensitivity-aware tiered RS gain over uniform RS, and where? | `experiments/eval_rs_uep.py`, `experiments/select_scheme.py` |
| Does training the decoder with erasures remove the need for UEP? | `experiments/train_erasure_robust.py` |
| Bursty losses, high resolution, bandwidth ratio, digital codecs? | `eval_burst.py`, `eval_kodak.py`, the k/n sweep, `eval_digital_baseline.py` |
| Is the effect specific to ADJSCC's attention? (G1) | `models/deepjsccq.py`, `experiments/compare_architectures.py` |
| Does it carry over to speech? (S1, not run) | `models/speech_jscc.py`, `experiments/train_speech_jscc.py` |

## 4. System in one picture

```text
image (3x32x32)
  -> CNN encoder (ADJSCC-Q: with attention gates | DeepJSCC-Q: without)
  -> power normalisation -> 16 channels x 8x8 = 1024 latent values
  -> learnable 256-level scalar quantiser -> 1024 bytes (one GF(256) symbol each)
  -> rank channels by importance (attention | static LOO ranking | energy | head | random)
  -> split into tiers, RS(n,k) per tier, 16-byte packets, 96 packets total (50% overhead)
       uniform:              one RS(96,64) block over all 16 channels
       three_tier_25_50_25:  RS(32,16) top 4 ch | RS(48,32) next 8 ch | no parity, last 4 ch
  -> packet-erasure channel (i.i.d. or Gilbert-Elliott bursts), one shared loss mask for every method
  -> RS erasure decoding; unrecovered symbols filled with 0.0 in latent space (optional learned inpainter)
  -> CNN decoder -> reconstructed image -> PSNR / SSIM / MS-SSIM
```

## 5. Results so far

All numbers below are on the CIFAR-10 **test split** (10,000 images x 5 channel seeds) unless stated
otherwise. Gains are paired differences to uniform RS under the same packet losses. The ±0.03 to 0.07 dB
confidence intervals cover **channel randomness only**, not image sampling or model seed (see Section 6).
The tier allocation was chosen on validation (`three_tier_25_50_25` for every ranking source in the
high-loss region).

### 5.1 ADJSCC-Q, primary model (seed 42, recipe v1, Kaggle T4; clean test PSNR 30.88 dB)

| Packet loss | 0.1 | 0.2 | 0.3 | 0.35 | 0.4 | 0.45 | 0.5 |
|---|---|---|---|---|---|---|---|
| `three_tier_25_50_25`, static LOO ranking | -0.61 | -1.19 | -0.95 | +1.31 | +2.86 | +2.69 | +1.89 |
| `three_tier_25_50_25`, sensitivity head | -0.58 | -1.15 | -0.87 | +1.42 | +3.00 | +2.83 | +2.00 |
| sensitivity head, scheme chosen per loss rate on validation | 0.00 | -0.12 | -0.87 | +1.42 | +3.00 | +2.95 | +2.10 |

* **Break-even about 32% loss.** Below it, uniform RS is better. With a per-rate choice of scheme the penalty
  is negligible at 10-20% loss (`two_tier_50_50`) and -0.87 dB at 30%. The penalty has two causes: the
  unprotected bottom tier, and shorter RS blocks than the single RS(96,64).
* **Which channels are protected matters.** Mean gain over loss >= 0.35: static LOO ranking +2.19 dB,
  energy +2.24, head +2.31, random order +0.50, attention +0.31 dB.
* **Attention gates carry no ranking signal, and are worse than a random order.** On 2,000 validation images,
  within-image Spearman with true damage is -0.015, and attention's top choice was never the most damaging
  channel.
  On seeds 13 and 21, attention-ranked UEP is below uniform RS at high loss.
* **One ranking per checkpoint explains most of the structure.** The static training ranking reaches
  within-image Spearman +0.89 and transfers to CIFAR-100 (Spearman +0.997).
* **The head is negligible in cost.** It uses 2,048 MACs (0.039% of the encoder), with top-4 overlap 0.92 on
  validation (static table 0.88).

### 5.2 Across model seeds (ADJSCC-Q, 5 seeds that trained cleanly)

* Static-ranking gain at 40% loss: **+2.35 to +3.11 dB**. Sorted damage profiles agree across seeds
  (Pearson 0.92), while channel indices permute.
* Energy vs static ranking is **seed-dependent**: from +0.30 dB to **-0.92 dB** (seed 21). The head and the
  random control were run on seed 42 only.
* Two recipe-v1 runs diverged (seed 2024; the first k/n = 1/3 model, seed 42) and are excluded and flagged
  ([`published_results/analysis/convergence_ledger.json`](published_results/analysis/convergence_ledger.json)).

### 5.3 G1: DeepJSCC-Q, attention removed (4 seeds, Mac MPS)

Source: [`docs/05_results_and_interpretation/G1_DEEPJSCCQ_ATTENTION_ABLATION_RESULTS_AND_INTERPRETATION.md`](docs/05_results_and_interpretation/G1_DEEPJSCCQ_ATTENTION_ABLATION_RESULTS_AND_INTERPRETATION.md).
The raw G1 files are not yet in `published_results/`.

* Clean test PSNR 31.5 to 32.4 dB, and all 4 runs trained cleanly.
* Every seed still has a channel worth more than 10 dB; the top-4 channels carry 33-40% of the damage.
* Static-ranking gain at 40% loss: **+2.05 to +2.76 dB**. Random order captures only 16-22% of it.
* The penalty below break-even is larger (about -2 dB at 30%), and break-even is about 35%.
* Pre-registered verdict: **"intermediate"** (C2 and C3 met, C1 concentration narrowly missed).
* The same-hardware ADJSCC-Q anchor (G1 Step 7) is not yet run, so ADJSCC-vs-DeepJSCC differences may still
  be hardware effects.

### 5.4 Other tracks (ADJSCC-Q seed 42)

* **Erasure-aware decoders (Phase 2):** the fine-tuned decoders shrink the worst low-loss penalty
  (-0.87 -> -0.64 / -0.41 dB) but also the peak gain (+3.00 -> +1.17 / +1.22 dB). They broke the clean-PSNR
  budget (-1.37 / -1.38 / -0.59 dB against a target of < 0.3). In absolute terms, the inpainter model with
  *plain uniform RS* reaches 25.09 dB at 40% loss, above the base model's best UEP (23.72 dB).
* **Digital baselines (JPEG / JPEG2000 / WebP + RS(96,64), same 1,536 bytes and losses):** better when clean
  (32.9 to 38.1 dB vs 30.88). From 35% loss, even uniform-RS Deep JSCC (24.40 dB) beats them (20.7 to 23.0 dB).
* **Bursts (Gilbert-Elliott):** interleaving tiers across the packet timeline keeps most of the gain
  (+0.80 to +2.45 dB at mean Bad-state length >= 4). Sending tiers in order loses it. The realised mean loss
  run is at most about 4.5 packets.
* **Kodak 768x512 with the CIFAR-trained model:** clean 24.3 dB, peak gain +1.46 dB at 45% loss. Only
  relative behaviour is meaningful. The apparent "no low-loss penalty" is a domain-shift artefact (zeroing
  the lowest-ranked channels slightly *improves* Kodak images).
* **Bandwidth ratio k/n (1/16, 1/12, 1/6, 1/3):** gains at >= 40% loss at every ratio. At 35% loss, k/n = 1/12
  is slightly negative. The k/n = 1/3 point uses a seed-7 model, the others seed 42.

Auto-generated findings: [`published_results/gpu/report/INSIGHTS.md`](published_results/gpu/report/INSIGHTS.md).
Headline table: [`published_results/gpu/report/headline.md`](published_results/gpu/report/headline.md).
Interpretation: [`docs/05_results_and_interpretation/ADJSCCQ_MAIN_STUDY_RESULTS_AND_INTERPRETATION.md`](docs/05_results_and_interpretation/ADJSCCQ_MAIN_STUDY_RESULTS_AND_INTERPRETATION.md).

## 6. Known issues to fix before the paper

Found in a code-and-results audit on 2026-11-08. None of these changes the core findings, but each would draw
a reviewer objection or misstates a number.

1. **"Oracle" is a static ranking.** The source named `oracle` is the training-set mean LOO ranking, so
   "head / energy beat the oracle" means "beat a static table". Rename it, and add the per-image LOO ranking
   to the test table.
2. **Test-set selection in the generated insights.** `experiments/reporting.py` and
   `experiments/make_report.py` take the best method per loss rate on the test split for break-even, peak and
   model-comparison numbers. Use the validation-selected scheme instead.
3. **Hard-coded explanation text.** The insight string calls the low-loss penalty "a block-length effect".
   It is mostly the unprotected bottom tier.
4. **Confidence intervals cover channel seeds only.** Add a bootstrap over images and model-seed spread.
   Report energy-vs-oracle and attention results for every seed.
5. **Non-standard metrics.** SSIM uses zero-padded windows, so it is not comparable to standard SSIM.
   MS-SSIM on 32x32 uses 2 scales. In `eval_digital_baseline.py`, `failed_symbol_fraction` actually stores
   the block-failure rate.
6. **The G1 report misdescribes the head.** It says the head was trained on train + validation, and gives its
   architecture as `Linear(C, C)`. The code trains on the 40k training split only, with a 16 -> 64 -> 16 MLP.
   Its RS explanation in §8 is also wrong: erasure decoding recovers n - k losses, and systematic packets are
   kept on failure.
7. **Provenance.** The headline table, Kodak, head and validation sweeps were produced from a dirty working
   tree. `requirements-local.txt` pins torch 2.4.1, which no published result used.
8. **Anonymity.** The public GitHub repository and its fork are searchable under the working title. Archived
   W&B logs contain a username, and published logs contain local paths and Kaggle usernames.

## 7. What comes next

In priority order:

1. Fix Section 6, items 1-6 (documentation and reporting only, no compute).
2. **G1 Step 7:** retrain ADJSCC-Q seed 42 on the same Mac, to separate hardware from architecture effects.
3. **Riskiest baseline:** Deep JSCC with no RS, using all 1,536 bytes and trained with erasures.
4. **Coding baselines (no training):** a PET shared-packet twin of each tiered scheme, an optimal parity
   search, a per-loss-rate scheme selector, a design-loss vs actual-loss mismatch study, and BPG + RS.
5. Multi-seed statistics for every importance source.
6. Write the GLOBECOM paper: DeepJSCC-Q as the primary model, ADJSCC-Q as the attention case study.
7. Later, for the journal version: a high-resolution model (DIV2K -> Kodak), a Transformer backbone, and an
   analytical allocation model. The speech study (S1) becomes a separate paper.

## 8. Repository structure

```text
rs-uep-jscc/
├── README.md                     this file
├── run_pipeline.py               single entry point for every phase (prints commands; runs with --execute)
├── requirements-local.txt        pinned packages for local runs (see Section 6, item 7)
├── .gitignore
│
├── coding/                       channel-coding layer (no neural networks)
│   ├── reed_solomon.py           GF(256) systematic RS erasure codec, column-wise block encode/decode
│   ├── rs_pipeline.py            CodingScheme, tiered schemes, encode_blocks/decode_blocks, side-info rule, 0.0 fallback
│   ├── scaling.py                scale the 16-channel schemes to other channel counts / packet sizes; split blocks > 255
│   ├── packetizer.py             PacketizedLatent container and plain (non-RS) packetisation
│   ├── erasure_channel.py        Bernoulli and Gilbert-Elliott packet loss, shared master loss mask
│   ├── interleave.py             sequential vs round-robin packet order on the timeline (burst study)
│   └── importance.py             channel order -> symbol permutation and its inverse
│
├── models/                       neural networks
│   ├── deepjscc.py               DeepJSCC backbone (Bourtsoulatze et al. 2019) and power normalisation
│   ├── adjscc.py                 ADJSCC backbone: DeepJSCC + SNR-conditioned attention (AF) modules
│   ├── Attention.py              AF module (squeeze-and-excitation-style channel gate)
│   ├── quantizer.py              learnable 256-level soft-to-hard scalar quantiser (1 value = 1 byte)
│   ├── adjsccq.py                ADJSCC-Q encoder (+ quantiser) and the SensitivityHead (Track A.3)
│   ├── deepjsccq.py              DeepJSCC-Q: ADJSCC-Q without attention (G1 ablation)
│   ├── inpainting.py             masked-latent transformer that fills unrecovered symbols (Track B.2)
│   ├── speech_jscc.py            1-D ADJSCC-Q / DeepJSCC-Q for speech (S1)
│   ├── registry.py               architecture name -> (encoder, decoder) builder
│   └── channel.py                AWGN layer (used by the backbone unit tests; not by the erasure study)
│
├── data/                         datasets
│   ├── splits.py                 the fixed 40k/10k/10k CIFAR-10 split
│   ├── datasets.py               CIFAR-10/100, STL-10, Kodak, folder crops; in-memory evaluation tensors
│   ├── speech.py                 LibriSpeech download, segmentation and cache (S1)
│   └── cifar10.py                simple CIFAR-10 loaders (used by the visualiser and clean-evaluation tools)
│
├── experiments/                  every training, evaluation, analysis and reporting entry point
│   ├── train_adjsccq_train_split.py   train ADJSCC-Q / DeepJSCC-Q on 40k/10k, save best_val.pth + LOO ranking
│   ├── train_erasure_robust.py        Track B: fine-tune decoder (and optional inpainter) under erasures
│   ├── train_sensitivity_head.py      Track A.3: LOO labels + the 2,048-MAC sensitivity head
│   ├── train_highres.py               Track C.1 fallback: fine-tune on high-resolution crops
│   ├── train_speech_jscc.py           S1 speech training
│   ├── eval_rs_uep.py                 main packet-erasure sweep (all schemes x importance sources)
│   ├── eval_rs_uep_oracle.py          thin wrapper of eval_rs_uep with the oracle defaults
│   ├── eval_burst.py                  Track D.1: Gilbert-Elliott bursts, sequential vs interleaved
│   ├── eval_digital_baseline.py       Track D.2: JPEG / JPEG2000 / WebP (+BPG) + RS on the same losses
│   ├── eval_kodak.py                  Track C.1: Kodak 768x512
│   ├── eval_speech_perceptual.py      S1: PESQ / STOI / WAV examples
│   ├── eval_adjsccq_clean.py          clean (no-loss) reference evaluation
│   ├── attention_vs_damage.py         attention gates vs true LOO damage (Figure 2)
│   ├── feature_saliency.py            Track A.2: energy / L1 / variance / Taylor proxies vs LOO damage
│   ├── transfer_sensitivity.py        Track A.1: does the ranking transfer to another dataset?
│   ├── multiseed_analysis.py          damage profiles across training seeds
│   ├── compare_architectures.py       G1: ADJSCC-Q vs DeepJSCC-Q with the pre-registered decision rule
│   ├── select_scheme.py               choose tier allocations on the VALIDATION sweep only
│   ├── make_report.py                 paper figures, headline.md, INSIGHTS.md
│   ├── visualize_reconstructions.py   side-by-side reconstruction grids
│   ├── importance_sources.py          every channel-ranking source behind one interface
│   ├── pipeline_core.py               shared transmit/accumulate/parallel-runner machinery
│   ├── sensitivity.py                 LOO damage, ranking loss, MAC counter
│   ├── model_io.py                    load checkpoints into a ModelBundle; reconstruct()
│   ├── metrics.py                     PSNR, SSIM, MS-SSIM, LPIPS, speech SNR / SI-SDR
│   ├── stats_utils.py                 ranks, correlations, bootstrap, paired differences
│   ├── reproducibility.py             seeding, provenance, seed statistics, test-access log
│   ├── reporting.py                   tables, standard figures, auto-generated insights
│   ├── plotting.py                    colour-blind-safe figure helpers
│   ├── download_data.py               CIFAR-10/100, STL-10, Kodak downloads
│   ├── prepare_speech_data.py         LibriSpeech download + segment cache (S1)
│   ├── preflight.py                   read-only environment / credentials / data checks
│   ├── benchmark_training.py          epoch-time benchmark used to size Kaggle jobs
│   ├── ingest_kaggle.py               pull a finished Kaggle kernel's outputs into results/
│   ├── publish_results.py             copy small result files from results/ to published_results/ + manifest
│   └── wandb_utils.py                 optional Weights & Biases logging
│
├── configs/
│   ├── adjsccq_identity_train.yaml    ADJSCC-Q recipe v1 (every published ADJSCC-Q model)
│   ├── deepjsccq_identity_train.yaml  DeepJSCC-Q (G1), identical except the architecture
│   ├── speech_adjsccq1d.yaml          S1, 1-D with attention
│   └── speech_deepjsccq1d.yaml        S1, 1-D without attention
│
├── tests/                        pytest suite (tiny random models; no datasets needed except test_cifar10.py)
│
├── kaggle/
│   ├── kernel_train/run.py       template of every Kaggle job (clones this repo, runs one JOB)
│   ├── make_kernels.py           generate kaggle/kernels/<slug>/ from the template
│   ├── kg.py                     Kaggle CLI wrapper using repo-local credentials
│   └── kernels/                  the generated jobs that produced published results (training, eval, Kodak, burst, digital)
│
├── published_results/            small, tracked copies of results/ (mirror of results/; see below)
│
├── docs/                         ALL documentation (see the guide below)
│
└── archive/                      superseded code, docs, results and logs; nothing here feeds the paper
                                  (index: docs/ARCHIVE_INDEX.md)
```

### 8.1 Documentation guide (`docs/`)

Folders are numbered in reading order.

| File | What it is |
|---|---|
| [`01_project_definition/ORIGINAL_PROBLEM_STATEMENT_AND_CONCEPT_EXPLAINER.md`](docs/01_project_definition/ORIGINAL_PROBLEM_STATEMENT_AND_CONCEPT_EXPLAINER.md) | The original brief plus a long concept explainer of Deep JSCC, latents, RS and UEP |
| [`02_research_plan_and_design_decisions/RESEARCH_ROADMAP_GLOBECOM_ICC_TO_TWC_TCOM.md`](docs/02_research_plan_and_design_decisions/RESEARCH_ROADMAP_GLOBECOM_ICC_TO_TWC_TCOM.md) | The research roadmap: Tracks A-E, integrity rules, success criteria |
| [`02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md`](docs/02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md) | Every design decision for G0/G1 and S1, with the pre-registered decision thresholds |
| [`03_runbooks/MAIN_ADJSCCQ_PIPELINE_RUNBOOK_PHASES_VERIFY_TO_FINAL.md`](docs/03_runbooks/MAIN_ADJSCCQ_PIPELINE_RUNBOOK_PHASES_VERIFY_TO_FINAL.md) | Commands, outputs and interpretation of every main-pipeline phase |
| [`03_runbooks/G1_DEEPJSCCQ_ATTENTION_ABLATION_RUNBOOK_MAC_MPS.md`](docs/03_runbooks/G1_DEEPJSCCQ_ATTENTION_ABLATION_RUNBOOK_MAC_MPS.md) | Step-by-step G1 run on an Apple-silicon Mac |
| [`03_runbooks/S1_SPEECH_LIBRISPEECH_STUDY_RUNBOOK_MAC_MPS.md`](docs/03_runbooks/S1_SPEECH_LIBRISPEECH_STUDY_RUNBOOK_MAC_MPS.md) | Step-by-step speech study |
| [`04_research_progress_log/RESEARCH_PROGRESS_LOG_PHASE0_TO_G1_AND_S1.md`](docs/04_research_progress_log/RESEARCH_PROGRESS_LOG_PHASE0_TO_G1_AND_S1.md) | Dated record of what was run, problems, fixes and decisions |
| [`05_results_and_interpretation/ADJSCCQ_MAIN_STUDY_RESULTS_AND_INTERPRETATION.md`](docs/05_results_and_interpretation/ADJSCCQ_MAIN_STUDY_RESULTS_AND_INTERPRETATION.md) | All ADJSCC-Q results and what they mean (see Section 6 for known corrections) |
| [`05_results_and_interpretation/G1_DEEPJSCCQ_ATTENTION_ABLATION_RESULTS_AND_INTERPRETATION.md`](docs/05_results_and_interpretation/G1_DEEPJSCCQ_ATTENTION_ABLATION_RESULTS_AND_INTERPRETATION.md) | G1 results and the pre-registered verdict |
| [`06_literature_review/LITERATURE_REVIEW_NOVELTY_ASSESSMENT_MUST_CITE_AND_EXPECTED_BASELINES.md`](docs/06_literature_review/LITERATURE_REVIEW_NOVELTY_ASSESSMENT_MUST_CITE_AND_EXPECTED_BASELINES.md) | Novelty assessment, must-cite list, baselines reviewers will expect |
| [`06_literature_review/research_notes_by_topic/`](docs/06_literature_review/research_notes_by_topic/) | Six topic notes behind the review (UEP for JSCC, packet-erasure JSCC, attention as importance, classic UEP, speech, ordered latents) |
| [`06_literature_review/reference_papers/`](docs/06_literature_review/reference_papers/) | PDF of Bourtsoulatze, Kurka & Gündüz (2019) |
| [`ARCHIVE_INDEX.md`](docs/ARCHIVE_INDEX.md) | What is in `archive/`, its original path, and why it is not used |

### 8.2 Result folders (`published_results/`)

`published_results/` is a **mirror of the local, git-ignored `results/` tree**, written by
`experiments/publish_results.py` with a SHA-256 manifest (`MANIFEST.json`). Its layout is kept identical to
`results/` on purpose. Every result file records the `results/...` paths and command line it was produced
with, and `run_pipeline.py` reads the published ADJSCC-Q family from these paths.

| Folder | Contents | Used in the paper? |
|---|---|---|
| `gpu/models/seed{42,100,7,13,21,2024}/` | metrics + LOO ranking of the 6 ADJSCC-Q models (seed 2024 diverged) | yes |
| `gpu/final/` | final test sweep (seed 42), `burst/`, `digital/` and figures | yes |
| `gpu/multiseed/seed*/` | test sweeps of the other seeds | yes |
| `gpu/phase1/` | attention vs damage, transfer, proxies, head, validation sweeps and scheme selection | yes |
| `gpu/phase2/robust_{channel,packet,inpaint}/` | erasure-aware decoders | yes |
| `gpu/phase3/` | k/n sweep (`kn_1_3_FAILED_seed42` is the diverged run) and Kodak | yes |
| `gpu/report/` | `headline.md`, `INSIGHTS.md`, `report_stats.json` | yes (after Section 6 fixes) |
| `gpu/logs/` | per-step logs and the pipeline manifest of the GPU-family runs | provenance |
| `analysis/` | convergence ledger and the recipe-v1 training-stability figure | yes |
| `local_models/`, `phase1/`, `logs/` (top level) | **CPU pilot** (seed 42 trained on a local CPU, 30.57 dB) and its smoke/pilot logs | independent replicate only |
| `phase3/kn_1_12`, `phase3/kn_1_16` (top level) | early k/n models trained before the GPU family | no (superseded by `gpu/phase3/`) |
| `test_access_log.jsonl` | every evaluation on the official test split | audit trail |

G1 outputs live in the local `results/g1/` and are not yet published.

## 9. Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install torch torchvision numpy pillow matplotlib pyyaml tqdm pytest wandb
python -m pip install lpips                              # optional (LPIPS on Kodak)
python -m experiments.download_data --cifar10 --cifar100 --kodak
python -m experiments.preflight                          # read-only check of data, credentials, checkpoints
```

* **Weights & Biases (optional):** run `wandb login` once, or put `WANDB_API_KEY=...` in the git-ignored `.env`.
  Add `--wandb` to any script.
* **Kaggle (optional, GPU/CPU jobs):** Kaggle -> Settings -> API -> *Create New Token*, then save it as
  `.kaggle/kaggle.json` inside this folder (git-ignored). Jobs are generated with
  `python kaggle/make_kernels.py <job>`, pushed with `python kaggle/kg.py kernels push -p kaggle/kernels/<slug>`,
  and pulled back with `python -m experiments.ingest_kaggle --kernel <slug> --dest results/gpu`.
* **Checkpoints:** weights (`*.pth`) are git-ignored. The pipeline expects the primary seed-42 model at
  `results/gpu/models/seed42/best_val.pth` (override with `--seed42_dir`).

## 10. How to run

Everything is driven by `run_pipeline.py`. It **prints** the commands by default and runs them only with
`--execute`:

```bash
python run_pipeline.py --list                                            # every step, grouped by phase
python run_pipeline.py --phase verify --execute                          # unit tests + smoke test first
python run_pipeline.py --phase phase1 --execute --skip_existing --workers 8 --wandb
python run_pipeline.py --phase g1 --train_device mps --execute           # G1 extension (not in --phase all)
```

Phases: `verify`, `phase1` (multi-seed + Track A), `phase2` (Track B), `phase3` (Track C), `phase4` (Track D),
`final` (test tables + report), plus the extensions `g1` and `speech`. Per-phase details and how to read the
outputs: [`docs/03_runbooks/`](docs/03_runbooks/).

Unit tests: `python -m pytest tests -q` (179 passed on the last recorded run, 2026-10-18, before the G1 and
S1 tests were added).

## 11. Scientific-integrity rules enforced by the code

1. The official test split is used for final tables only; every access is appended to
   `results/test_access_log.jsonl` (published copy: `published_results/test_access_log.jsonl`).
2. Checkpoints and tier allocations are chosen on the validation split (`experiments/select_scheme.py`
   refuses a test sweep).
3. All methods face the **same** packet losses: one master mask per batch, seed and loss rate.
4. Failed RS blocks are filled with 0.0 in latent space; no ground truth reaches the receiver.
5. Overhead is reported with and without side information (0 B for unsorted schemes, one byte per channel
   for ranked ones).
6. Error bars are across channel seeds with paired differences; model-seed variance is reported separately.
7. Deterministic seeding everywhere. Every result file carries the git commit, dirty flag, versions, device,
   checkpoint SHA-256 and command line.

## 12. Naming and organisation conventions

* **Code modules keep their names** (`experiments/*.py`, `models/*.py`, `coding/*.py`, `data/*.py`).
  Their names are already descriptive, and they are part of the recorded provenance: every published result
  file and every Kaggle job calls them as `python -m experiments.<name>`. Renaming them would break the
  ability to re-run a recorded command.
* **Documentation** lives only in `docs/`, in numbered folders and with descriptive upper-case file names.
  The only Markdown file at the root is this README.
* **Archived files** carry descriptive lower-case names with their stage (week / phase / date) and why they
  were superseded. Their original paths are listed in `docs/ARCHIVE_INDEX.md`, and `git log --follow` shows
  their history.
* **Local, git-ignored folders:** `data/` downloads, `results/`, `checkpoints/`, `wandb/`, `.env`, `.kaggle/`.
