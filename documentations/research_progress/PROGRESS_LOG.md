# Research Progress Log

Living record of what was done after each phase of
[`TOP_TIER_RESEARCH_ROADMAP.md`](../TOP_TIER_RESEARCH_ROADMAP.md), what was
decided and why, what has and has not been verified, and what comes next.
**A new section is appended after every phase.** Newest phase last.
How to run everything: [`RUNBOOK.md`](RUNBOOK.md).

---

## Standing rules (apply to every phase and every command given)

1. **Nothing is executed without the owner's approval.** Code is written first;
   tests, smoke runs and full runs are proposed as exact commands and run only
   after a yes.
2. **Full-protocol commands** use the whole split (no `--max_images`), channel seeds
   `0,1,2,3,4`, an explicit `--checkpoint`, an explicit frozen ranking where relevant,
   and an explicit `--output_dir`. Anything smaller is labelled **smoke test** and its
   numbers are never quoted.
3. **Test set is touched once** for final tables (Roadmap Rule 1). Tuning (checkpoint
   choice, tier allocation, head selection) uses the validation split. Every test access
   is appended to `results/test_access_log.jsonl`.
4. **Error bars are over channel seeds**, not over images; gains are paired per seed.
   Model-seed variance (retraining) is reported separately and never mixed in.
5. **Side-info overhead:** `uniform` and `no_rs` do not sort channels, so 0 B. Every method
   that sorts by importance (attention, oracle, predicted, random control, ...) is listed with
   one byte per channel (16 B/image at 16 channels) as a separate overhead line. Coding-only and
   total overhead are always two columns. (A static table could cost 0 B once Step A.1 shows it
   transfers; deferred by decision.)
6. **Determinism:** every entry point calls `seed_everything`; results are identical for any `--workers`.

CIFAR-10 has 60,000 images: 50,000 official training + 10,000 official test.
The 40k/10k/10k protocol splits the 50k training set into 40k train + 10k
validation and keeps the 10k official test set separate.

---

## Phase 0: Scientific-hygiene clean-up

**Status:** code written; unit tests **passed** (17 new tests; whole suite 103 passed, run before
the Phase 1-5 code below was added); the smoke evaluation was **not run**.
**Date:** 2026-10-17.

### Goal
Fix the integrity gaps found when auditing the repo against the roadmap
(Rules 1-5, section 10) *before* any new experiment is built on top of them.

### What was done

| # | Problem found | Fix |
|---|---|---|
| 1 | `no_rs` (64 packets) drew its **own** loss mask, a second draw after the 96-packet one. | One **master mask** per (batch, seed, rate). Every method uses it; smaller schemes read its first columns (`sample_master_mask` / `slice_master_mask`). |
| 2 | Side-info bytes inconsistent. | Single rule in `coding/rs_pipeline.py` (`UNSORTED_SCHEMES`, `side_info_bytes`): `uniform`, `no_rs` = 0, everything that sorts = one byte per channel. |
| 3 | `uniform` / `no_rs` were being sorted by channel order. | Both run with `order=None` everywhere. |
| 4 | Seeding incomplete. | `experiments/reproducibility.py::seed_everything`: python / numpy / torch / CUDA seeds, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `use_deterministic_algorithms(warn_only)`, `CUBLAS_WORKSPACE_CONFIG`, `PYTHONHASHSEED`; `seed_worker` for DataLoader workers. Used by every entry point. |
| 5 | No provenance. | `build_provenance()`: git commit/branch/dirty, UTC time, python/torch/numpy/CUDA/cuDNN versions, platform, device, dataset + split, seeds, determinism flags, command line, args, checkpoint + ranking + head SHA-256. |
| 6 | Std was across images. | `aggregate_over_seeds()`: mean, sample std (n-1), 95% t-interval across channel seeds; per-image std kept as a secondary column. |
| 7 | Two near-duplicate evaluators. | One shared evaluator; `eval_rs_uep_oracle.py` is a thin wrapper. |
| 8 | Oracle JSON key mismatch (would have crashed the roadmap's headline command). | `load_ranking_file()` accepts both key names, validates a permutation, hashes the file, warns if `source_split != "train"`. |
| 9 | Train-split sweep defaulted to 3 seeds. | `configs/adjsccq_identity_train.yaml` gained an `rs_uep` block (seeds 0-4). |
| 10 | Visualizer used a hard-coded, test-derived ranking. | `--oracle_ranking_file`, warning when absent. |
| 11 | `import wandb` unconditional. | Imported only with `--wandb`. |

Roadmap Rule 2 (no ground-truth leakage into erased packets) was already satisfied by the code;
`tests/test_reproducibility.py` now proves it (garbage in erased packets changes nothing).

### Verification (what was actually run)
* CIFAR-10 downloaded into `./data` (`data/cifar-10-batches-py` present).
* `python -m pytest tests/test_reproducibility.py -v`: **17 passed**.
* `python -m pytest tests -q --ignore=tests/test_cifar10.py`: **103 passed**.
* Not run: `tests/test_cifar10.py` (dataset test; assumed correct), the smoke evaluation, any training.

### Findings from the provided checkpoint (`checkpoints/train_split_adjsccq`, produced by the pre-Phase-0 script)
Read from its `metrics.json`/`oracle_ranking.json`; nothing was re-run.

* Trained on the 40k split, best epoch 60, init seed 42. Clean PSNR / SSIM: train 31.23 dB / 0.9647,
  validation 31.18 dB / 0.9643, **test 31.22 dB / 0.9647** (no overfitting: gap 0.05 dB).
* Oracle derived from the 40k **training** split: ranking `[10, 5, 3, 6, 1, 8, 4, 14, 0, 13, 2, 15, 12, 7, 9, 11]`.
  Sorted damage (dB): 12.16, 6.68, 6.43, 5.58, 5.37, 3.87, 3.71, 2.73, 2.64, 2.22, 2.20, 2.14, 2.07, 2.02, 1.97, 1.49.
  Shape matches the other model (one catastrophic channel, ~4 high, low tail) while the *indices* differ
  (that model's ranking was `[3, 7, 4, 6, ...]`): indices permute, the profile does not. Only two models so
  far; Phase 1 tests it properly.
* Old-code erasure sweep (3 seeds, mean PSNR dB; `uniform` here was sorted, `no_rs` had its own mask):

  | loss | uniform | best attention UEP | best oracle UEP | oracle - uniform |
  |---|---|---|---|---|
  | 0.30 | 28.98 | 27.26 | 27.69 | -1.29 |
  | 0.35 | 24.70 | 24.16 | 25.27 | +0.57 |
  | 0.40 | 21.03 | 22.13 | 23.06 | +2.03 |
  | 0.45 | 19.40 | 20.54 | 21.39 | +1.99 |
  | 0.50 | 18.63 | 19.36 | 19.95 | +1.33 |

  The 35-50% rows match, to two decimals, the table in `16th_oct_PROJECT_REPORT.md`, so that table
  came from this leak-free run. Its 0-30% rows (30.50 dB at 0%, 28.15 dB at 30%) do **not** match this run
  (31.22 / 28.98), so the report mixes two runs below 35% loss. Regenerate the whole table from one run.
* These numbers predate the Phase 0 fixes; they are superseded by the new evaluator.

### Behaviour changes that affect numbers
`no_rs` now faces the master mask prefix and is no longer sorted; `uniform` is no longer sorted in the
train-split script (matters only beyond RS capacity, > 33% loss); 96-packet schemes keep their previous masks.
Result JSON shape is now `{metadata, seed_summary, paired_vs_baseline, insights, rows}`.

---

## Phases 1-5: code written (not yet executed)

**Status:** all code, tests, the orchestrator and the runbook are written. **Nothing in this section has been
executed; the new unit tests have not been run.** Treat every claim below as "implemented", not "verified",
until the `verify` phase passes (see RUNBOOK section 3).
**Date:** 2026-10-17.

### Architecture (one implementation of each rule)

```
coding/   scaling.py     scheme templates -> any channel count / packet size, GF(256) sub-blocking
          interleave.py  sequential vs round-robin transmission order (burst study)
          erasure_channel.py  + GilbertElliottChannel, master-mask helpers
experiments/
  reproducibility.py     seeding, provenance, seed statistics, test-access log
  model_io.py            ModelBundle (encoder, decoder, optional inpainter), reconstruct()
  sensitivity.py         per-image leave-one-out damage, ranking loss, MAC counter
  importance_sources.py  attention / energy / variance / l1 / taylor / predicted / oracle_perimage / static:<name>
  pipeline_core.py       transmit(), CellAccumulator, precompute(), parallel runner
  reporting.py           tables, standard figures, automatic insights
  stats_utils.py         ranks, bootstrap, paired differences, crossover (no scipy)
  plotting.py / wandb_utils.py
  eval_rs_uep.py (+_oracle)  eval_burst.py  eval_digital_baseline.py  eval_kodak.py
  train_sensitivity_head.py  train_erasure_robust.py  train_highres.py
  multiseed_analysis.py  attention_vs_damage.py  transfer_sensitivity.py  feature_saliency.py
  select_scheme.py  make_report.py  download_data.py
data/     splits.py (40k/10k/10k, model seed separate from split seed)  datasets.py (CIFAR-100, STL-10, Kodak, crops)
models/   inpainting.py (LatentInpainter), adjsccq.py + SensitivityHead, adjscc.py decoder takes spatial_hw
run_pipeline.py          dry-run-by-default orchestrator for all phases
tests/                   test_scaling_and_channels.py, test_phase_modules.py, test_digital_baseline.py (+ Phase 0 tests)
```

### Phase 1: multi-seed models + Track A
* `train_adjsccq_train_split.py` gained `--seed` (model), `--split_seed` (partition, default 42),
  `--epochs`, `--k_over_n`, `--skip_rs_sweep`; the oracle now adapts to any channel count.
* `multiseed_analysis.py`: permutation-invariant comparison (sorted profile, bands, top-4 share).
* `attention_vs_damage.py`: Figure 2 data and figures, pooled and **within-image** correlations with bootstrap CIs.
* `transfer_sensitivity.py` (A.1): frozen model, CIFAR-10 validation vs CIFAR-100 / STL-10 test; Spearman / Pearson /
  Kendall / top-4 overlap with bootstrap CI; decision rule printed; native ranking files for the follow-up sweep.
* `feature_saliency.py` (A.2): energy, variance, L1, Taylor, attention, random, static table vs true damage.
* `train_sensitivity_head.py` (A.3): label generation (cached), SmoothL1 + pairwise hinge, 5 head seeds,
  validation-based selection, baselines, protection regret, MAC fraction check, optional test evaluation.
* `eval_rs_uep.py`: every importance source in one sweep, `--split val|test`, `--dataset`, `--workers`,
  per-image PSNR arrays, paired gains, insights, figures; `select_scheme.py` picks the allocation on validation.

### Phase 2: Track B
`train_erasure_robust.py`: channel / packet / mixed dropout, per-batch or per-sample rate, curriculum, clean-batch
fraction, encoder frozen then joint, optional inpainter (B.2), validation robustness score for checkpoint choice,
oracle recomputed for the fine-tuned model, before/after figures.

### Phase 3: Track C
`coding/scaling.py` + `eval_kodak.py` (native 768x512, 128-byte packets, MS-SSIM, optional LPIPS, saved reconstructions),
`train_highres.py` fallback, k/n sweep via `--k_over_n` + channel-count-generic evaluation.

### Phase 4: Track D
`GilbertElliottChannel` (parameterised by long-run loss and mean burst length), `coding/interleave.py`,
`eval_burst.py` (iid reference, sequential vs round-robin, interleave gain), `eval_digital_baseline.py`
(JPEG 2000 / JPEG / WebP, BPG if installed).

### Phase 5: Track E and orchestration
`make_report.py` (Figure 3, model comparison, cliff figure, k/n figure, headline table, consolidated `INSIGHTS.md`),
`run_pipeline.py` (all phases, dry-run default, skip-existing, manifest).

### Deviations from the roadmap (and why)

| Roadmap says | What the code does | Reason |
|---|---|---|
| Assert Spearman r_s > 0.85 between the damage vectors of independently trained models | Compare **sorted profiles**, band counts, top-4 share; plot the index-aligned matrix only as evidence that labels permute | A conv net is symmetric under channel relabelling, so index-aligned correlation across models is ~0 by construction; the roadmap's check is ill-posed |
| RS(1536, 768) etc. for Kodak | Tiers split into equal sub-blocks of <= 255 packets, same total data/parity | GF(2^8) codewords have at most 255 symbols |
| Gradient saliency by autograd | Closed form `mean |z|` (d||z||^2/dz = 2z); plus a Taylor proxy using the decoder | The literal gradient is trivially 2z; the Taylor term is the informative gradient-based variant |
| Script named `test_feature_saliency.py` | `feature_saliency.py` | A `test_*.py` name inside `experiments/` would be collected by pytest |
| Oracle/UEP numbers from the test set | Rankings from the 40k train split only; schemes selected on validation; test logged | Rule 1 |
| Channel-level dropout only | Also packet-level and mixed | A failed RS block erases 16-symbol packets, not whole channels |
| Kodak 768x512 only | The four portrait images are rotated to landscape | One tensor shape; resolution and content unchanged |
| Evaluate BPG | JPEG 2000 / JPEG / WebP always; BPG when `bpgenc`/`bpgdec` exist | BPG is not installed here; the substitutes are strong baselines |

### Known limitations and risks
* **Unverified code.** ~5k new lines have not been executed. The unit tests were written to catch the likely
  failures (round-trip losslessness, shared losses, determinism, imports of every script) but have not been run.
* **Runtime.** Pure-numpy Reed-Solomon dominates. Estimate from the smoke test before launching `final`;
  full test sweeps scale as images x seeds x rates x methods. Use `--workers`.
* **Seed-42 model** was trained by the old script (no provenance); it is valid but its sweeps must be re-run.
* **Taylor / per-image oracle** need the image or a decoder copy at the transmitter: they are bounds/diagnostics, not
  deployable rankings. Only `static` and `predicted` are.
* **Finite block length** at Kodak scale softens the uniform-RS threshold; discussed, not hidden.
* **Kodak with a CIFAR-trained model** may reconstruct poorly; the script warns and `train_highres` is the fix.
* `test_cifar10.py` and the smoke evaluation remain unrun.

### Next
1. Approve running the `verify` phase: `python -m pytest tests -q` then `python run_pipeline.py --only smoke_eval --execute`;
   fix whatever breaks (expect some small integration bugs in a first run of this much new code).
2. Re-evaluate the seed-42 model with the new evaluator (validation first), then train seeds 100 and 2024.
3. Work through RUNBOOK sections 4-8; append a results entry here after each phase (numbers, figures, decisions).

---

## Execution log (2026-10-17 / 18): first end-to-end run

This section records what actually happened when the code was run, in order, including every problem and
decision. Times are the machine's clock (IST).

### Verification
* `pytest tests`: 179 passed on the first run (later 180); the same suite passes on Kaggle (Python 3.13, torch 2.11).
* Every script was smoke-tested on tiny inputs before any long run (attention, saliency, head, burst, digital,
  robust training with and without the inpainter, multiseed, selection, report, transfer, Kodak).
* Preflight (`python -m experiments.preflight --wandb_test --kaggle_test`): W&B logged a test run, Kaggle authenticated,
  CIFAR-10 md5-verified.

### Problems found and fixed while running
| Problem | Cause | Fix |
|---|---|---|
| `test_cifar10` hung | default data folder did not exist, so it re-downloaded 170 MB at ~50 kB/s | test now uses the verified `./data` |
| Kaggle kernel ran 35+ min then errored | repo's own tracked `wandb/` folder (with a symlink) crashed my copy step; CIFAR-10 download from Toronto is slow | run in `/tmp`, never raise on copy errors, use the CIFAR-10 copy hosted on Kaggle (md5-verified by torchvision) |
| two kernels failed instantly | my patch split an f-string across lines | compile-check every launcher in the test suite |
| Windows console crashed on `Δ` / `±` | cp1252 stdout | UTF-8 reconfigure in `experiments/__init__.py`; explicit UTF-8 for every text file |
| training logged to the wrong W&B account | script called wandb without loading `.env`; another account was logged in on the machine | `.env` is loaded on package import |
| Kodak/digital insight texts misleading ("break-even at 0.00", "1064% of oracle gain") | edge cases of the generated text | guarded; reference ranking is now the per-image one |
| a methodology gap | the best ranking source (energy) was not in the validation sweep, so its tier allocation would have been chosen on test numbers | added a validation sweep + selection for energy/L1; reports take allocations only from validation selection files |
| selection sweeps too slow | single-threaded workers: ~28 min per cell | selection sweeps use 3 seeds x 5,000 validation images; final tables keep 5 seeds x all 10,000 |
| Kaggle GPU queue | a GPU job waited ~2 h; sessions are capped at 2 | per-step timeouts; CPU kernels (no queue, ~7.5 h per model) and GPU kernels run in parallel |

### Training instability of the original recipe (recipe v1) and the change to recipe v2
Recipe v1 = Adam, constant lr 1e-3, no gradient clipping, quantiser sigma annealed 1.0 -> 0.05 over 50 epochs (the project's
original recipe). Runs under v1 (figure `training_stability_v1`, data from the training logs):

| run | best val PSNR | last-epoch val PSNR | largest one-epoch drop |
|---|---|---|---|
| CPU seed 42 | 30.52 dB | 21.05 dB | 8.5 dB |
| GPU seed 42 | 30.84 dB | 30.84 dB | 4.0 dB |
| GPU seed 100 | 31.30 dB | 31.30 dB | 3.3 dB |
| GPU seed 2024 | 26.49 dB (epoch 21) | 24.12 dB | 3.6 dB |
| GPU k/n = 1/3 | 26.70 dB | 12.48 dB | 13.6 dB |

Best-epoch selection on validation protects the saved model in stable runs, but two models (GPU seed 2024 and k/n = 1/3)
never converged, so their damage profiles (worst channel only 4.6 dB for seed 2024) are not comparable. Rather than drop
those runs silently, the recipe was changed for **all** models: **recipe v2 = cosine learning-rate decay (1e-3 -> 1e-5) and
gradient-norm clipping at 1.0**, everything else identical. All models are retrained from scratch with v2 on Kaggle T4 GPUs
(`v2_a`: seeds 42, 2024 + the three erasure-robust fine-tunes; `v2_b`: seed 100 + the three k/n models). The v1 runs are
kept as a documented pilot. The recipe change was motivated by training divergence observed on validation data only; no test
number influenced it.

### Pilot results (v1, CPU-trained seed 42; validation split unless stated)
* Clean test PSNR 30.57 dB (SSIM 0.9594). Worst channel costs 11.9 dB when lost; next 5.4 dB; least 1.7 dB.
* Attention vs true damage (2,000 val images): within-image Spearman +0.18 (95% CI 0.17-0.19), top-4 overlap 0.33 (chance 0.25):
  weakly informative, not zero.
* A.1 transfer to CIFAR-100: Spearman +0.988, Pearson +0.999, top-4 overlap 0.75 -> a static table transfers.
* A.2 proxies (within-image Spearman): energy +0.82, L1 +0.81, static table +0.75, Taylor +0.46, variance +0.35, attention +0.18.
* A.3 head: top-4 overlap 0.88 vs 0.75 (static) vs 0.25 (random); protection regret 0.27 dB vs 1.68 dB vs 13.0 dB; 0.039% of encoder MACs.
* Erasure sweep at 40% loss (10,000 val images, 5 seeds; gain over uniform RS, paired, 95% CI +-0.02-0.06 dB):
  energy +1.94, L1 +1.90, true per-image leave-one-out +1.83, trained head +1.71, Taylor +1.53, variance +1.22, static
  oracle +1.20, attention +0.95, random order +0.86. Uniform RS is better by 1.5-3 dB at 30% loss (break-even ~34%).
* Scheme selection (validation): `three_tier_25_50_25` is best from 35% loss up for every ranking source; `two_tier_50_50`
  wins at 10-30%; selected union = {three_tier_25_25_50, three_tier_25_50_25, two_tier_25_75, two_tier_50_50}.

These pilot numbers are not final: they come from a v1 model. They are repeated on the v2 family below.
