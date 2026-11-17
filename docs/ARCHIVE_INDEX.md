# Archive index: what is in `archive/`, where it came from, and why it is not used

Nothing in `archive/` feeds a result in the paper. It is kept so the project history stays complete and
auditable. Files were moved with `git mv` on 2026-11-08, so `git log --follow <new path>` shows each file's
full history under its old name. Every item below lists its **original path**.

Archived Python scripts import modules that still live in the active packages (`data`, `models`,
`coding`, `experiments`). Run them from the repository root with the root on the import path, for example:

```bash
PYTHONPATH=. python archive/legacy_code_weeks01_to_05_deepjscc_and_adjscc_over_awgn/experiments/week01_overfit_deepjscc_awgn_ten_cifar10_images.py
```

Their default `--config` paths were updated to the archived config files.

---

## 1. `legacy_code_weeks01_to_05_deepjscc_and_adjscc_over_awgn/`

Weeks 1 to 5: reproducing DeepJSCC and ADJSCC over an **AWGN** channel, before quantisation, Reed-Solomon
coding or packet erasures existed in the project. The paper uses an identity channel with packet erasures, so
none of this is used. The backbones themselves (`models/deepjscc.py`, `models/adjscc.py`,
`models/Attention.py`) and the AWGN layer (`models/channel.py`) stay active because the current models
and unit tests build on them.

| Archived file | Original path | What it is |
|---|---|---|
| `experiments/week01_overfit_deepjscc_awgn_ten_cifar10_images.py` | `experiments/overfit_test.py` | Week-1 sanity check: overfit DeepJSCC on 10 CIFAR-10 images over AWGN (target > 35 dB). |
| `configs/week01_overfit_deepjscc_awgn.yaml` | `configs/week1_overfit.yaml` | Its config. |
| `notebooks/week01_deepjscc_awgn_ten_image_overfit_sanity_check.ipynb` | `archive/pynbs/week-1-overfitting_check.ipynb` | Kaggle notebook version of the same check. |
| `experiments/week04_train_adjscc_awgn_snr_sweep_full_cifar10.py` | `experiments/train_adjscc.py` | Week 4-5: ADJSCC trained over AWGN with per-sample SNR sampling. |
| `configs/week04_train_adjscc_awgn.yaml` | `configs/week4_adjscc.yaml` | Its config. |
| `experiments/scratch_eval_week04_adjscc_awgn_checkpoint.py` | `scratch_eval.py` (repo root) | Scratch PSNR-vs-SNR evaluation of the week-4 ADJSCC checkpoint. |
| `experiments/toy_demo_jscc_awgn_reconstruction_figure.py` | `experiments/toy_demo.py` | Side-by-side reconstruction figure over AWGN (demo only). |
| `experiments/toy_sweep_demo_jscc_awgn_snr.py` | `experiments/sweep_demo.py` | SNR sweep built on the toy demo. |
| `experiments/toy_sweep_compare_deepjscc_vs_adjscc_awgn.py` | `experiments/sweep_compare.py` | DeepJSCC vs ADJSCC SNR comparison built on the toy demo. |

## 2. `legacy_code_phases03_to_05_identity_channel_before_train_split_and_phase0/`

Phases 3 to 5 (14 Sep to 4 Oct 2026). The project switched to an identity channel, added the 256-level
quantiser and built the first RS pipeline. These scripts trained on the **full 50k CIFAR-10 training set
with no validation split**, or analysed results before the Phase-0 integrity fixes (shared loss masks,
side-information rule, provenance). They were superseded by `experiments/train_adjsccq_train_split.py`,
`experiments/attention_vs_damage.py` and `coding/rs_pipeline.py`.

| Archived file | Original path | What it is |
|---|---|---|
| `experiments/phase3_overfit_adjsccq_identity_channel.py` | `experiments/overfit_test_adjsccq.py` | 10-image overfit smoke test of ADJSCC-Q (quantised), identity channel. |
| `configs/phase3_overfit_adjsccq_identity_channel.yaml` | `configs/adjsccq_identity_overfit.yaml` | Its config. |
| `experiments/phase3_overfit_unquantized_adjscc_identity_channel.py` | `experiments/overfit_test_adjscc_identity.py` | The same overfit test without the quantiser (baseline for the quantisation-loss delta). |
| `configs/phase3_overfit_unquantized_adjscc_identity_channel.yaml` | `configs/adjscc_identity_overfit.yaml` | Its config. |
| `experiments/phase3_train_adjsccq_identity_full_cifar10_no_val_split.py` | `experiments/train_adjsccq_identity.py` | First full ADJSCC-Q trainer (no validation split). |
| `experiments/phase3_train_unquantized_adjscc_identity_full_cifar10_no_val_split.py` | `experiments/train_adjscc_identity.py` | Its unquantised counterpart. |
| `configs/phase3_train_unquantized_adjscc_identity_full_cifar10.yaml` | `configs/adjscc_identity_train.yaml` | Config of the unquantised trainer. |
| `experiments/phase5_attention_vs_oracle_correlation_superseded_by_attention_vs_damage.py` | `experiments/attention_correlation.py` | First attention-vs-oracle correlation (pre-Phase-0). Replaced by `experiments/attention_vs_damage.py`. |
| `coding/phase3_symbol_and_packet_tier_layout_superseded_by_rs_pipeline.py` | `coding/tiering.py` | Prototype symbol/packet tier layout (0.2/0.3/0.5 tiers). Replaced by `coding/rs_pipeline.py` + `coding/scaling.py`. |
| `tests/test_phase3_symbol_and_packet_tier_layout.py` | `tests/test_tiering.py` | Its unit test. It now imports the archived module from `../coding/` and is no longer part of `pytest tests`. |

## 3. `superseded_documentation_2026-09_to_2026-10/`

Status reports and result write-ups from before the Phase-0 audit (17 Oct 2026). Their numbers come from
code paths that were later fixed: separate loss masks per method, sorted `uniform`, and no provenance.
**Do not quote them.** The current documents are in `docs/`.

| Archived file | Original path | Note |
|---|---|---|
| `2026-09-14_phase3_report_scalar_quantizer_integration_adjsccq_identity_channel.md` | `documentations/PHASE3_DEEPJSCCQ_REPORT.md` | Quantiser integration report. |
| `2026-09-20_project_status_and_implementation_roadmap_before_rs_pipeline.md` | `documentations/20thsept_PROJECT_STATUS_AND_ROADMAP.md` | Repository status and the first roadmap. |
| `2026-09-20_phase3_five_step_record_attention_mapping_rs_codec_tier_layout.md` | `documentations/PHASE3_FIVE_STEPS_IMPLEMENTED.md` | Attention mapping, symbol ordering, RS codec, tier layout. |
| `2026-09-27_phase4_status_interleaved_rs_pipeline_erasure_channel_clean_baseline.md` | `documentations/PHASE4_RS_INTERLEAVING_STATUS.md` | Interleaved RS pipeline and erasure channel. |
| `2026-09-30_phase5_oracle_importance_and_uep_validation_results_pre_phase0.md` | `documentations/PHASE5_ORACLE_RESULTS.md` | First oracle results. |
| `2026-09-30_phase5_validation_sweep_and_oracle_analysis_results_pre_phase0.md` | `documentations/PHASE5_VALIDATION_RESULTS.md` | First validation sweep. |
| `2026-10-01_phase5_attention_uep_vs_oracle_uep_comparison_pre_phase0.md` | `documentations/PHASE5_ATTENTION_VS_ORACLE_COMPARISON.md` | First attention-vs-oracle comparison. |
| `2026-10-04_train_validation_test_split_workflow_of_train_adjsccq_train_split.md` | `documentations/TRAIN_SPLIT_RS_UEP.md` | Original description of the 40k/10k/10k trainer; superseded by the runbook. |
| `2026-10-16_project_chronology_report_WARNING_table_mixes_two_runs_below_35pct_loss.md` | `documentations/16th_oct_PROJECT_REPORT.md` | Its results table mixes two runs below 35% loss (found in the Phase-0 audit). |
| `byte_identical_duplicate_of_docs_literature_review_formerly_in_documentations.md` | `documentations/LITERATURE_REVIEW.md` | Exact copy of the literature review now in `docs/06_literature_review/`. |

## 4. `superseded_results/pre_phase0_adjsccq_seed42_model_trained_2026-10_31.22dB/`

Original path: `checkpoints/train_split_adjsccq/{metrics.json, oracle_ranking.json}`.

This is the first 40k/10k/10k seed-42 ADJSCC-Q model, trained by the pre-Phase-0 script (clean test PSNR
31.22 dB). Its RS sweep predates the shared-mask fix. The paper's seed-42 model is the GPU-trained one in
`published_results/gpu/models/seed42/` (30.88 dB). The pipeline default `--seed42_dir` now points to
`results/gpu/models/seed42`.

## 5. `early_wandb_local_run_mirrors_2026-09-04_and_2026-09-14/`

Original path: `wandb/`. These are local Weights & Biases run mirrors from the week-1 and phase-3 trainings.
They were committed by mistake before `wandb/` was added to `.gitignore`. The folder includes
`debug-cli.<username>.log`. **Remove it before making the repository public or submitting under double-blind
review.**

## 6. `kaggle_kernels_rejected_training_recipes_v2_cosine_clip_and_v3_low_lr/`

Original path: `kaggle/kernels/rs-uep-jscc-{v2-a,v2-b,v3-a,v3-b}/`. These Kaggle jobs trained the alternative
recipes v2 (cosine learning-rate decay + gradient clipping at 1.0) and v3 (lr 3e-4 + v2). Both were
**rejected**: they did not fix the training instability of recipe v1 (see the progress log, "Training recipe
decision", and `published_results/analysis/convergence_ledger.json`). Every paper model uses recipe v1.

## 7. `kaggle_kernels_infrastructure_unit_tests_smoke_and_timing/`

Original path: `kaggle/kernels/rs-uep-jscc-{tests,smoke,timing,timing-cpu}/`. These are infrastructure jobs
that ran the unit tests, a smoke run and epoch-timing benchmarks on Kaggle. They produced no results.

The kernels that produced published results stay in `kaggle/kernels/`. `kaggle/make_kernels.py` can
regenerate any kernel, including the archived ones, into `kaggle/kernels/`.
