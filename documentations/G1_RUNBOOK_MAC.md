# G1 runbook (Apple-silicon Mac, MPS): is UEP an ADJSCC effect or a Deep JSCC effect?

G1 trains **DeepJSCC-Q** (ADJSCC-Q with its 10 attention modules removed, nothing else
changed) and runs the same protocol that produced the ADJSCC-Q results. It then compares the
two families with a decision rule fixed in advance.

Design decisions behind every choice below: [`ARCHITECTURE_DECISIONS.md`](ARCHITECTURE_DECISIONS.md).
Status: **code written, nothing executed yet.** Run the steps in order; each one says what
to check before moving on.

All decisions were confirmed by the owners on 2026-11-06:

* the training recipe (AD-05);
* 4 seeds plus spare 21 (AD-06);
* running the same-hardware anchor (AD-07);
* the decision thresholds, now **frozen** (AD-09).

Nothing blocks the run.

All commands are for zsh/bash, run from the repository root.

---

## 0. What runs where

| Step | What | Device | Output |
|---|---|---|---|
| 2 | unit tests, dry run | CPU | terminal |
| 3 | 2-epoch smoke run + tiny sweep (numbers never quoted) | MPS + CPU | `results/g1/smoke_*` |
| 4 | train DeepJSCC-Q, seeds 42, 100, 7, 13 | **MPS** | `results/g1/deepjsccq/seed*/` |
| 5 | profiles, proxies (A.2), head (A.3), validation sweep, scheme selection | CPU | `results/g1/deepjsccq/...` |
| 6 | test sweeps (decision rule frozen, AD-09) | CPU | `.../seed*/test_sweep/` |
| 7 | same-hardware ADJSCC-Q anchor (AD-07) | MPS + CPU | `results/g1/adjsccq_anchor/` |
| 8 | comparison of record, including the anchor | CPU | `results/g1/compare_with_anchor/` |

Training uses MPS. The evaluation sweeps use the CPU with parallel workers: the Reed-Solomon
codec is pure numpy, so the GPU would not help, and results are bit-identical for any worker
count (AD-08).

The ADJSCC-Q reference numbers are read from `published_results/gpu/...`, which is in git, so
no ADJSCC-Q checkpoint is needed on the Mac.

---

## 1. One-time setup

```bash
git pull                                   # the branch with G0/G1
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pillow matplotlib pyyaml tqdm pytest
python -m pip install wandb                 # optional, only if you pass --wandb

python -c "import torch; print(torch.__version__, 'MPS available:', torch.backends.mps.is_available())"
```

The last line must print `MPS available: True`. That needs Apple silicon, macOS 12.3+ and
torch ≥ 2.0. Then:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1        # put this in ~/.zshrc too (AD-08)
python -m experiments.download_data --cifar10 --data_dir ./data
```

Pick the worker count for the CPU sweeps: use the number of performance cores.

```bash
sysctl -n hw.perflevel0.physicalcpu         # e.g. 4 on M1/M2, 6-12 on Pro/Max
export W=$(sysctl -n hw.perflevel0.physicalcpu)
```

Long runs must not be interrupted by sleep. Prefix every long command with
`caffeinate -i` (shown below), keep the Mac on power, and preferably run inside `tmux` or
`screen` so a closed terminal does not kill the job.

---

## 2. Verify (no training)

```bash
python -m pytest tests -q
python -m pytest tests/test_deepjsccq.py -v
python run_pipeline.py --list                         # the [g1] block lists every G1 step
python run_pipeline.py --phase g1 --train_device mps  # DRY RUN: prints every command, runs nothing
```

**Gate:** all tests pass. `test_deepjsccq.py` checks that:
* DeepJSCC-Q equals ADJSCC-Q minus the attention modules;
* old checkpoints still load as ADJSCC-Q;
* the registry builds ADJSCC-Q with bit-identical initialisation;
* a loss-free transmission is lossless on DeepJSCC-Q;
* the comparison applies the decision rule correctly;
* the `g1` phase is well formed.

If anything fails, stop and report the output. Do not work around it.

---

## 3. Smoke test on MPS (numbers are never quoted)

```bash
caffeinate -i python run_pipeline.py --only g1_smoke_train --train_device mps --execute
```

This trains 2 epochs (seed 7) and computes the oracle on 500 images. Log:
`results/logs/g1_smoke_train.log`. Check:

1. `results/g1/smoke_train/metrics.json` has `"arch": "deepjsccq"` and `"train_device": "mps"`.
2. The log shows two `epoch ... | validation PSNR ...` lines. **Note the time per epoch:**
   ×60 is the time per real model. The ADJSCC-Q reference is 87 min for 60 epochs on an M2
   (RUNBOOK §4.1); DeepJSCC-Q should be slightly faster.
3. Warnings about `pin_memory` not being supported on MPS are harmless.

Then smoke-test one evaluation cell, which also measures the sweep speed:

```bash
caffeinate -i python -m experiments.eval_rs_uep \
  --checkpoint results/g1/smoke_train/best_val.pth \
  --oracle_ranking_file results/g1/smoke_train/oracle_ranking.json \
  --data_dir ./data --split val --importance oracle,energy --schemes three_tier_25_50_25 \
  --erasure_rates 0.0,0.4 --seeds 0,1 --max_images 200 --include_no_rs --include_random_control \
  --workers $W --output_dir results/g1/smoke_eval --name smoke
```

**Gate:** in `results/g1/smoke_eval/smoke.md`, at loss 0.0 every method has the same PSNR as
`uniform` (a lossless round trip). Note the wall time: full sweeps scale linearly with
images × seeds × rates × methods.

---

## 4. Train the four DeepJSCC-Q models (MPS)

All four in sequence (skips any that already finished):

```bash
caffeinate -i python run_pipeline.py --only g1_train_seed42,g1_train_seed100,g1_train_seed7,g1_train_seed13 \
  --train_device mps --skip_existing --execute
```

Equivalent direct command for one seed (the orchestrator runs exactly this):

```bash
caffeinate -i python -m experiments.train_adjsccq_train_split \
  --config configs/deepjsccq_identity_train.yaml --arch deepjsccq --seed 42 --split_seed 42 \
  --skip_rs_sweep --device mps --data_dir ./data --output_dir results/g1/deepjsccq/seed42
```

Add `--wandb` to either command to log the run to W&B.

**Check each model** in `results/g1/deepjsccq/seed*/metrics.json`:

| Field | Expect | If not |
|---|---|---|
| `best_epoch` | ≥ 50 (the convergence rule) | Not converged. **Do not delete or hide it.** Train the spare seed 21 (below) and keep both; the comparison reports and excludes it automatically (AD-06) |
| `clean_validation.psnr_db_mean_per_image` | about 30–31.5 dB (ADJSCC-Q: 30.8–31.7) | Note it. A clearly lower PSNR is itself a finding about attention, not a bug |
| `arch`, `train_device` | `deepjsccq`, `mps` | Wrong config or device: rerun |

Spare seed, only if a run did not converge:

```bash
caffeinate -i python -m experiments.train_adjsccq_train_split \
  --config configs/deepjsccq_identity_train.yaml --arch deepjsccq --seed 21 --split_seed 42 \
  --skip_rs_sweep --device mps --data_dir ./data --output_dir results/g1/deepjsccq/seed21
```

If you use the spare, edit `G1_SEEDS` in `run_pipeline.py` to list the seeds actually used
before steps 5–7, and record it in `PROGRESS_LOG.md`.

---

## 5. Analysis on the validation split (CPU)

```bash
caffeinate -i python run_pipeline.py \
  --only g1_multiseed,g1_saliency_A2,g1_head_A3,g1_val_sweep,g1_select \
  --workers $W --skip_existing --execute
```

What each step gives and what to look at:

| Step | Output | Look at |
|---|---|---|
| `g1_multiseed` | `results/g1/deepjsccq/multiseed/multiseed.md` | top-4 share and worst-channel damage per seed, against ADJSCC-Q's 45–53% and 8.7–17.1 dB. This is the first answer to "is importance still concentrated without attention?" |
| `g1_saliency_A2` | `.../seed42/saliency/saliency.md` | Does energy / L1 still track damage (ADJSCC-Q: within-image Spearman +0.84)? No attention row here, by design |
| `g1_head_A3` | `.../seed42/head/head_metrics.md` | Head top-4 overlap vs the static table (ADJSCC-Q: 0.92 vs 0.88). Labels 50k images on the CPU, the slowest analysis step |
| `g1_val_sweep` | `.../seed42/val_sweep/val_sweep.md` | Validation only: 3 channel seeds × 5,000 images, all 7 schemes |
| `g1_select` | `.../seed42/val_sweep/selected_schemes.json` | The tier allocations the test sweeps will use |

---

## 6. Test sweeps (the decision rule is frozen)

The owners fixed the thresholds of **AD-09** on 2026-11-06, before any G1 model existed:

* C1: top-4 share ≥ 0.40;
* C2: oracle gain at 40% loss ≥ +1.5 dB with CI > 0;
* C3: random < 0.5 × oracle;
* FLAT: top-4 < 0.30 and oracle within 0.5 dB of random.

These are the defaults of `compare_architectures`. Do not pass other values. Make sure the
commit that froze AD-09 is pushed before running this step, so the git history shows the rule
came first.

```bash
caffeinate -i python run_pipeline.py \
  --only g1_final_seed42,g1_final_seed100,g1_final_seed7,g1_final_seed13 \
  --workers $W --skip_existing --execute
```

Each sweep covers 5 channel seeds × all 10,000 test images. Seed 42 uses `oracle, predicted,
energy, l1`; the others use `oracle, energy`; all include the random control and `no_rs`, with
the schemes chosen in `g1_select`. Every run appends to `results/test_access_log.jsonl`.
**Run each test sweep once.** If one crashes, rerunning it is fine (record why); rerunning to
get a better number is not.

---

## 7. Same-hardware anchor (AD-07)

This separates "different hardware" from "different architecture": ADJSCC-Q seed 42 is
retrained on this Mac and test-swept with the published ADJSCC-Q validation selection, so it
needs no new tuning.

```bash
caffeinate -i python run_pipeline.py --only g1_anchor_train --train_device mps --skip_existing --execute
caffeinate -i python run_pipeline.py --only g1_anchor_final --workers $W --skip_existing --execute
```

Check `results/g1/adjsccq_anchor/seed42/metrics.json`: `best_epoch` ≥ 50, `arch` = `adjsccq`,
`train_device` = `mps`.

---

## 8. Compare: the comparison of record includes the anchor

```bash
python run_pipeline.py --only g1_compare_with_anchor,g1_compare --execute
```

`g1_compare_with_anchor` is the comparison of record (AD-07); `g1_compare` is the same without
the anchor. Both read:

* ADJSCC-Q seeds 42, 100, 7, 13, 21 and 2024 from `published_results/gpu/` (2024 is shown but
  excluded as not converged);
* DeepJSCC-Q from `results/g1/deepjsccq/`;
* the anchor from `results/g1/adjsccq_anchor/` (`g1_compare_with_anchor` only), as its own family
  `adjsccq_anchor`.

If the anchor matches the published ADJSCC-Q seeds (top-4 share, gains), hardware is ruled out
as the explanation for any ADJSCC/DeepJSCC difference.

Outputs in `results/g1/compare_with_anchor/` (and `results/g1/compare/`):

* `compare.md`: per-model and per-family tables, the verdict per family, the differences between
  family means, and the insights;
* `compare.json`: everything, with provenance;
* `gain_vs_concentration.{png,pdf}`: top-4 share against oracle gain at 40% loss, one point per model;
* `sorted_profiles.{png,pdf}`: sorted LOO damage per model;
* `oracle_gain_by_family.{png,pdf}`: mean oracle gain against loss rate per family.

How to read the DeepJSCC-Q verdict:

| Verdict | Meaning for the paper |
|---|---|
| **criteria met** | Unequal sensitivity and the UEP gain survive without attention. The claim can be stated for quantized Deep JSCC latents, with attention just a poor sensor of it |
| **no exploitable structure** | The concentration disappears without attention: the attention gating creates the imbalance UEP exploits. The paper narrows to attention-based JSCC, with a sharper story |
| **intermediate** | Report how UEP gain scales with concentration (the scatter plot), not a yes/no |
| **incomplete** | A sweep or a random control is missing; see the notes in `compare.md` |

---

## 9. Bring the results back

Commit the small files: JSON, Markdown, figures and logs under `results/g1/`. Checkpoints are
about 0.65 MB each and may be committed too. Leave out `head_labels/` (large label caches) and
per-image `.npz` arrays. Append a dated "G1 results" section to
`documentations/research_progress/PROGRESS_LOG.md` with the per-family table, the verdict, any
non-converged runs and anything that deviated from this runbook.

---

## 10. Time budget (estimate; replace with the smoke-test measurements)

| Item | Estimate on an M2 | Based on |
|---|---|---|
| One DeepJSCC-Q training (60 epochs + train-split oracle) | ~1.5 h | 87 min for ADJSCC-Q on M2/MPS |
| Four models | ~6 h | |
| A.3 head labels (50k images × 17 decodes, CPU) | ~0.5–1 h | |
| Validation sweep (3 seeds × 5k images × 8 rates × all schemes) | a few hours | smoke-test cell time × cells |
| Four test sweeps (5 seeds × 10k images × 8 rates) | several hours each | smoke-test cell time × cells |
| Anchor: one ADJSCC-Q training + one test sweep | ~1.5 h + several hours | as above |

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `--device mps requested but MPS is not available` | Intel Mac, old macOS, or a CPU-only torch build | check `torch.backends.mps.is_available()`; reinstall torch from pip |
| `NotImplementedError: ... not currently implemented for the MPS device` | an op without an MPS kernel | `export PYTORCH_ENABLE_MPS_FALLBACK=1` (the orchestrator sets it with `--train_device mps`) |
| `UserWarning: 'pin_memory' ... not supported on MPS` | the shared loader sets `pin_memory=True` | harmless; left unchanged so other platforms behave as before |
| training stalls overnight | the Mac slept | `caffeinate -i`, power adapter, `tmux` |
| validation PSNR collapses mid-run (drops of several dB) | recipe-v1 instability, also seen on ADJSCC-Q | let the run finish (best-validation checkpointing protects the model); apply the convergence rule in step 4 |
| `importance source 'attention' needs a bottleneck attention gate` | `attention` passed for DeepJSCC-Q | expected; DeepJSCC-Q has no attention; drop it from `--importance` |
| a sweep is very slow | too few workers | `--workers $W`; never more than the core count |
