# Runbook: how to run every phase

Everything below is **written but not yet executed** (see `PROGRESS_LOG.md` for the
status of each phase). Run things in the order given; each phase tells you what to
look at before moving on.

All commands are for PowerShell/bash from the repository root. Paths use forward
slashes so they work on Windows, Linux and Kaggle.

---

## 0. Rules the code enforces (so you do not have to remember them)

| Rule | Where it is enforced |
|---|---|
| Test split only for final tables | every script defaults to `--split val`; each test access is appended to `results/test_access_log.jsonl` |
| Tier allocation chosen on validation, not test | `select_scheme.py` refuses a test sweep; `final_*` steps read its output |
| Same packet losses for every method | one master mask per (batch, seed, rate); smaller schemes read its first columns; burst study shares one timeline mask |
| No ground-truth leakage after RS failure | failed symbols are set to 0.0 in latent space; unit test feeds garbage into erased packets |
| Side-info counted | 0 B for `uniform`/`no_rs`, one byte per channel for any ranked method; coding-only and total overhead are separate columns |
| Error bars across channel seeds, not images | `seed_summary` (mean ± sample std, 95% t-interval); gains are *paired* per seed |
| Determinism | `seed_everything` (python/numpy/torch/CUDA + cuDNN deterministic) in every entry point; results identical for any `--workers` |
| Provenance | every JSON has `metadata` (git commit, dirty flag, versions, device, command line, checkpoint SHA-256, ranking/head hashes) |

---

## 1. One-time setup

```powershell
python -m pip install torch torchvision numpy pillow matplotlib pyyaml tqdm pytest wandb
python -m pip install lpips        # optional: LPIPS on Kodak (skipped with a note if absent)
wandb login                        # optional: only if you will pass --wandb
```

Data (CIFAR-10 is already in `./data`). The rest only when you reach that phase:

```powershell
python -m experiments.download_data --cifar100 --kodak          # 169 MB + ~15 MB
python -m experiments.download_data --stl10                      # optional, 2.6 GB
```

Optional BPG (roadmap's digital codec): install `bpgenc`/`bpgdec` from
<https://bellard.org/bpg/> and put them on `PATH`. Without them the digital baseline
uses JPEG 2000, JPEG and WebP (they are stronger than JPEG at 1 kB, so the
comparison is conservative for Deep JSCC).

Weights & Biases is **always optional** (`--wandb`). Use `WANDB_MODE=offline` to log
locally and sync later.

---

## 2. The orchestrator

`run_pipeline.py` knows every step of every phase. **By default it only prints.**

```powershell
python run_pipeline.py --list                          # all steps, grouped by phase
python run_pipeline.py --phase phase1                  # print the commands of one phase
python run_pipeline.py --phase phase1 --execute --skip_existing --workers 8 --wandb
python run_pipeline.py --only train_seed100 --execute  # a single step
```

* `--skip_existing` skips a step whose outputs already exist (safe to re-run after a crash).
* `--workers N` parallelises the packet-loss sweeps over (seed, rate) cells. Results are
  bit-identical for any N. On this machine (16 cores) use 8-12.
* Logs go to `results/logs/<step>.log`; `results/logs/pipeline_manifest.jsonl` records
  command, exit code, duration and git commit for every executed step.
* Defaults: `--seed42_dir checkpoints/train_split_adjsccq` (the model you already have),
  `--data_dir ./data`, `--results results`, `--highres_dir <folder>` (only for the Kodak fallback).

You can always run the underlying command yourself; they are shown below.

---

## 3. Phase `verify` (do this first)

```powershell
python -m pytest tests -q
python run_pipeline.py --only smoke_eval --execute
```

* The unit tests include end-to-end checks on a tiny random model (lossless round trip,
  identical losses across methods, determinism) and import every script.
* `smoke_eval` runs 200 validation images, 2 seeds, 3 loss rates. **Never quote its numbers.**
  Its job is to prove the machinery works and to **measure speed**: note the wall time of
  one (seed, rate) cell; the full runs scale linearly with images x seeds x rates x methods.

Gate: all tests pass, smoke run produces `results/smoke/smoke.{json,md}` + figures.

---

## 4. Phase 1: multiple model seeds + Track A (online predictor)

### 4.1 Two more models (init seeds 100 and 2024; same 40k/10k split)

```powershell
python -m experiments.train_adjsccq_train_split --seed 100  --split_seed 42 --skip_rs_sweep --data_dir ./data --output_dir results/models/seed100
python -m experiments.train_adjsccq_train_split --seed 2024 --split_seed 42 --skip_rs_sweep --data_dir ./data --output_dir results/models/seed2024
```

Each run trains 60 epochs, writes `best_val.pth` (chosen on validation), `final.pth`,
`oracle_ranking.json` (derived from the 40k training split only) and `metrics.json`. On a
CPU this is hours per model (the original run took 87 min on an M2/MPS); a GPU (e.g. Kaggle)
is the one place it really helps. `--skip_rs_sweep` leaves the packet-loss sweeps to the faster,
parallel `eval_rs_uep`.

Seed 42 is the model you already have (`checkpoints/train_split_adjsccq`). It was produced by
the pre-Phase-0 script; its weights, ranking and clean numbers are valid, but re-evaluating it
with the new evaluator is required for the paired tables.

### 4.2 Is unequal importance a property of deep JSCC? (multi-seed)

```powershell
python -m experiments.multiseed_analysis --run seed42=checkpoints/train_split_adjsccq --run seed100=results/models/seed100 --run seed2024=results/models/seed2024 --output_dir results/phase1/multiseed
```

Read `multiseed.md`. **Do not expect the same channel indices**: a conv net is symmetric under
relabelling channels, so the roadmap's "assert Spearman > 0.85 between index-aligned damage
vectors" cannot hold. The script tests what *can* hold: the sorted damage profile (one
catastrophic channel, a few high, a low tail), band counts, and the share of damage in the top-4.
Success = every model has a > 10 dB channel and the sorted profiles agree.

### 4.3 Figure 2: attention does not predict damage

```powershell
python -m experiments.attention_vs_damage --checkpoint checkpoints/train_split_adjsccq/best_val.pth --split val --max_images 2000 --output_dir results/phase1/attention
```

Gives the pooled Pearson (the roadmap's ρ), the **within-image** Spearman with a bootstrap CI
(the more telling number), top-4 overlap vs chance (0.25) and the same for energy/variance/L1.

### 4.4 Step A.1: does the ranking transfer to another dataset?

```powershell
python -m experiments.download_data --cifar100
python -m experiments.transfer_sensitivity --checkpoint checkpoints/train_split_adjsccq/best_val.pth --reference_ranking checkpoints/train_split_adjsccq/oracle_ranking.json --targets cifar100 --max_images 2000 --output_dir results/phase1/transfer
```

Decision printed per target: **STATIC** (Spearman >= 0.85 and top-4 preserved; hard-code the table,
0 B side-info, A.2/A.3 become ablations), **CONTENT-DEPENDENT** (<= 0.50; go to A.3) or
**PARTIAL**. The frozen model is never fine-tuned on the target set.
To check the *gain* transfers, run on CIFAR-100 with the CIFAR-10 ranking vs the native one:

```powershell
python -m experiments.eval_rs_uep --checkpoint checkpoints/train_split_adjsccq/best_val.pth --dataset cifar100 --split test --importance oracle,static:cifar100 --oracle_ranking_file checkpoints/train_split_adjsccq/oracle_ranking.json --rankings cifar100=results/phase1/transfer/ranking_cifar100.json --schemes three_tier_25_50_25 --erasure_rates 0.3,0.4,0.5 --include_no_rs --workers 8 --output_dir results/phase1/transfer_sweep --name transfer_sweep
```

(`static_cifar100` is an upper bound because it is derived from the evaluation set itself.)

### 4.5 Step A.2: cheap proxies

```powershell
python -m experiments.feature_saliency --checkpoint checkpoints/train_split_adjsccq/best_val.pth --reference_ranking checkpoints/train_split_adjsccq/oracle_ranking.json --max_images 2000 --output_dir results/phase1/saliency
```

Within-image Spearman of energy / variance / L1 (closed-form gradient saliency) / Taylor /
attention vs the true damage, against random and the static table. "Good" (roadmap): a proxy reaches
>= 80% of the oracle gain in the sweep of 4.8; "bad": Spearman < 0.30.

### 4.6 Step A.3: the sensitivity head

```powershell
python -m experiments.train_sensitivity_head --checkpoint checkpoints/train_split_adjsccq/best_val.pth --reference_ranking checkpoints/train_split_adjsccq/oracle_ranking.json --data_dir ./data --labels_dir results/phase1/head_labels --output_dir results/phase1/head --seeds 0,1,2,3,4
```

Stage 1 labels 40k + 10k images (16 decoder passes each; CPU-heavy but one-off and cached in
`head_labels/`); stage 2 trains 5 head seeds in seconds. If the head collapses to identical outputs
re-run with `--use_variance` (roadmap fix). It is judged against random, the static table, energy,
variance and attention on top-4 overlap, top-1 hit rate, within-image Spearman and *protection
regret* (dB of true damage in the true top-4 that the predicted top-4 misses). The MAC fraction
(< 2% required) is printed. Add `--eval_test` only for the final report (logged).

### 4.7 Compare every importance source in a real erasure sweep (validation)

```powershell
python -m experiments.eval_rs_uep --checkpoint checkpoints/train_split_adjsccq/best_val.pth --split val --importance attention,energy,variance,l1,taylor,predicted,oracle,oracle_perimage --oracle_ranking_file checkpoints/train_split_adjsccq/oracle_ranking.json --head_checkpoint results/phase1/head/sensitivity_head.pt --schemes three_tier_25_50_25 --erasure_rates 0.3,0.4,0.5 --seeds 0,1,2,3,4 --include_random_control --include_no_rs --workers 8 --output_dir results/phase1/saliency_sweep --name saliency_sweep
```

`INSIGHTS` inside the markdown gives each source's share of the oracle gain.
`oracle_perimage` is the true per-image upper bound (needs the image and decoder; not deployable);
`oracle` is the static training ranking.

### 4.8 Choose the tier allocation (validation only)

```powershell
python -m experiments.eval_rs_uep --checkpoint checkpoints/train_split_adjsccq/best_val.pth --split val --importance attention,oracle,predicted --oracle_ranking_file checkpoints/train_split_adjsccq/oracle_ranking.json --head_checkpoint results/phase1/head/sensitivity_head.pt --schemes all --seeds 0,1,2,3,4 --include_random_control --include_no_rs --workers 8 --output_dir results/phase1/val_sweep --name val_sweep
python -m experiments.select_scheme --sweep results/phase1/val_sweep/val_sweep.json --output results/phase1/val_sweep/selected_schemes.json
```

Writes the best scheme per loss rate and source, plus the `--schemes` string the final test run uses.

**Milestone 1 (roadmap):** predicted (or static) ranking reaches >= 90% of the oracle gain at 40% loss.

---

## 5. Phase 2: Track B (erasure-aware decoder)

```powershell
python -m experiments.train_erasure_robust --init_checkpoint checkpoints/train_split_adjsccq/best_val.pth --drop_unit channel --output_dir results/phase2/robust_channel
python -m experiments.train_erasure_robust --init_checkpoint checkpoints/train_split_adjsccq/best_val.pth --drop_unit packet  --output_dir results/phase2/robust_packet
python -m experiments.train_erasure_robust --init_checkpoint checkpoints/train_split_adjsccq/best_val.pth --drop_unit packet --inpaint --output_dir results/phase2/robust_inpaint
```

30 epochs, lr 1e-4, encoder frozen for 15 then joint. `channel` is the roadmap setting; `packet`
matches what a failed RS block really erases (16-symbol packets); the third adds the B.2 transformer.
The checkpoint is selected on a **validation robustness score** (mean PSNR over drop rates 0-0.4, both drop
units). The oracle ranking is **recomputed** for each fine-tuned model (the encoder moves when unfrozen).
Outputs: `adjsccq_erasure_robust.pth`, `oracle_ranking.json`, `metrics.{json,md}`, before/after
robustness figures. If clean PSNR falls by > 2 dB: `--p_max 0.3` or `--curriculum_epochs 20`.

Then sweep each on validation, select its own allocation, and compare in the final report:

```powershell
python run_pipeline.py --phase phase2 --execute --skip_existing --workers 8
```

**Milestone 2 (roadmap):** UEP gain at 20% loss improves by >= 2.5 dB (the low-loss penalty disappears).

---

## 6. Phase 3: Track C

### 6.1 Kodak at 768x512 (C.1)

```powershell
python -m experiments.download_data --kodak
python -m experiments.eval_kodak --checkpoint checkpoints/train_split_adjsccq/best_val.pth --oracle_ranking_file checkpoints/train_split_adjsccq/oracle_ranking.json --kodak_dir ./data/kodak --importance oracle,attention --seeds 0,1,2,3,4 --include_no_rs --save_images_rate 0.4 --workers 8 --output_dir results/phase3/kodak
```

128-byte packets, 3,072 data / 4,608 transmitted packets. **GF(256) limit:** a codeword has at most 255
symbols, so the roadmap's RS(1536, 768) cannot exist as one code; every tier is split into equal
sub-blocks of <= 255 packets with the same total data and parity (`coding/scaling.py`). Expect the
sharp "one big block corrects any loss <= 33%" threshold to soften at this scale; that is reported,
not hidden. Four Kodak images are portrait and are rotated to landscape (resolution unchanged).
It prints the clean PSNR; **if it is below ~24 dB** the CIFAR model does not generalise:

```powershell
python -m experiments.train_highres --init_checkpoint checkpoints/train_split_adjsccq/best_val.pth --train_dir D:/path/to/images --output_dir results/phase3/highres
python -m experiments.eval_kodak --checkpoint results/phase3/highres/best_val.pth --oracle_ranking_file results/phase3/highres/oracle_ranking.json --kodak_dir ./data/kodak --importance oracle,attention --include_no_rs --save_images_rate 0.4 --workers 8 --output_dir results/phase3/kodak_highres
```

(Any image folder works, e.g. COCO val2017; it must not contain Kodak images.)
Reconstructions at the chosen loss rate are written to `<output_dir>/images/<method>/<kodimXX.png>` for
the zoom-crop figure. Metrics: PSNR, SSIM, MS-SSIM, LPIPS (if installed).

### 6.2 Bandwidth ratio sweep (C.2)

```powershell
python -m experiments.train_adjsccq_train_split --k_over_n 0.0625   --seed 42 --split_seed 42 --skip_rs_sweep --data_dir ./data --output_dir results/phase3/kn_1_16
python -m experiments.train_adjsccq_train_split --k_over_n 0.083333 --seed 42 --split_seed 42 --skip_rs_sweep --data_dir ./data --output_dir results/phase3/kn_1_12
python -m experiments.train_adjsccq_train_split --k_over_n 0.333333 --seed 42 --split_seed 42 --skip_rs_sweep --data_dir ./data --output_dir results/phase3/kn_1_3
```

These have 6, 8 and 32 latent channels (16 at k/n = 1/6, reused). The evaluator rescales the tier
templates to any channel count while keeping channel alignment and 50% overhead.

**Milestone 3:** Figure 4 (Kodak crops) shows edge preservation; UEP gain >= +1.8 dB and ΔMS-SSIM >= +0.04 at 40% loss.

---

## 7. Phase 4: Track D

```powershell
python -m experiments.eval_burst --checkpoint checkpoints/train_split_adjsccq/best_val.pth --oracle_ranking_file checkpoints/train_split_adjsccq/oracle_ranking.json --split val --avg_losses 0.35,0.45 --burst_lengths 1,2,4,8,16 --seeds 0,1,2,3,4 --include_no_rs --workers 8 --output_dir results/phase4/burst
python -m experiments.eval_digital_baseline --split val --codecs jpeg2000,jpeg,webp --seeds 0,1,2,3,4 --workers 8 --output_dir results/phase4/digital
```

* Burst: Gilbert-Elliott (Good loses 1%, Bad 85%), long-run loss and mean burst length set independently;
  burst length 1 = iid reference. Every tiered scheme runs `sequential` and `round_robin` (`#rr`).
  The script prints the best gain per burst length and the gain from interleaving.
* Digital: each image is compressed to <= 1022 bytes (+2-byte length header = the same 1024-byte
  budget), RS(96, 64) protected, sent over the same masks; failure to decode shows mid-grey. Clean PSNR
  and stream sizes are in the metadata.

**Milestone 4:** interleaved UEP beats uniform by > 3 dB for burst length >= 4, and the codec baseline shows a cliff.

---

## 8. Final: test tables and the report (run once)

```powershell
python run_pipeline.py --phase final --execute --skip_existing --workers 8 --wandb
```

This runs, on the **official test split**, 5 channel seeds and the whole set:
the base-model sweep (schemes chosen on validation), the three robust variants, the k/n points, the
burst study and the digital baseline, then `make_report`, which writes to `results/report/`:
`fig3_primary`, `fig3_gain`, `fig_model_comparison`, `fig5_cliff`, `fig_kn_sweep`, `headline.md`, `INSIGHTS.md`.
Each figure is a vector PDF plus a 300-dpi PNG.

---

## 9. Output map

```
results/
  models/seed100, seed2024/          trained models + oracle rankings
  phase1/{multiseed,attention,transfer,saliency,head,head_labels,saliency_sweep,val_sweep}/
  phase2/{robust_channel,robust_packet,robust_inpaint}/   checkpoint, ranking, val_sweep/, test_sweep/
  phase3/{kodak,kodak_highres,kn_1_16,kn_1_12,kn_1_3}/
  phase4/{burst,digital}/
  final/                             test sweeps + burst/ + digital/
  report/                            figures, headline.md, INSIGHTS.md
  logs/                              per-step logs + pipeline_manifest.jsonl
  test_access_log.jsonl              every touch of the official test split
```

Every sweep folder holds `<name>.json` (rows, `seed_summary`, `paired_vs_baseline`, `insights`, `metadata`),
`<name>.md` (readable tables + insights), `<name>.per_image.npz` (PSNR of every image, for CDFs and win rates)
and `figures/` (PSNR curves, worst-case curves, block failures, paired gain with CI bands, gain heat-map, PSNR CDF).

---

## 10. Reproducibility checklist

1. `git status` clean before a final run (the JSON records `dirty_repo`).
2. `python -m pytest tests -q` passes.
3. Same command twice gives identical numbers; also with a different `--workers` (verify once with `--max_images 500`).
4. Checkpoint hashes in `metadata.checkpoint.sha256` match the files you ship; ranking/head hashes are in `metadata.files`.
5. `results/test_access_log.jsonl` shows the test split touched only by final steps (plus the one clean-PSNR convergence check per trained model).
6. Report **channel-seed** variability (`seed_summary`) separately from **model-seed** variability (Phase 1 multi-seed models); never merge them.
7. For the paper repository: scrub names, e-mail domains and W&B entity (double-blind), and ship `run_pipeline.py --list` output as the recipe.

---

## 11. Weights & Biases: what is logged

| Script | Logged |
|---|---|
| `train_adjsccq_train_split` | train MSE/PSNR/sigma per `log_every` steps, validation PSNR/SSIM per epoch, oracle table, checkpoints artifact |
| `train_erasure_robust` | per-epoch train MSE, validation clean/robust/channel-drop/packet-drop PSNR, curves, artifact |
| `train_sensitivity_head` | per-epoch loss and validation top-4/Spearman/R², per-seed and comparison tables, figures, head artifact |
| every sweep (`eval_rs_uep`, burst, digital, Kodak) | per-cell PSNR for every method as it finishes, `seed_summary`, paired gains, all figures, result artifact |
| analyses (multi-seed, attention, transfer, saliency, report) | tables, summary metrics, figures |

---

## 12. Decision guide (what the insights should make you do)

| Finding | Next action |
|---|---|
| A.1 says STATIC | Use the static table as the headline method (0 B side-info); keep the head as an ablation |
| A.1 says CONTENT-DEPENDENT and head beats the static table | Headline = predicted ranking; report its 16-byte cost |
| Proxy (energy/L1/Taylor) >= 80% of oracle gain | Mention as a zero-training alternative; skip the head in the main figure |
| Low-loss penalty persists after B.1 | Compare `packet` vs `channel` dropout, then try `--inpaint`, then a larger `--p_max` curriculum |
| Kodak clean PSNR < 24 dB | `train_highres`, then re-run Kodak |
| Burst gain < 3 dB at L >= 4 | Check `#rr` rows; if interleaving does not help, report the finding honestly |
| Index-aligned multi-seed Spearman is high | Surprising; investigate (should be ~0), do not celebrate |

---

## 13. Troubleshooting

* `ModuleNotFoundError: experiments`: run from the repository root with `python -m ...`.
* Sweeps are slow: raise `--workers`; use `--max_images 1000` for dry-runs; Reed-Solomon decoding (pure numpy) dominates, not the network.
* Windows + `--workers`: worker processes are spawned; keep scripts under `if __name__ == "__main__"` (they are).
* `Checkpoint not found`: pass the real path; `checkpoints/...` for the provided model, `results/...` for new ones.
* `wandb` login errors: omit `--wandb` or use `WANDB_MODE=offline`; the experiment never depends on it.
* Kodak out of memory: use `--max_images 8` first; reduce `--workers` (each worker holds the 24 images).
