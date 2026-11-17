# Speech runbook (Apple-silicon Mac, MPS): does sensitivity-aware UEP carry over to speech?

S1 repeats the image study on **LibriSpeech** with two 1-D models:

| Model | What it is |
|---|---|
| `adjsccq1d` | 1-D ADJSCC-Q, with an attention gate after every layer |
| `deepjsccq1d` | the same backbone without attention, the speech counterpart of G1 |

Both use the image models' latent layout: 16 channels × 64 steps = 1024 bytes per 1.024 s segment
(8 kbit/s). The Reed-Solomon coding, the packet masks and the side-info accounting are therefore
exactly those of the CIFAR study.

Design decisions behind every choice: [`docs/02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md`](../02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md), AD-14 to AD-21.
Status: **code written, nothing executed yet.**

**Open decisions** (block only the steps named):

* **AD-19 (blocks step 7, the test sweeps):** confirm that the image decision thresholds apply to speech.
* AD-14 / AD-15 / AD-18: widths 16–32, 40k training segments, 3 seeds per model. They are accepted
  as defaults and must be confirmed before step 4 (training).

All commands are for zsh/bash, run from the repository root. Setup is shared with
[`G1_RUNBOOK_MAC.md`](G1_DEEPJSCCQ_ATTENTION_ABLATION_RUNBOOK_MAC_MPS.md) §1 (venv, `PYTORCH_ENABLE_MPS_FALLBACK=1`, `W` = performance-core
count, `caffeinate -i`, `tmux`).

---

## 0. What runs where

| Step | What | Device | Output |
|---|---|---|---|
| 1 | extra packages | – | – |
| 2 | unit tests, dry run | CPU | terminal |
| 3 | download + segment LibriSpeech | CPU | `data/LibriSpeech/`, `data/librispeech_segments/` |
| 4 | smoke run (numbers never quoted) | MPS + CPU | `results/speech/smoke_*` |
| 5 | train 2 architectures × seeds 42, 100, 7 | **MPS** | `results/speech/<arch>/seed*/` |
| 6 | profiles, proxies (A.2), head (A.3), attention vs damage, validation sweep, selection | CPU | `results/speech/<arch>/...` |
| 7 | test sweeps (after AD-19 is confirmed) | CPU | `.../seed*/test_sweep/` |
| 8 | PESQ / STOI and listening examples | CPU | `.../seed42/perceptual/` |
| 9 | comparison under the frozen decision rule | CPU | `results/speech/compare/` |

---

## 1. Extra packages

```bash
source .venv/bin/activate
python -m pip install soundfile                 # required: reads LibriSpeech FLAC, writes WAV examples
python -m pip install pesq pystoi               # optional: step 8 (PESQ needs the Xcode command-line tools)
python -c "import soundfile; print(soundfile.__libsndfile_version__)"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export W=$(sysctl -n hw.perflevel0.physicalcpu)
```

Without `pesq` / `pystoi`, step 8 reports those metrics as unavailable; nothing is faked.

---

## 2. Verify (no data, no training)

```bash
python -m pytest tests -q
python -m pytest tests/test_speech.py -v
python run_pipeline.py --phase speech --train_device mps   # DRY RUN: prints every command, runs nothing
```

`tests/test_speech.py` builds a tiny fake LibriSpeech tree and checks:

* both speech models produce the 16 × 64 latent and differ only by the attention modules;
* the image metric path is bit-identical to before;
* SNR damage equals PSNR damage;
* a loss-free speech transmission is lossless;
* the segment cache is deterministic and drops silence;
* a one-epoch training run writes the image trainer's files;
* the `speech` phase is well formed.

**Gate:** all tests pass. If the data tests are *skipped*, `soundfile` is missing; install it and rerun.

---

## 3. Download and segment LibriSpeech (once per machine)

```bash
caffeinate -i python run_pipeline.py --only speech_data --execute
#   = python -m experiments.prepare_speech_data --download --build --data_dir ./data
```

This downloads about 7 GB (train-clean-100 6.3 GB, dev-clean 0.3 GB, test-clean 0.3 GB) from OpenSLR,
then builds the caches (about 1.6 GB of int16). Expect tens of minutes, mostly the download and
the first scan of about 28,500 FLAC files.

**Check** `data/librispeech_segments/{train,val,test}.json`:

| Split | `num_segments` | `speakers` |
|---|---|---|
| train | 40000 | ≈ 251 |
| val | 4000 | 40 |
| test | 4000 | 40 |

The terminal prints each array's SHA-256. Record the three hashes in `docs/04_research_progress_log/RESEARCH_PROGRESS_LOG_PHASE0_TO_G1_AND_S1.md`. Another
machine that builds the cache must get the same hashes; otherwise the results are not comparable.

The archives in `data/LibriSpeech_archives/` can be deleted afterwards to free 7 GB.

---

## 4. Smoke test on MPS (numbers are never quoted)

```bash
caffeinate -i python run_pipeline.py --only speech_smoke_train --train_device mps --execute
```

This trains 2 epochs of `adjsccq1d` on 2,000 segments and computes an oracle on 500. Check:

1. `results/speech/smoke_train/metrics.json`: `"arch": "adjsccq1d"`, `"train_device": "mps"`,
   `"quality_metric": "per-segment SNR ..."`.
2. The log shows `validation SNR ... dB | SI-SDR ... dB` per epoch. **Note the epoch time:** a full
   epoch is 20× more segments, so (time per epoch × 20 × 60) is the time per real model.
3. Then smoke-test one evaluation cell (also measures sweep speed):

```bash
caffeinate -i python -m experiments.eval_rs_uep --dataset librispeech \
  --checkpoint results/speech/smoke_train/best_val.pth \
  --oracle_ranking_file results/speech/smoke_train/oracle_ranking.json \
  --data_dir ./data --split val --importance oracle,energy,attention --schemes three_tier_25_50_25 \
  --erasure_rates 0.0,0.4 --seeds 0,1 --max_images 200 --include_no_rs --include_random_control \
  --workers $W --output_dir results/speech/smoke_eval --name smoke
```

**Gate:** in `results/speech/smoke_eval/smoke.md` the columns read "SNR (dB)" and "SI-SDR (dB)", and
at loss 0.0 every method equals `uniform` (a lossless round trip).

---

## 5. Train the six speech models (MPS)

```bash
caffeinate -i python run_pipeline.py --only \
speech_train_adjsccq1d_seed42,speech_train_adjsccq1d_seed100,speech_train_adjsccq1d_seed7,\
speech_train_deepjsccq1d_seed42,speech_train_deepjsccq1d_seed100,speech_train_deepjsccq1d_seed7 \
  --train_device mps --skip_existing --execute
```

Equivalent direct command for one model:

```bash
caffeinate -i python -m experiments.train_speech_jscc --config configs/speech_deepjsccq1d.yaml \
  --arch deepjsccq1d --seed 42 --device mps --data_dir ./data --output_dir results/speech/deepjsccq1d/seed42
```

**Check each** `results/speech/<arch>/seed*/metrics.json`:

| Field | Expect | If not |
|---|---|---|
| `best_epoch` | ≥ 50 (the convergence rule, as for images) | Not converged. Keep it (the comparison excludes and names it) and add seed 13 for that architecture by editing `SPEECH_SEEDS` in `run_pipeline.py`; record it in PROGRESS_LOG |
| `clean_validation.psnr_db_mean_per_image` | a positive SNR (no reference value yet) | Note it. Very low clean SNR (a few dB) means the 16–32-wide model is too small; raise it with the owners (AD-14) before evaluating |
| `clean_validation.si_sdr_db_mean` | close to the SNR | A large gap means the output level is off |

---

## 6. Analysis on dev-clean (CPU)

```bash
caffeinate -i python run_pipeline.py --only \
speech_multiseed_adjsccq1d,speech_multiseed_deepjsccq1d,\
speech_saliency_adjsccq1d,speech_saliency_deepjsccq1d,\
speech_head_adjsccq1d,speech_head_deepjsccq1d,speech_attention_vs_damage,\
speech_val_sweep_adjsccq1d,speech_val_sweep_deepjsccq1d,speech_select_adjsccq1d,speech_select_deepjsccq1d \
  --workers $W --skip_existing --execute
```

| Output | Look at | Image reference |
|---|---|---|
| `<arch>/multiseed/multiseed.md` | top-4 share, worst-channel damage. Its "PSNR" column is SNR here | ADJSCC-Q: 45–53%, 8.7–17.1 dB |
| `<arch>/seed42/saliency/saliency.md` | does energy / L1 still track damage? | Spearman +0.84 |
| `adjsccq1d/seed42/attention/attention_vs_damage.json` | does attention fail in speech too? | within-image Spearman −0.015 |
| `<arch>/seed42/head/head_metrics.md` | head top-4 overlap vs static table | 0.92 vs 0.88 |
| `<arch>/seed42/val_sweep/selected_schemes.json` | the schemes the test sweeps will use | |

---

## 7. STOP: confirm the speech decision rule (AD-19), then run the test sweeps

The proposal is the image thresholds (AD-09), unchanged:

* C1: top-4 share ≥ 0.40;
* C2: oracle gain at 40% loss ≥ +1.5 dB with CI > 0;
* C3: random < 0.5 × oracle;
* FLAT: top-4 < 0.30 and oracle within 0.5 dB of random.

They carry over because damage and gains are dB of MSE ratios in both modalities (AD-16). Once
the owners confirm, mark AD-19 accepted in `docs/02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md` and commit **before** running:

```bash
caffeinate -i python run_pipeline.py --only \
speech_final_adjsccq1d_seed42,speech_final_adjsccq1d_seed100,speech_final_adjsccq1d_seed7,\
speech_final_deepjsccq1d_seed42,speech_final_deepjsccq1d_seed100,speech_final_deepjsccq1d_seed7 \
  --workers $W --skip_existing --execute
```

These are 5 channel seeds × all 4,000 test-clean segments, with a random control and `no_rs`. Every run
is appended to `results/test_access_log.jsonl`. **Run each once.**

---

## 8. Perceptual check and listening examples

```bash
caffeinate -i python run_pipeline.py --only speech_perceptual_adjsccq1d,speech_perceptual_deepjsccq1d --execute
```

Output in `results/speech/<arch>/seed42/perceptual/`:

* `perceptual.md`: SNR, SI-SDR, PESQ-WB and STOI for `no_rs`, `uniform` and the validation-selected
  scheme per source, at loss 0, 0.3, 0.35, 0.4 and 0.5. It uses 500 test segments and channel seed 0.
* `examples/*.wav`: the originals and every method at every loss rate for 3 segments, for listening
  (and for a supplementary page).

Check that PESQ and STOI order the methods the same way SNR does at high loss. If they disagree,
report it; do not drop it.

---

## 9. Compare under the frozen rule

```bash
python run_pipeline.py --only speech_compare --execute
```

`results/speech/compare/compare.md` gives the per-model and per-family tables and the verdict for
`adjsccq1d` (reference) and `deepjsccq1d`. Reading it together with the image G1 comparison gives the
2 × 2 picture:

| | with attention | without attention |
|---|---|---|
| **images** | ADJSCC-Q (published) | DeepJSCC-Q (G1) |
| **speech** | `adjsccq1d` (S1) | `deepjsccq1d` (S1) |

| Speech verdict | Meaning for the paper |
|---|---|
| both **criteria met** | Sensitivity-aware UEP carries over to speech, independent of attention: a modality-level claim |
| `adjsccq1d` met, `deepjsccq1d` **no exploitable structure** | In speech the imbalance comes from the attention gates |
| **intermediate** | Report the gain-vs-concentration relation (scatter plot) |

Absolute SNR (speech) and PSNR (images) are **not** comparable; gains and damages in dB are (AD-16).

---

## 10. Bring the results back

Commit JSON, Markdown, figures and logs under `results/speech/`. Keep the WAV examples small (3 segments
is the default). Leave out `head_labels/`, `*.per_image.npz`, `data/LibriSpeech*` and
`data/librispeech_segments/`. Append a dated "S1 results" section to `docs/04_research_progress_log/RESEARCH_PROGRESS_LOG_PHASE0_TO_G1_AND_S1.md` with:

* the three cache hashes;
* the per-family table and the verdicts;
* the PESQ/STOI table;
* any non-converged run;
* any deviation from this runbook.

---

## 11. Time budget (estimate; replace with the smoke-test measurements)

| Item | Estimate on an M2 | Based on |
|---|---|---|
| Download + cache build | 20–60 min | network speed; one scan of ~28.5k FLAC files |
| One speech model (60 epochs × 40k segments + oracle) | ~2.5 h | ~1.7× the image model's compute; 87 min for ADJSCC-Q |
| Six models | ~15 h | |
| A.3 labels (44k segments × 17 decodes, CPU), per architecture | ~0.5–1.5 h | |
| Validation sweep per architecture (3 seeds × 4k × 8 rates × all schemes) | a few hours | smoke cell time × cells |
| Six test sweeps (5 seeds × 4k × 8 rates) | a few hours each | smoke cell time × cells |
| Perceptual check (500 segments, PESQ) | < 1 h per architecture | |

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no .flac files under data/LibriSpeech/...` | download step not run or interrupted | rerun step 3; a partial archive `*.part` is overwritten |
| `ModuleNotFoundError: soundfile` | step 1 skipped | `pip install soundfile` |
| `dataset 'librispeech' does not match the checkpoint's modality` | an image checkpoint was passed with `--dataset librispeech` (or the reverse) | use the matching checkpoint |
| `importance source 'attention' needs a bottleneck attention gate` | `attention` passed for `deepjsccq1d` | expected; drop it |
| `pesq` raises `NoUtterancesError` | a segment PESQ considers silent | counted in the "PESQ rejected" column; expected to be rare after the −40 dBFS filter |
| MPS op not implemented | missing MPS kernel | `export PYTORCH_ENABLE_MPS_FALLBACK=1` (set automatically with `--train_device mps`) |
| out of memory in a sweep | too many workers × 260 MB each | lower `--workers` |
| training stalls overnight | Mac slept | `caffeinate -i`, power adapter, `tmux` |
