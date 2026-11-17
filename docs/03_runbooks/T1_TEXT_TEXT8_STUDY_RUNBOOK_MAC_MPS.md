# Text runbook (T1): does sensitivity-aware UEP carry over to text?

T1 repeats the speech study (S1) on **text8** with two character-level models:

| Model | What it is | Counterpart |
|---|---|---|
| `adjsccq_text` | 1-D ADJSCC-Q on characters, with an attention gate after every layer | ADJSCC-Q (images), `adjsccq1d` (speech) |
| `deepjsccq_text` | the same backbone without attention | DeepJSCC-Q (G1), `deepjsccq1d` (speech) |

Both use the image and speech latent layout: 16 channels × 64 steps = 1,024 bytes per window of
2,048 characters (4 bits per character). The Reed-Solomon coding, the packet masks and the
side-info accounting are therefore exactly those of the CIFAR and LibriSpeech studies.

**The one thing that is different: the quality metric.** Text quality is the per-window **character
accuracy in %**, stored in the same `psnr_db_*` fields. In every shared table and figure (multiseed,
sweeps, comparison) a number labelled "dB" therefore means **percentage points (pp) of character
accuracy** for text. Example: "LOO damage 23.4 dB" means "losing this channel costs 23.4 pp of
character accuracy". Word accuracy (%) replaces SSIM / SI-SDR as the secondary column. See AD-24.

Design decisions behind every choice: [`ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md`](../02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md), AD-22 to AD-30.
Status: **code written, nothing executed yet** (not even the unit tests).

**Open decisions** (each blocks only the step named):

* **AD-28 (blocks step 7, the test sweeps):** confirm the text decision thresholds (C2 ≥ +2.0 pp, FLAT margin 0.5 pp).
* AD-22 / AD-23 / AD-25 / AD-26: widths 32–64, 2,048-character windows, 40k training windows, cross-entropy
  loss, 3 seeds per model.
  They are accepted as defaults and must be confirmed before step 5 (training).

Commands are given for **macOS (zsh/bash, MPS)** first, as for the speech study, then for **Windows
(PowerShell)**. Run everything from the repository root.

---

## 0. What runs where

| Step | What | Device | Output |
|---|---|---|---|
| 1 | environment | – | – |
| 2 | unit tests, dry run | CPU | terminal |
| 3 | download text8 + build the window caches | CPU | `data/text8/`, `data/text8_windows/` |
| 4 | smoke run (numbers never quoted) | MPS / CUDA + CPU | `results/text/smoke_*` |
| 5 | train 2 architectures × seeds 42, 100, 7 | **MPS / CUDA** | `results/text/<arch>/seed*/` |
| 6 | damage profiles, proxies (A.2), head (A.3), attention vs damage, validation sweep, selection | CPU | `results/text/<arch>/...` |
| 7 | test sweeps (after AD-28 is confirmed) | CPU | `.../seed*/test_sweep/` |
| 8 | word accuracy, bits/char and readable examples | CPU | `.../seed42/examples/` |
| 9 | comparison under the frozen decision rule | CPU | `results/text/compare/` |

The fastest route to the speech-style headline (clean quality + LOO damage + top-4 share for all six
models) is steps 1–5 followed by the two `text_multiseed_*` steps of step 6. Everything after that
adds the UEP gains (C2, C3).

---

## 1. Environment

No new packages are needed: text8 is plain ASCII, read with NumPy.

**macOS:**

```bash
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export W=$(sysctl -n hw.perflevel0.physicalcpu)    # CPU workers for the sweeps
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
$W = [int]([Environment]::ProcessorCount / 2)       # physical cores is a good default
python -c "import torch; print(torch.cuda.is_available())"   # True -> use --train_device cuda, else cpu
```

On Windows, replace `caffeinate -i` in the commands below by nothing (keep the PC from sleeping in
the power settings), `--train_device mps` / `--device mps` by `cuda` (or `cpu`), and the trailing `\`
line continuations by writing the command on one line (or ending lines with a backtick `` ` ``).
`$W` works unchanged in PowerShell.

---

## 2. Verify (no data, no training)

```bash
python -m pytest tests/test_text.py -v
python -m pytest tests -q                                   # everything, incl. speech and images
python run_pipeline.py --phase text --train_device mps      # DRY RUN: prints every command, runs nothing
```

`tests/test_text.py` builds a tiny synthetic corpus in the text8 alphabet and checks:

* both text models produce the 16 × 64 latent and per-character logits, and differ only by the attention modules;
* the image and speech metric paths are bit-identical to before;
* character accuracy, word accuracy and bits/char are right on hand-made strings;
* the Taylor importance source uses cross-entropy for text;
* a loss-free text transmission is lossless, and sweep rows carry word accuracy;
* the window cache is deterministic and respects the train / val / test character ranges;
* a one-epoch training run writes the image trainer's files;
* the `text` phase is well formed.

**Gate:** all tests pass (none should be skipped: no optional packages are involved).

---

## 3. Download text8 and build the window caches (once per machine)

```bash
python run_pipeline.py --only text_data --execute
#   = python -m experiments.prepare_text_data --download --build --data_dir ./data
```

This downloads `text8.zip` (31 MB) from mattmahoney.net, extracts the 100 MB file `data/text8/text8`,
checks that it has exactly 100,000,000 characters in the 27-symbol alphabet, and writes the caches
(about 90 MB of uint8). Expect under a minute after the download.

**Check** `data/text8_windows/{train,val,test}.json`:

| Split | characters | `windows_available` | `num_segments` |
|---|---|---|---|
| train | 0 – 90M | 43,945 | **40,000** |
| val | 90M – 95M | 2,441 | **2,441** |
| test | 95M – 100M | 2,441 | **2,441** |

The terminal prints each array's SHA-256, and every manifest holds `source_sha256` (the text8 file).
Record all four hashes in [`RESEARCH_PROGRESS_LOG`](../04_research_progress_log/RESEARCH_PROGRESS_LOG_PHASE0_TO_G1_AND_S1.md). A second machine must get the same hashes, otherwise
the results are not comparable. `data/text8/text8.zip` can be deleted afterwards.

---

## 4. Smoke test (numbers are never quoted)

```bash
caffeinate -i python run_pipeline.py --only text_smoke_train --train_device mps --execute
```

This trains 2 epochs of `adjsccq_text` on 2,000 windows and computes an oracle on 500. Check:

1. `results/text/smoke_train/metrics.json`: `"arch": "adjsccq_text"`, `"train_device": "mps"` (or `cuda`),
   `"quality_metric": "per-window character accuracy (%) ..."`, `"recipe": {... "loss": "cross_entropy"}`.
2. The log (`results/logs/text_smoke_train.log`) shows
   `validation char acc ...% | word acc ...% | ... bits/char` per epoch. **Note the epoch time:** a full
   epoch has 20× more windows, so (time per epoch × 20 × 60) is the time per real model.
3. Smoke-test one evaluation cell (this also measures sweep speed):

```bash
caffeinate -i python -m experiments.eval_rs_uep --dataset text8 \
  --checkpoint results/text/smoke_train/best_val.pth \
  --oracle_ranking_file results/text/smoke_train/oracle_ranking.json \
  --data_dir ./data --split val --importance oracle,energy,attention --schemes three_tier_25_50_25 \
  --erasure_rates 0.0,0.4 --seeds 0,1 --max_images 200 --include_no_rs --include_random_control \
  --workers $W --output_dir results/text/smoke_eval --name smoke
```

**Gate:** in `results/text/smoke_eval/smoke.md` the columns read "Char acc (%)" and "Word acc (%)", and
at loss 0.0 every method equals `uniform` (a lossless round trip).

---

## 5. Train the six text models

```bash
caffeinate -i python run_pipeline.py --only \
text_train_adjsccq_text_seed42,text_train_adjsccq_text_seed100,text_train_adjsccq_text_seed7,\
text_train_deepjsccq_text_seed42,text_train_deepjsccq_text_seed100,text_train_deepjsccq_text_seed7 \
  --train_device mps --skip_existing --execute
```

Equivalent direct command for one model:

```bash
caffeinate -i python -m experiments.train_text_jscc --config configs/text_deepjsccq.yaml \
  --arch deepjsccq_text --seed 42 --device mps --data_dir ./data --output_dir results/text/deepjsccq_text/seed42
```

Windows (one line, CUDA):

```powershell
python run_pipeline.py --only text_train_adjsccq_text_seed42,text_train_adjsccq_text_seed100,text_train_adjsccq_text_seed7,text_train_deepjsccq_text_seed42,text_train_deepjsccq_text_seed100,text_train_deepjsccq_text_seed7 --train_device cuda --skip_existing --execute
```

All six models must be trained on the **same machine and device** (no hardware anchor is planned
within T1, as in S1; AD-26).

**Check each** `results/text/<arch>/seed*/metrics.json`:

| Field | Expect | If not |
|---|---|---|
| `best_epoch` | ≥ 50 (the convergence rule, as for images and speech) | Not converged. Keep it (the comparison excludes and names it), add seed 13 for that architecture by editing `TEXT_SEEDS` in `run_pipeline.py`, and record it in PROGRESS_LOG (as for `deepjsccq1d` seed 100 in S1) |
| `clean_validation.char_acc_mean` | no reference value yet; a 27-symbol guess gets ≈ 18%, so anything near that means it did not learn. Our estimate is 85–99% | Below ~70%, the 32–64-wide model is too small for 4 bits/char. Raise it with the owners (AD-22) before evaluating: wider widths, or 1,024-character windows |
| `clean_validation.word_acc_mean` | below the character accuracy (a word needs every letter right) | — |
| `clean_validation.bits_per_char` | falls over epochs; below 4.75 (= log2 27, no knowledge) | Above 4.75 means the decoder is not confident; check the log |
| `codebook_levels_used` | most of 256 | Very few levels means the quantiser collapsed; report it |

---

## 6. Analysis on the validation split (CPU)

```bash
caffeinate -i python run_pipeline.py --only \
text_multiseed_adjsccq_text,text_multiseed_deepjsccq_text,\
text_saliency_adjsccq_text,text_saliency_deepjsccq_text,\
text_head_adjsccq_text,text_head_deepjsccq_text,text_attention_vs_damage,\
text_val_sweep_adjsccq_text,text_val_sweep_deepjsccq_text,text_select_adjsccq_text,text_select_deepjsccq_text \
  --workers $W --skip_existing --execute
```

| Output | Look at | Image / speech reference |
|---|---|---|
| `<arch>/multiseed/multiseed.md` | top-4 share, worst-channel damage (its "PSNR"/"dB" columns are char-accuracy pp) | top-4: images 45–53%, speech 43.0% / 46.6% |
| `<arch>/seed42/saliency/saliency.md` | do energy / L1 still track damage? | images Spearman +0.84 |
| `adjsccq_text/seed42/attention/attention_vs_damage.json` | does attention fail on text too? | images: within-image Spearman −0.015 |
| `<arch>/seed42/head/head_metrics.md` | head top-4 overlap vs static table | images 0.92 vs 0.88 |
| `<arch>/seed42/val_sweep/selected_schemes.json` | the schemes the test sweeps will use | |

The multiseed band labels (">10 dB catastrophic", "<2 dB low") were written for dB. For text read them as
pp of character accuracy; the numbers are right, only the unit in the label is not.

---

## 7. STOP: confirm the text decision rule (AD-28), then run the test sweeps

The proposal:

* C1: top-4 share ≥ 0.40 (unit-free; identical to images and speech);
* C2: oracle gain at 40% loss ≥ **+2.0 pp** of character accuracy, with CI > 0;
* C3: random < 0.5 × oracle (unit-free);
* FLAT: top-4 < 0.30 and oracle within **0.5 pp** of random.

Unlike speech, the dB thresholds cannot be copied, because accuracy differences are not dB (AD-24).
Once the owners confirm (or change) the two pp values, mark AD-28 accepted in
[`ARCHITECTURE_DECISION_RECORD`](../02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md), update `TEXT_DECISION_RULE` in `run_pipeline.py` if they changed, and
commit **before** running:

```bash
caffeinate -i python run_pipeline.py --only \
text_final_adjsccq_text_seed42,text_final_adjsccq_text_seed100,text_final_adjsccq_text_seed7,\
text_final_deepjsccq_text_seed42,text_final_deepjsccq_text_seed100,text_final_deepjsccq_text_seed7 \
  --workers $W --skip_existing --execute
```

These are 5 channel seeds × all 2,441 test windows, with a random control and `no_rs`. Every run is
appended to `results/test_access_log.jsonl`. **Run each once.**

---

## 8. Readable check: word accuracy, bits/char and example passages

```bash
caffeinate -i python run_pipeline.py --only text_examples_adjsccq_text,text_examples_deepjsccq_text --execute
```

Output in `results/text/<arch>/seed42/examples/`:

* `text_examples.md`: character accuracy, word accuracy and bits/char for `no_rs`, `uniform` and the
  validation-selected scheme per source, at loss 0, 0.3, 0.35, 0.4 and 0.5 (500 test windows, channel seed 0);
* `examples.md`: the first 240 characters of 3 windows, original and every method at every loss rate,
  with every **wrong character in UPPER CASE** (a wrong space as `_`). This is the text counterpart of the
  speech WAV examples and makes a good figure or supplementary page.

Check that word accuracy orders the methods the same way character accuracy does at high loss. If
they disagree, report it; do not drop it.

---

## 9. Compare under the frozen rule

```bash
python run_pipeline.py --only text_compare --execute
```

`results/text/compare/compare.md` gives the per-model and per-family tables and the verdict for
`adjsccq_text` (reference) and `deepjsccq_text`. Together with G1 and S1 this completes a 3 × 2 picture:

| | with attention | without attention |
|---|---|---|
| **images** (2-D, continuous) | ADJSCC-Q (published) | DeepJSCC-Q (G1) |
| **speech** (1-D, continuous) | `adjsccq1d` (S1) | `deepjsccq1d` (S1) |
| **text** (1-D, discrete symbols) | `adjsccq_text` (T1) | `deepjsccq_text` (T1) |

| Text verdict | Meaning for the paper |
|---|---|
| both **criteria met** | Sensitivity concentration and UEP gains hold for discrete symbols too: the claim covers continuous and categorical sources |
| `adjsccq_text` met, `deepjsccq_text` **no exploitable structure** | For text the imbalance needs attention; unlike images and speech |
| **intermediate** | Report the gain-vs-concentration relation (scatter plot) |

Top-4 share and the C3 ratio are unit-free and **directly comparable** across all three modalities.
Absolute quality and damages in pp (text) vs dB (images, speech) are **not** comparable.

---

## 10. Bring the results back

Commit JSON, Markdown, figures and logs under `results/text/`. Leave out `head_labels/`,
`*.per_image.npz`, `data/text8/` and `data/text8_windows/`. Then:

* fill in [`T1_TEXT_TEXT8_STUDY_RESULTS_AND_INTERPRETATION_TEMPLATE.md`](../05_results_and_interpretation/T1_TEXT_TEXT8_STUDY_RESULTS_AND_INTERPRETATION_TEMPLATE.md) (a template with the same
  sections as the speech results write-up);
* append a dated "T1 results" section to [`RESEARCH_PROGRESS_LOG`](../04_research_progress_log/RESEARCH_PROGRESS_LOG_PHASE0_TO_G1_AND_S1.md) with the four hashes, the per-family table and
  verdicts, the example table, any non-converged run and any deviation from this runbook.

---

## 11. Time budget (estimate; replace with the smoke-test measurements)

| Item | Estimate on an Apple M-series laptop | Based on |
|---|---|---|
| Download + cache build | 1–3 min | 31 MB download, one pass over 100 MB |
| One text model (60 epochs × 40k windows + oracle) | ~2–3 h | ~1.3× the speech model's MACs per item (≈ 21M vs 16M); speech took 1h40–2h10 on a MacBook Air |
| Six models | ~12–18 h | |
| A.3 labels (42k windows × 17 decodes, CPU), per architecture | ~0.5–1.5 h | |
| Validation sweep per architecture (3 seeds × 2,441 × 8 rates × all schemes) | 1–3 h | fewer items than speech (2,441 vs 4,000) |
| Six test sweeps (5 seeds × 2,441 × 8 rates) | ~1–2 h each | smoke cell time × cells |
| Examples (500 windows) | minutes | no slow perceptual metric |

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `data/text8/text8 missing` | download step not run | rerun step 3 |
| `expected 100000000 characters` | truncated download | delete `data/text8/` and rerun step 3 |
| `characters outside the text8 alphabet` | a different file was placed at `data/text8/text8` | use the original text8 |
| `dataset 'text8' does not match the checkpoint's modality` | an image / speech checkpoint was passed with `--dataset text8` (or the reverse) | use the matching checkpoint |
| `importance source 'attention' needs a bottleneck attention gate` | `attention` passed for `deepjsccq_text` | expected; drop it |
| `arch ... is not a text model` | `train_text_jscc` given a speech or image config | use `configs/text_*.yaml` |
| MPS op not implemented | missing MPS kernel | `export PYTORCH_ENABLE_MPS_FALLBACK=1` (set automatically with `--train_device mps`) |
| out of memory in a sweep | too many workers (logits are 27 × 2,048 per window) | lower `--workers` or `--batch_size` |
| training stalls overnight | the computer slept | `caffeinate -i` / power settings, power adapter, `tmux` |
