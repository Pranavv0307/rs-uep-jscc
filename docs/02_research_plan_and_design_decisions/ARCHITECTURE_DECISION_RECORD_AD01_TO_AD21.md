# Architecture decision record

Living record of the design decisions behind the code, newest last. Each entry states the
decision, why it was taken, what was rejected, and what it costs. An entry marked
**PENDING** needs the owners' confirmation before the step it affects is run. To change a
decision, add a new entry that supersedes it; do not edit history.

Scope so far:
* **G0:** make the pipeline model-agnostic.
* **G1:** DeepJSCC-Q, the attention ablation. How to run: [`G1_RUNBOOK_MAC.md`](../03_runbooks/G1_DEEPJSCCQ_ATTENTION_ABLATION_RUNBOOK_MAC_MPS.md).
* **S1:** the speech study, LibriSpeech with and without attention (AD-14 to AD-21). How to run:
  [`S1_SPEECH_LIBRISPEECH_STUDY_RUNBOOK_MAC_MPS.md`](../03_runbooks/S1_SPEECH_LIBRISPEECH_STUDY_RUNBOOK_MAC_MPS.md).
* **T1:** the text study, text8 characters with and without attention (AD-22 to AD-30). How to run:
  [`T1_TEXT_TEXT8_STUDY_RUNBOOK_MAC_MPS.md`](../03_runbooks/T1_TEXT_TEXT8_STUDY_RUNBOOK_MAC_MPS.md).

| ID | Decision | Status |
|---|---|---|
| AD-01 | Model-agnostic pipeline through a registry plus a shared call signature | Accepted (G0) |
| AD-02 | DeepJSCC-Q subclasses the frozen `models/deepjscc.py` classes | Accepted (G1) |
| AD-03 | Same latent budget and quantiser as ADJSCC-Q | Accepted (G1) |
| AD-04 | `snr_db` accepted and ignored by models without SNR-conditioned layers | Accepted (G0) |
| AD-05 | Training recipe v1 unchanged for DeepJSCC-Q | Accepted, confirmed by owners 2026-11-06 |
| AD-06 | Init seeds 42, 100, 7, 13 (+21 spare); convergence rule unchanged | Accepted, confirmed by owners 2026-11-06 |
| AD-07 | Hardware: G1 on Apple MPS; same-hardware ADJSCC-Q anchor is run | Accepted, confirmed by owners 2026-11-06 |
| AD-08 | Device handling: explicit `--device mps`, `auto` unchanged, CPU evaluation | Accepted (G1) |
| AD-09 | G1 evaluation protocol and pre-registered decision rule | Accepted, **frozen** 2026-11-06 (before any G1 test result) |
| AD-10 | `g1` is an extension phase of `run_pipeline.py`, not part of `--phase all` | Accepted (G1) |
| AD-11 | Attention-dependent code paths degrade explicitly; legacy scripts stay ADJSCC-only | Accepted (G0) |
| AD-12 | Keep the training entry point's name | Accepted (G0) |
| AD-13 | Out of scope for G0/G1 | Accepted |
| AD-14 | Speech models: 1-D port of the image backbone, with and without attention | Accepted (S1), **confirm widths** |
| AD-15 | Speech data: LibriSpeech clean subsets, speaker-disjoint, 1.024 s segments | Accepted (S1), **confirm size** |
| AD-16 | Speech quality metric: per-segment SNR in the `psnr_db_*` fields; SI-SDR instead of SSIM | Accepted (S1) |
| AD-17 | PESQ / STOI on a fixed subset only, plus listening examples | Accepted (S1) |
| AD-18 | Speech protocol: recipe v1, 3 seeds per architecture, image protocol repeated | Accepted (S1), **confirm seeds** |
| AD-19 | Speech decision rule: the AD-09 thresholds, frozen before the speech test sweeps | **PENDING** |
| AD-20 | Shared code made modality-aware; image path bit-identical | Accepted (S1) |
| AD-21 | Out of scope for S1 | Accepted |
| AD-22 | Text models: character-level port of the speech backbone, with and without attention; supersedes AD-21 "Text" | Accepted (T1), **confirm widths** |
| AD-23 | Text data: text8, standard 90M/5M/5M split, 2,048-character windows | Accepted (T1), **confirm size** |
| AD-24 | Text quality metric: character accuracy (%) in the `psnr_db_*` fields; word accuracy instead of SSIM | Accepted (T1) |
| AD-25 | Text training loss: cross-entropy instead of MSE (the only recipe change) | Accepted (T1), **confirm** |
| AD-26 | Text protocol: recipe v1, 3 seeds per architecture, speech protocol repeated | Accepted (T1), **confirm seeds** |
| AD-27 | Readable examples and bits/char instead of PESQ / STOI | Accepted (T1) |
| AD-28 | Text decision rule: AD-09 structure, thresholds re-expressed in percentage points | **PENDING** |
| AD-29 | Shared code: integer inputs mean text; image and speech paths bit-identical | Accepted (T1) |
| AD-30 | Out of scope for T1 | Accepted |

---

## AD-01: model-agnostic pipeline through a registry plus a shared call signature (G0)

**Date:** 2026-11-06. **Status:** accepted.

**Context.** Before G0 the pipeline could only build ADJSCC-Q: `experiments/model_io.py`
constructed `ADJSCCQEncoder` directly, and the channel count was read from model
internals (`encoder.encoder.af5.num_channels`) in `model_io.py` and the training script.
The coding layer (`coding/`) was already generic: it only sees `(B, C·D)` byte indices and a
`(B, C)` channel order.

**Decision.**
* `models/registry.py` maps `config["model"]["arch"]` to `(encoder_class, decoder_class)`.
  An absent key means `adjsccq`, so every pre-G0 config, checkpoint and Kaggle launcher is
  unchanged.
* Every architecture implements the ADJSCC-Q call signature
  (`encoder(x, snr_db, return_attn, return_indices)`, `encoder.encoder(x, snr_db, return_attn)`,
  `encoder.quantizer`, `encoder.k`, `encoder.calibrate_quantizer`, `decoder(z, snr_db, spatial_hw=...)`)
  plus two read-only attributes: `num_channels` and `has_attention`.
* `ModelBundle` gains `arch` and `has_attention`; `channels` now reads `encoder.num_channels`.
  Scripts use the bundle, not model internals.
* `build_jscc` builds the encoder before the decoder with the same arguments the training
  script always used, so ADJSCC-Q initialisation for a given seed is **bit-identical** to the
  pre-G0 code (`tests/test_deepjsccq.py::test_registry_builds_adjsccq_exactly_like_pre_g0_code`).

**Alternatives rejected.**
* *An abstract `JSCCModel` base class with new `encode()` / `decode()` methods.* Cleaner on
  paper, but it means editing ADJSCC-Q and every one of the ~15 scripts that call the models,
  with a real risk of changing published numbers.
* *Separate copies of the scripts per architecture.* Duplicates the integrity rules (shared
  masks, side-info accounting, test logging) and invites protocol drift between families.

**Consequences.** ADJSCC-Q only gains `has_attention = True` and a `num_channels` property;
its weights and state-dict keys are untouched. A new architecture must honour the signature,
and `tests/test_deepjsccq.py` checks it.

## AD-02: DeepJSCC-Q subclasses the frozen `models/deepjscc.py` classes (G1)

**Decision.** `models/deepjsccq.py` defines `DeepJSCCBackbone(DeepJSCCEncoder)`,
`DeepJSCCQEncoder` (backbone plus the same `ScalarSoftToHardQuantizer`) and
`DeepJSCCQDecoder(DeepJSCCDecoder)`. `models/deepjscc.py` is not edited.

**Why.** The repo's convention is that the Week-2/3 DeepJSCC classes are a frozen reference.
Subclassing keeps their layer definitions, parameter names (`net.0.weight`, ...) and
initialisation exactly, while adding the shared signature. The decoder subclass also adds
`spatial_hw` (the original hard-codes the 8×8 grid), so Kodak-size decoding works as for
ADJSCC-Q.

**Verified by tests:** the Conv2d / ConvTranspose2d shapes and the PReLU count equal
ADJSCC-Q's; no `AFModule` is present; the parameter difference equals exactly the AF-module
parameters (3,851 at k/n = 1/6).

## AD-03: same latent budget and quantiser as ADJSCC-Q (G1)

**Decision.** k/n = 1/6 → 16 latent channels × 8×8 = 1024 values. The same 256-level learnable
codebook (1 byte per value) and the same calibration (1st–99th percentile) are used.

**Why.** With an identical byte budget, every RS scheme in `coding/rs_pipeline.py`, every
packet mask and the side-info rule (16 B per image for ranked methods) apply unchanged. Any
difference in UEP behaviour can then only come from the model.

## AD-04: `snr_db` accepted and ignored by models without SNR-conditioned layers (G0)

**Decision.** DeepJSCC-Q accepts the `snr_db` argument and ignores it. Its config keeps
`af_conditioning.nominal_snr_db: 10.0` only because the evaluation code passes one nominal SNR
to every model.

**Note.** For ADJSCC-Q this nominal value is also a constant (identity-channel training at a
fixed 10 dB), so its AF gates already act as SNR-independent squeeze-and-excitation gates. G1
therefore removes input-dependent per-channel gating, not SNR adaptation. Wording in the
paper should reflect this.

## AD-05: training recipe v1 unchanged for DeepJSCC-Q (G1)

**Date:** 2026-11-06. **Status:** accepted; confirmed by the owners on 2026-11-06.

**Decision.** `configs/deepjsccq_identity_train.yaml` is the ADJSCC-Q config with only
`model.arch` and the output folder changed: Adam lr 1e-3 constant, no gradient clipping,
batch 64, 60 epochs, quantiser σ annealed 1.0 → 0.05 over 50 epochs, best-validation
checkpoint.

**Why.** The architecture must be the only difference between the two families. Recipes v2
(cosine + clipping) and v3 (lr 3e-4) were tried on ADJSCC-Q and rejected (PROGRESS_LOG).

**Risk.** Recipe v1 diverged for 2 of 7 ADJSCC-Q runs. DeepJSCC-Q may behave similarly;
AD-06 handles this.

## AD-06: init seeds 42, 100, 7, 13 (+21 spare); convergence rule unchanged (G1)

**Status:** accepted; confirmed by the owners on 2026-11-06 (4 seeds plus spare 21, not 5).

**Decision.** Train four DeepJSCC-Q models with init seeds 42, 100, 7 and 13, split seed 42.
These are four of the five converged ADJSCC-Q seeds. Because the loader seed is
`seed + 1`, each DeepJSCC-Q model sees the same data order as its ADJSCC-Q namesake.
Convergence uses the existing rule (best validation epoch ≥ 50, `convergence_ledger.json`).

If a run does not converge, it is **reported, not hidden or replaced silently**: train the
spare seed 21 and list both in the comparison (non-converged models are excluded from family
statistics automatically and named in `compare.md`).

**Alternative.** Five seeds (adding 21) would match the ADJSCC-Q family size exactly, at
about +1.5 h of MPS training plus one more test sweep.

## AD-07: hardware; G1 on Apple MPS, optional same-hardware anchor (G1)

**Status:** accepted on 2026-11-06: the owners chose to run the anchor. Steps `g1_anchor_train`,
`g1_anchor_final` and `g1_compare_with_anchor` are part of the G1 plan, and
`g1_compare_with_anchor` is the comparison of record.

**Context.** The ADJSCC-Q family was trained on Kaggle T4 GPUs. G1 will be trained on a
Mac (MPS). PyTorch does not promise bit-identical results across backends, and kernel
choices can differ. A systematic hardware effect on the damage profile is unlikely, but it
cannot be ruled out without a control.

**Decision.** Steps `g1_anchor_train` and `g1_anchor_final` retrain
ADJSCC-Q seed 42 on the same Mac and evaluate it with the published ADJSCC-Q scheme selection.
`g1_compare_with_anchor` then shows it as its own family (`adjsccq_anchor`). If the anchor
matches the published ADJSCC-Q profile and gains, hardware is excluded as an explanation for
any ADJSCC/DeepJSCC difference. Cost: one training run (~1.5 h on an M2, from the 87-minute
ADJSCC-Q MPS run in the RUNBOOK) plus one test sweep.

## AD-08: device handling; explicit `--device mps`, `auto` unchanged, CPU evaluation (G1)

**Decision.**
* `train_adjsccq_train_split.py --device {auto,cpu,cuda,mps}`. `auto` keeps the pre-G0 rule
  (CUDA if available, else CPU), so no existing launcher changes behaviour. `mps` and `cuda`
  must be requested explicitly and fail loudly if unavailable (no silent CPU fallback for the
  whole run).
* `PYTORCH_ENABLE_MPS_FALLBACK=1` lets an individual op with no MPS kernel fall back to the CPU
  (e.g. `torch.quantile` in the quantiser calibration on some PyTorch versions).
  `run_pipeline.py --train_device mps` sets it for its child processes.
* Evaluation sweeps stay on the CPU (`--device cpu`, parallel `--workers`), as before. The
  pure-numpy Reed-Solomon code dominates their run time, so MPS would not help, and results are
  bit-identical for any number of workers.
* The device is recorded in `metrics.json` (`train_device`) and in every result's provenance.

**Caveat.** MPS training is not guaranteed to be bit-reproducible between two runs of the
same seed. Conclusions rest on seed-level statistics, never on one exact number.

## AD-09: G1 evaluation protocol and pre-registered decision rule (G1)

**Status:** accepted and **frozen** on 2026-11-06 by the owners, before any G1 model was
trained or any G1 test result existed. The thresholds below are the defaults of
`experiments/compare_architectures.py`. They must not be changed after test results are seen;
a change would need a new superseding entry that says what had been seen by then.

**Protocol.** This mirrors the ADJSCC-Q family exactly.
* Seed 42 (primary): train-split oracle; A.2 proxies; A.3 sensitivity head; a validation sweep
  over all 7 schemes with `oracle, predicted, energy, l1` plus a random-order control and
  `no_rs` (3 channel seeds × 5,000 validation images, as for ADJSCC-Q); then `select_scheme`.
* Seeds 100/7/13 reuse seed 42's validation selection, as ADJSCC-Q seeds 100/7/13/21 did.
* Test sweeps: 5 channel seeds × all 10,000 test images, sources `oracle, energy` (+ `predicted, l1`
  for seed 42), random control and `no_rs`, sequential layout. Each test access is logged.
* Headline quantity per model: gain over uniform RS of the **best validation-selected tiered
  scheme** per source and loss rate (paired, 95% CI over channel seeds). This is the rule
  behind INSIGHTS.md and the README table.

**Decision rule** (`experiments/compare_architectures.py`; defaults shown):

| Check | Definition |
|---|---|
| C1 concentration | family mean top-4 share of LOO damage ≥ 0.40 (an even split is 0.25) |
| C2 UEP gain | every converged model: oracle gain at 40% loss ≥ +1.5 dB, 95% CI above 0 |
| C3 identity matters | every model with a random control: random gain < 0.5 × oracle gain (mean over loss ≥ 0.35) |
| FLAT | family top-4 share < 0.30 **and** oracle gain within 0.5 dB of random |

C1–C3 all hold → **"criteria met"**: UEP works on DeepJSCC-Q as on ADJSCC-Q, i.e. not an attention
effect. FLAT holds → **"no exploitable structure"**: an ADJSCC/attention-specific effect.
Otherwise → **"intermediate"**: report how the gain scales with concentration (`gain_vs_concentration` figure).

**Why these numbers.** ADJSCC-Q's converged seeds have top-4 shares of 45–53% and gains of
+2.6 to +3.2 dB at 40% loss. +1.5 dB is about half of that, and 0.40 is well above the 0.25 of
an even split.

**Known asymmetries, reported rather than fixed.**
* The published ADJSCC-Q per-seed sweeps (100/7/13/21) have no random control, so C3 for ADJSCC-Q
  rests on seed 42.
* The published seed-42 test sweep covered 5 selected schemes, while the other ADJSCC-Q seeds covered 3.
  Every G1 seed uses the same selection.

## AD-10: `g1` is an extension phase of `run_pipeline.py`, not part of `--phase all` (G1)

**Decision.** `run_pipeline.py` has `EXTENSION_PHASES = ("g1",)`. `--phase all` still means the
roadmap phases only, so re-running the roadmap never trains new models by accident.
`--phase g1`, `--only <step>` and `--list` cover the G1 steps. The comparison reads the
ADJSCC-Q results from git (`published_results/gpu/...`), so no ADJSCC-Q checkpoint has to be on
the Mac.

## AD-11: attention-dependent code paths degrade explicitly; legacy scripts stay ADJSCC-only (G0)

**Decision.**
* Importance source `attention` raises a clear `ValueError` for a model without a bottleneck gate.
* `attention_vs_damage.py` exits with a message (nothing to correlate).
* `feature_saliency.py` and `train_sensitivity_head.py` skip the attention column. Their label
  caches simply have no `attention` key.
* The training script's in-script RS sweep (attention vs oracle, legacy) is skipped for models
  without attention. Every model is evaluated by `eval_rs_uep.py`.
* These pre-pipeline scripts remain ADJSCC-only by design, since the pipeline does not use them:
  `overfit_test*.py`, `train_adjscc*.py`, `train_adjsccq_identity.py`, `toy_demo.py`,
  `attention_correlation.py`, `benchmark_training.py`, `eval_adjsccq_clean.py`.

## AD-12: keep the training entry point's name (G0)

**Decision.** `experiments/train_adjsccq_train_split.py` keeps its name although it now trains
any registered architecture. Kaggle launchers, the RUNBOOK and the progress log reference it,
and renaming would break their reproducibility.

## AD-13: out of scope for G0/G1

* No changes to `coding/` and no changes to any published result.
* No Kaggle kernels for G1 (the owners asked for a Mac/MPS route); `kaggle/make_kernels.py`
  could generate them later.
* G2 (transformer JSCC) is not started. The speech study is S1 below.

---

# Speech study (S1)

Question: does sensitivity-aware UEP carry over from images to speech, and in speech, does it
depend on attention? Both speech models are trained and evaluated with the image protocol, so
modality and attention are varied one at a time. How to run: [`S1_SPEECH_LIBRISPEECH_STUDY_RUNBOOK_MAC_MPS.md`](../03_runbooks/S1_SPEECH_LIBRISPEECH_STUDY_RUNBOOK_MAC_MPS.md).

## AD-14: speech models; 1-D port of the image backbone, with and without attention (S1)

**Date:** 2026-11-06. **Status:** accepted; owners to confirm the widths.

**Decision.** `models/speech_jscc.py` defines two registered architectures that share one code path:

| | `adjsccq1d` | `deepjsccq1d` |
|---|---|---|
| Image counterpart | ADJSCC-Q | DeepJSCC-Q (G1) |
| AF attention after every layer | yes (unmodified `AFModule` on a `(B, C, L, 1)` view) | no |

* **Input:** 16 kHz mono, 16,384 samples (1.024 s).
* **Encoder:** 5 Conv1d layers (kernel 9, strides 4,4,4,4; then kernel 5, stride 1), widths 16,32,32,32,16,
  PReLU, then the same power normalisation and the same 256-level quantiser as the image models.
* **Latent:** 16 channels × 64 steps = **1024 bytes per segment = 8 kbit/s**, which is exactly the
  CIFAR latent layout (16 × 64).
* **Decoder:** mirrors the encoder (ConvTranspose1d, kernel 8, stride 4, padding 2 = ×4 exactly), with a
  `tanh` output for waveforms in [-1, 1].

**Why.** The identical 16 × 64 latent keeps every RS scheme, packet mask and the 16 B side-info
rule unchanged, so speech-vs-image differences cannot come from the coding layer. Using the image
widths keeps the "small model" character of the image study. Two variants, rather than one, keep
modality and attention from being confounded.

**Alternatives rejected.**
* *A 2-D model on mel spectrograms reusing the image code.* It needs phase reconstruction or a
  vocoder, which would add a second, uncontrolled distortion source.
* *DeepSC-S or another published speech architecture.* It changes too many things at once.
* *Wider layers (32/64).* About 4× the compute per segment, which is too slow for six MPS runs. Revisit
  if absolute quality turns out too low to be credible.

**Risk.** At these widths the absolute quality (SNR / PESQ) will be modest. The study is about the
*relative* behaviour of UEP, as on CIFAR; PESQ/STOI (AD-17) document the absolute level honestly.

## AD-15: speech data; LibriSpeech clean subsets, speaker-disjoint, 1.024 s segments (S1)

**Status:** accepted; owners to confirm the dataset size.

**Decision** (`data/speech.py`, `experiments/prepare_speech_data.py`).

* **Splits:** train = train-clean-100, val = dev-clean, test = test-clean. These are LibriSpeech's
  official subsets and are speaker-disjoint. Only test-clean counts as the held-out test split
  (`is_official_test`); every access is logged.
* **Segmenting:** non-overlapping 16,384-sample segments from the start of each utterance.
  Segments below **−40 dBFS** RMS are dropped, since per-segment SNR is meaningless on silence. At most
  **2 segments per utterance** are drawn (seeded), then the pool is shuffled (seeded) and capped.
* **Sizes:** train **40,000** segments (about 11.4 h; mirrors CIFAR-10's 40k), val **4,000**, test **4,000**.
* **Amplitude:** raw LibriSpeech levels, no normalisation. Per-segment normalisation would need a
  gain sent as side information.
* **Cache:** int16 `.npy` plus a JSON manifest that lists every segment (utterance, speaker, offset,
  level) and the array's SHA-256. Two machines can verify they use identical data.
* **Download:** the official OpenSLR archives, extracted with a path-safety filter. Archive checksums
  are not hard-coded, since they could not be verified here. Integrity is checked by successful
  extraction, and the cache hash pins the data actually used.

**Why 4,000 evaluation segments, not 10,000.** Every evaluation worker holds the whole set in RAM
(64 KB per segment). 4,000 segments use 260 MB per worker. Statistics are paired over channel
seeds, as for CIFAR.

## AD-16: speech quality metric; per-segment SNR in the `psnr_db_*` fields, SI-SDR instead of SSIM (S1)

**Decision.**
* `experiments/metrics.quality_db_per_item` returns PSNR for images (the unchanged function) and
  per-segment **SNR** = 10 log10(‖x‖² / ‖x − x̂‖²) for audio.
* Results keep the existing `psnr_db_*` keys, so selection, reporting and the comparison work
  unchanged. Every speech result file records `quality_metric`, and the sweep tables label the
  column "SNR (dB)".
* The secondary metric is SI-SDR (`si_sdr_db_mean`), replacing SSIM for audio.

**Why this is sound.** For one segment, the difference of two SNRs equals the difference of the
corresponding PSNRs, because the signal power cancels: both are 10 log10 of an MSE ratio. So
leave-one-out damage, paired UEP gains and the decision-rule thresholds mean the same thing in both
modalities. A unit test checks this. Absolute SNR and PSNR levels are **not** comparable.

## AD-17: PESQ / STOI on a fixed subset only, plus listening examples (S1)

**Decision.** `experiments/eval_speech_perceptual.py` computes wide-band PESQ (P.862.2, `pesq`
package) and STOI (`pystoi`) on these settings:

* the first 500 test segments;
* channel seed 0;
* loss rates 0, 0.3, 0.35, 0.4 and 0.5;
* methods `no_rs`, `uniform`, and each source's validation-selected high-loss scheme.

It replays the same packet-loss realisations as the sweep and writes WAV examples.

**Why.** PESQ costs tens of milliseconds per segment, which is far too slow for the full grid
(5 seeds × 8 rates × all methods × 4,000 segments). SNR remains the metric of the decision rule;
PESQ and STOI show that the SNR ordering is perceptually meaningful.

**Rule.** A missing package or a segment PESQ rejects is reported as unavailable or rejected, never
filled in.

## AD-18: speech protocol; recipe v1, 3 seeds per architecture, image protocol repeated (S1)

**Status:** accepted; owners to confirm the number of seeds.

**Decision.**
* **Training:** `experiments/train_speech_jscc.py` mirrors the image trainer: recipe v1 (Adam 1e-3,
  constant, no clipping, MSE, σ 1 → 0.05 over 50 of 60 epochs), best-validation checkpoint, and a
  train-split oracle (mean LOO SNR drop). The checkpoint format and `metrics.json` keys are the image
  trainer's.
* **Seeds:** 42, 100 and 7 for each architecture, giving 6 models.
* **Analysis on seed 42:** A.2 proxies, A.3 head, and a validation sweep on all 7 schemes
  (3 channel seeds × 4,000 dev segments), then scheme selection.
* **Other seeds:** they reuse that selection.
* **Test sweeps:** 5 channel seeds × 4,000 test segments, with a random control and `no_rs`.
* **Attention:** `attention_vs_damage` runs on `adjsccq1d`.
* **Comparison:** `compare_architectures` with families `adjsccq1d` (reference) and `deepjsccq1d`.
* **Hardware:** both speech families train on the same Mac (MPS), so no hardware anchor is needed
  within the speech study.

**Cost.** One speech model is about 1.7× the per-item compute of the image model. The estimate is
~2.5 h per model on an M2, from the 87-minute ADJSCC-Q run, so about 15 h for six models. The smoke
test measures the real figure.

## AD-19: speech decision rule; the AD-09 thresholds, frozen before the speech test sweeps (S1)

**Status:** **PENDING**: to be confirmed before any `speech_final_*` step runs.

**Proposal.** The same thresholds as AD-09 (C1 top-4 share ≥ 0.40; C2 oracle gain at 40% loss
≥ +1.5 dB with CI > 0; C3 random < 0.5 × oracle; FLAT top-4 < 0.30 and oracle within 0.5 dB of
random), applied to each speech family. They transfer meaningfully because gains are dB of MSE
ratios in both modalities (AD-16).

**Reading.**
* `adjsccq1d` meets the criteria → UEP carries over to speech.
* `deepjsccq1d` also meets them → it does so without attention.
* The image G1 result and the speech result together give the 2 × 2 (modality × attention) picture.

## AD-20: shared code made modality-aware; image path bit-identical (S1)

**Decision.**
* **Metrics:** `experiments/metrics.py` gains `quality_db_per_item`, `per_item_mse`,
  `snr_db_per_item` and `si_sdr_db_per_item`. For 4-D (image) inputs, each returns exactly the
  previous expression (unit-tested with `torch.equal`).
* **Pipeline:** `pipeline_core`, `sensitivity` (LOO) and the Taylor source use these helpers.
  `CellAccumulator` writes `si_sdr_db_mean` instead of `ssim_mean` for audio and keeps the key order
  for images.
* **Bundle:** `ModelBundle` gains `modality` and `example_input()`. `symbols_per_channel` comes from
  the model when it reports one, otherwise the old image formula.
* **MAC counter:** `count_macs` also counts Conv1d / ConvTranspose1d layers.
* **Datasets:** `data.datasets.load_eval_tensor` dispatches `librispeech` to the speech cache.
  `eval_rs_uep`, `feature_saliency`, `train_sensitivity_head` and `attention_vs_damage` gain
  `--dataset` (default `cifar10`, so existing commands are unchanged).
* **Mismatch check:** `eval_rs_uep` refuses a dataset whose modality does not match the checkpoint.
* **Registry:** entries became `ArchSpec(encoder, decoder, modality, options)`. A NamedTuple, so
  `ARCHITECTURES[a][0]` still works.
* **Orchestrator:** a `speech` extension phase (not part of `--phase all`).

## AD-21: out of scope for S1

* **Text.** Deferred, as planned: variable-length, token-level latents do not fit the
  fixed channels × symbols layout.
* **Training losses.** Perceptual or multi-resolution STFT training losses are not used. MSE only, for
  parity with the image models; it will sound muffled, and AD-17 reports that honestly.
* **Channels.** No AWGN or SNR-adaptive training, as for the images.
* **Platform.** No Kaggle kernels; the Mac/MPS route was requested.
* **Bursts.** No speech burst study yet. `eval_burst` would need the same `--dataset` plumbing.

---

# Text study (T1)

Question: does sensitivity-aware UEP carry over to a **discrete** source, text, and does it depend on
attention there? S1 showed that the concentration of channel sensitivity is not an artefact of attention
for a continuous 1-D signal. T1 asks the same of categorical symbols, where the reconstruction is a
classification per position instead of a regression. Both text models are trained and evaluated with the
speech protocol, so modality and attention are again varied one at a time. How to run:
[`T1_TEXT_TEXT8_STUDY_RUNBOOK_MAC_MPS.md`](../03_runbooks/T1_TEXT_TEXT8_STUDY_RUNBOOK_MAC_MPS.md).

## AD-22: text models; character-level port of the speech backbone, with and without attention (T1)

**Date:** 2026-11-16. **Status:** accepted; owners to confirm the widths.

**Supersedes** the "Text" item of AD-21. AD-21 deferred text because variable-length, token-level
latents do not fit the fixed channels × symbols layout. Fixed-length **character windows** remove that
obstacle: every window has the same length, so every latent has the same 16 × 64 shape.

**Decision.** `models/text_jscc.py` defines two registered architectures that share one code path:

| | `adjsccq_text` | `deepjsccq_text` |
|---|---|---|
| Counterparts | ADJSCC-Q, `adjsccq1d` | DeepJSCC-Q (G1), `deepjsccq1d` |
| AF attention after every layer | yes (unmodified `AFModule` on a `(B, C, L, 1)` view) | no |

* **Input:** a window of 2,048 text8 characters as integer ids (27 symbols).
* **Front end:** `nn.Embedding(27, 32)`. This is the only layer without a speech counterpart; it turns
  symbols into vectors so the rest of the network is the speech design.
* **Encoder:** 5 Conv1d layers (strides 4, 2, 2, 2 with kernel 2s + 1; then kernel 5, stride 1), widths
  32, 64, 64, 64, 16, PReLU, then the same power normalisation and the same 256-level quantiser as every
  other model.
* **Latent:** 16 channels × 64 steps = **1,024 bytes per window = 4 bits per character**: again the CIFAR
  and LibriSpeech layout.
* **Decoder:** mirrors the encoder (ConvTranspose1d, kernel 2s, stride s, padding s/2 = ×s exactly). It
  outputs **logits** `(B, 27, L)`, and the reconstruction is their argmax. There is no output non-linearity
  (the speech model's `tanh` has no role here).
* Strides are a config key (`model.strides`), so a different rate keeps the 16 × 64 latent: for example
  4,096-character windows with strides 4, 4, 2, 2 = 2 bits per character.

**Why these sizes.** The speech widths (16–32) are tuned for one input channel. A 32-dimensional
embedding is a 32-channel input, so the widths are doubled (32–64). That puts the model at about
21M MACs per window, about 1.3× the speech model, which keeps six training runs feasible on one laptop.
4 bits per character is below the 4.75 bits of a context-free 27-symbol code, so the model has to use
context to do well, but leaves room for high accuracy. That keeps "clean" reconstructions readable, so
erasure damage is easy to see.

**Alternatives rejected.**
* *Sub-word tokens (BPE) with a Transformer.* Variable token counts per window, and a vocabulary of
  thousands makes a non-autoregressive decoder far harder. It changes too much at once.
* *A pretrained language model as the decoder.* An uncontrolled, very strong prior; damage would then
  measure the LM, not the latent.
* *An autoregressive decoder.* A single erased channel would cascade through the generation, so
  leave-one-out damage would no longer be a per-channel property. It is also slow for the
  17-decodes-per-item LOO and sweep grid.

**Risk.** A small non-autoregressive conv autoencoder may stay well short of perfect accuracy. As for
speech, the study is about the *relative* behaviour of UEP. The runbook gates (step 5) say when to raise
it.

## AD-23: text data; text8, standard 90M / 5M / 5M split, 2,048-character windows (T1)

**Status:** accepted; owners to confirm the dataset size.

**Decision** (`data/text.py`, `experiments/prepare_text_data.py`).

* **Corpus:** text8 (first 10^8 bytes of a 2006 English Wikipedia dump, cleaned to `a-z` and space). It is
  the standard character-level benchmark, a single 31 MB download, and has no markup to clean.
* **Splits:** the standard contiguous split used across the character-level language-modelling literature:
  train = characters [0, 90M), val = [90M, 95M), test = [95M, 100M). The regions are disjoint stretches of
  text. Only the test region counts as the held-out test split (`is_official_test`); every access is logged.
* **Windows:** non-overlapping 2,048-character windows aligned to the start of each region (remainder
  dropped), shuffled (seeded) and capped.
* **Sizes:** train **40,000** of 43,945 windows (≈ 82M characters; mirrors the 40k of CIFAR-10 and
  LibriSpeech); val and test **all 2,441** windows each (≈ 5M characters each).
* **Cache:** uint8 `.npy` plus a JSON manifest with every window offset, the array's SHA-256 and the
  source file's SHA-256. The loader checks the length (10^8) and the alphabet; the source file's hash is
  recorded rather than hard-coded, because it could not be verified here.

**Why 2,441 and not 4,000 evaluation windows.** That is every non-overlapping window of the standard 5M
region. Overlapping windows would add correlated items, not information. Statistics are paired over
channel seeds as everywhere else; 2,441 windows are about 5M characters.

## AD-24: text quality metric; character accuracy (%) in the `psnr_db_*` fields, word accuracy instead of SSIM (T1)

**Decision.**
* `experiments/metrics.quality_db_per_item` returns, for integer targets, the per-window **character
  accuracy** = 100 × (1 − Hamming error rate). The decoder is non-autoregressive with a fixed length, so
  there are no insertions or deletions and the Hamming rate is the character error rate (CER).
* Results keep the existing `psnr_db_*` keys, so selection, reporting and the comparison work
  unchanged. Every text result file records `quality_metric`, and the sweep tables label the columns
  "Char acc (%)" and "Word acc (%)".
* Damages and gains are therefore **percentage points (pp)**. Shared figure and table labels that say
  "dB" (multiseed bands, compare tables, paired-gain tables) read as pp for text. They were not renamed,
  to keep the image path byte-identical; the runbook says so wherever it matters.
* The secondary metric is **word accuracy** (`word_acc_mean`): the share of words whose characters are all
  right. The trainer also records **bits per character** (cross-entropy of the true text under the
  decoder's softmax), a smooth confidence measure.

**Why accuracy and not a dB quantity.** A dB form exists (10 log10 of the ratio of error counts, the
analogue of SNR), but it is undefined when a window has no errors, which a good model often achieves. It
is also unfamiliar to text readers. Accuracy is bounded, interpretable ("losing channel 7 costs 23 pp of
characters") and standard.

**Consequence.** Unlike S1 (AD-16), text damages are **not** in the same unit as image and speech
damages. Two quantities stay directly comparable across all three modalities: the **top-4 share**
(a ratio of damages) and the **random / oracle gain ratio** (C3). The decision thresholds in pp are
set separately (AD-28).

## AD-25: text training loss; cross-entropy instead of MSE (the only recipe change) (T1)

**Status:** accepted; owners to confirm.

**Decision.** The text trainer (`experiments/train_text_jscc.py`) uses recipe v1 unchanged (Adam 1e-3,
constant, no clipping, σ 1 → 0.05 over 50 of 60 epochs, calibration on the first batch, best-validation
checkpoint) except for the loss: **cross-entropy** of the per-position logits instead of MSE.

**Why.** MSE is the maximum-likelihood loss for a Gaussian continuous output; cross-entropy is the
maximum-likelihood loss for a categorical output. So this is the same principle, not a tuned choice.
MSE on one-hot vectors would also train, but it is a non-standard choice that a reviewer would question.

**Also changed for consistency:** the Taylor importance source uses the training loss, so for text its
gradient is that of the cross-entropy (`metrics.per_item_loss`). For images and speech, `per_item_loss` is
exactly the old `per_item_mse`.

## AD-26: text protocol; recipe v1 (with AD-25), 3 seeds per architecture, speech protocol repeated (T1)

**Status:** accepted; owners to confirm the number of seeds.

**Decision.** The `text` phase of `run_pipeline.py` is `speech` with the names changed:
* **Seeds:** 42, 100 and 7 for each architecture, giving 6 models. A non-converged run (best epoch < 50)
  is kept and named, and seed 13 is added for that architecture, as in S1.
* **Analysis on seed 42:** A.2 proxies, A.3 head, and a validation sweep on all 7 schemes
  (3 channel seeds × 2,441 val windows), then scheme selection.
* **Other seeds:** they reuse that selection.
* **Test sweeps:** 5 channel seeds × 2,441 test windows, with a random control and `no_rs`.
* **Attention:** `attention_vs_damage` runs on `adjsccq_text`.
* **Comparison:** `compare_architectures` with families `adjsccq_text` (reference) and
  `deepjsccq_text`, using the AD-28 thresholds.
* **Hardware:** all six models on the same machine and device (MPS or CUDA). No anchor is needed within T1.

## AD-27: readable examples and bits/char instead of PESQ / STOI (T1)

**Decision.** `experiments/eval_text_examples.py` replays the sweep's exact losses on a fixed subset
(the first 500 test windows, channel seed 0, loss 0 / 0.3 / 0.35 / 0.4 / 0.5, methods `no_rs`, `uniform`
and each source's validation-selected high-loss scheme). It reports character accuracy, word accuracy
and bits per character, and writes example passages with every wrong character in upper case.

**Why.** Text has no perceptual metric with the standing of PESQ. Word accuracy is the closest
"does a reader notice" measure and is cheap enough for the full sweep, so it is already there. This
script adds what a reader can judge directly: the received text.

## AD-28: text decision rule; AD-09 structure, thresholds re-expressed in percentage points (T1)

**Status:** **PENDING**: to be confirmed before any `text_final_*` step runs.

**Proposal.**
* **C1:** family mean top-4 share ≥ 0.40. Unit-free, so identical to AD-09 and AD-19.
* **C2:** every model's oracle gain at 40% loss ≥ **+2.0 pp** of character accuracy, with the 95%
  interval above zero.
* **C3:** random-order gain < 0.5 × oracle gain (mean over loss ≥ 0.35). Unit-free, identical.
* **FLAT:** top-4 share < 0.30 and oracle within **0.5 pp** of random.

Passed as `--min_gain_db 2.0 --equivalence_margin_db 0.5` (`TEXT_DECISION_RULE` in `run_pipeline.py`).

**Why +2.0 pp.** One extra correct character in every 50. At typical high-loss accuracies that is
roughly a 10% relative change in word accuracy, because a ~5.5-character word needs every letter right.
A reader notices that, and it is larger than the run-to-run noise expected from 5 seeds × 2,441
windows. It plays the role 1.5 dB plays for images. It must be fixed **before** any text test sweep is
seen, like AD-09.

**Reading.** The same as AD-19: `adjsccq_text` met → UEP carries over to text; `deepjsccq_text` also
met → without attention; together with G1 and S1 → a 3 × 2 (modality × attention) picture.

## AD-29: shared code; integer inputs mean text; image and speech paths bit-identical (T1)

**Decision.**
* **Metrics:** `experiments/metrics.py` gains `is_text`, `char_accuracy_per_item`,
  `word_accuracy_per_item`, `bits_per_char_per_item` and `per_item_loss`. `quality_db_per_item` keeps its
  image and speech branches unchanged and sends integer targets to character accuracy (unit-tested).
* **Pipeline:** `CellAccumulator` writes `word_acc_mean` instead of `ssim_mean` for text, and
  `SUMMARY_METRICS` includes it. The Taylor source uses `per_item_loss`.
* **Reporting:** the sweep table labels its columns "Char acc (%)" / "Word acc (%)" when rows carry
  `word_acc_mean`.
* **Bundle:** `ModelBundle.example_input()` returns a long zero tensor for text.
* **Datasets:** `data.datasets` gains `TEXT_DATASETS = ("text8",)`, `tensor_modality()`, and dispatches
  `text8` to `data/text.py`. `text8`/`test` counts as an official test split.
* **Mismatch check:** `eval_rs_uep` compares `tensor_modality(items)` with the checkpoint's modality. The
  old check (3-D or not) could not tell text from images.
* **Registry:** two `text` entries; `build_jscc` reads `strides`, `vocab_size` and `embed_dim` for them.
* **Orchestrator:** a `text` extension phase (not part of `--phase all`).

## AD-30: out of scope for T1

* **Sub-word / byte-level general text, other languages.** text8 only. A byte-level variant (256 symbols,
  any UTF-8 text) is a natural follow-up.
* **Semantic metrics** (BLEU, BERTScore, sentence-embedding similarity). Character and word accuracy are
  the metrics of record. A semantic metric on the example subset is a cheap later addition.
* **Language-model post-correction** at the receiver. It would mix a strong prior into the damage measure.
* **Rate sweep** (1,024 / 4,096-character windows). Supported by `model.strides` but not run.
* **Bursts.** `eval_burst` has no `--dataset` plumbing yet, as for speech.
* **Training losses / channels.** No AWGN or SNR-adaptive training, as for images and speech.
