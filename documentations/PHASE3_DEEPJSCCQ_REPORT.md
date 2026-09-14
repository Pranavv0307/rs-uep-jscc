# Phase 3 Report: DeepJSCC-Q Integration (ADJSCC-Q, Identity Channel)

**Date:** 2026-09-14
**Scope:** Add a learnable soft-to-hard scalar quantizer on top of the existing ADJSCC model, as the continuous-latent -> discrete-RS-symbol interface Phase 5 (Reed-Solomon coding) will need.

---

## 1. What was decided (recap of the planning conversation)

Three architecture questions were resolved before any code was written:

| Question | Decision | Why |
| --- | --- | --- |
| Build on DeepJSCC baseline or ADJSCC? | **ADJSCC directly** | Avoids doing the quantization integration twice — the final RS-tiering pipeline needs ADJSCC's importance signal anyway. |
| What channel during this training phase? | **Identity** (quantize -> dequantize -> decoder, no noise) | Isolates pure quantization distortion as its own number, without compounding it with the already-flagged AWGN/erasure channel-mismatch question. |
| Codebook type? | **Learnable, shared, 256-level 1D scalar codebook** | True to DeepJSCC-Q's actual contribution (a learned constellation); 256 levels maps 1:1 onto a future GF(2⁸) RS symbol byte. |

**Consequence flagged explicitly to you and repeated here:** a model trained under the identity channel is *not* expected to be robust under AWGN or packet erasure without a separate retraining run. This is not a limitation of the code — it's inherent to how the quantizer's centers and the encoder/decoder weights adapt jointly to whatever corruption they're exposed to during training. Switching channel modes later means rerunning training, not flipping a config flag.

---

## 2. Files added

All new files, none of your existing files were modified.

```
models/quantizer.py                          # ScalarSoftToHardQuantizer (new, reusable)
models/adjsccq.py                             # ADJSCCQEncoder + ADJSCCQDecoder (new)
tests/test_adjsccq.py                         # 11 tests, all passing
configs/adjsccq_identity_overfit.yaml         # smoke-test config
experiments/overfit_test_adjsccq.py           # smoke-test script (see caveat below)
```

### `models/quantizer.py` — `ScalarSoftToHardQuantizer`

The reusable piece. A single learnable 1D codebook (`nn.Parameter`, shape `(256,)`) shared across every latent dimension — one constellation reused for every transmitted symbol, matching how DeepJSCC-Q reuses one constellation rather than learning a per-dimension codebook.

- **`forward(z)`** — returns `(z_q, idx)`. `z_q` is numerically the *hard* (nearest-center) quantized value on the forward pass — i.e. exactly what would actually be transmitted — but gradients flow backward through the *soft* softmax-weighted assignment (straight-through estimator: `z_soft + (z_hard - z_soft).detach()`). `idx` is the hard nearest-center index in `[0, 256)` — this is the future RS symbol byte value, one per latent scalar.
- **`sigma_q`** (softmax temperature / "hardness") is a buffer, not a learnable parameter, deliberately — an external training loop anneals it from soft (good early gradients) to near-hard (matches true inference behavior) via `set_sigma()`. Making it learnable would let gradient descent just keep it soft forever, defeating the point.
- **`calibrate(z_sample)`** — re-initializes codebook centers from the empirical 1st/99th percentile range of a real calibration batch, instead of a guessed fixed range. This matters because `power_normalize` (in your `deepjscc.py`) enforces a *global* L2-norm constraint over the whole 2k-length vector, not a per-element range — so the actual per-scalar spread of `z` isn't something to hard-code confidently.
- **`dequantize_indices(idx)`** — looks up center values from indices. Not called by anything yet, but this is the exact function signature Phase 5's RS-decode path will need (turn recovered/imputed byte symbols back into decoder-ready floats), so the interface is already in place.

### `models/adjsccq.py` — `ADJSCCQEncoder`, `ADJSCCQDecoder`

Composition, not reimplementation: `ADJSCCQEncoder` wraps your existing `ADJSCCEncoder` (imported directly from `models/adjscc.py`, untouched) and pipes its output through `ScalarSoftToHardQuantizer`. `ADJSCCQDecoder` is a plain alias for your existing `ADJSCCDecoder` — under the identity channel, the quantized `z_q` has exactly the same shape/role as the `z` your decoder already consumes, so a separate decoder class would be pure duplication for zero behavioral difference. (Flagged in the code as something to revisit once Phase 5 puts a real channel between quantizer and decoder — at that point the decoder may need to consume RS-decoded/imputed symbols instead.)

`ADJSCCQEncoder.forward()` supports the same `return_attn` flag as your existing `ADJSCCEncoder` (so `af5_bottleneck` is still reachable for Phase 6), plus a new `return_indices` flag for the hard byte-index tensor.

`NOMINAL_SNR_DB_IDENTITY = 10.0` — since `AFModule.forward()` architecturally requires an `snr_db` value even though no noise is injected this phase, this constant fixes it at one nominal value (matching Week 1's overfit-test convention) rather than sampling a random SNR that would have no actual causal effect on any real corruption. This is explicitly documented in the code as a Phase-3-only placeholder, not a modeling claim.

---

## 3. What was verified vs. what's a scaffold

**Verified — ran against your real uploaded files, not just written and assumed correct:**

- `tests/test_adjsccq.py`: **11/11 passing**, run with `PYTHONPATH=. python3 -m pytest tests/test_adjsccq.py -v` against your actual `models/adjscc.py`, `Attention.py`, `channel.py`, `deepjscc.py`. Covers: quantizer shape contracts, hard output values are always exactly one of the 256 codebook centers, `dequantize_indices` round-trips correctly, gradients actually reach the codebook centers (straight-through estimator isn't silently dead), sigma annealing measurably changes the soft/hard gap, calibration matches the target percentile range, full `ADJSCCQEncoder` → `ADJSCCQDecoder` round trip produces correctly-shaped `[0,1]`-range output, `return_attn`/`return_indices` both work, and — most importantly — gradients reach all the way back to the ADJSCC encoder's first conv layer through the quantizer.
- A short synthetic-data training loop (4 random images, 300 steps, `sigma` annealed 1.0 → 0.05) was run end-to-end outside the test suite to confirm the training mechanics actually work, not just the forward pass in isolation: loss decreased monotonically, PSNR rose from ~10.8 dB to ~14.8 dB, and 254/256 codebook levels ended up in active use (not collapsed onto a handful of centers). This is not a real result — 4 random-noise images and 300 steps prove nothing about actual reconstruction quality — it's a plumbing check, and it passed.

**Update (second pass, after you sent `data/cifar10.py` and `experiments/overfit_test.py`):** both TODOs are now resolved, not guessed.

- `experiments/overfit_test_adjsccq.py` now calls the real `get_overfit_subset_loader(data_dir, n_images, batch_size)` from your `data/cifar10.py` (my first draft had guessed a nonexistent `get_first_n_images`).
- The config (`configs/adjsccq_identity_overfit.yaml`) was restructured to match `week1_overfit.yaml`'s real **nested** schema (`model:`, `data:`, `train:`, `output:`, `wandb:`) instead of my original flat guess, and the script now logs to **Weights & Biases** exactly the way `overfit_test.py` does — same `wandb.init()`/`wandb.log()`/checkpoint-artifact-upload pattern, same device-selection block (CUDA → MPS → CPU with a warning).
- One field I still can't fill in for you: `wandb.project` in the config is a placeholder (`"REPLACE_WITH_YOUR_WANDB_PROJECT_NAME"`) — I don't have `configs/week1_overfit.yaml`, which is where your real project name lives. The script will fail at `wandb.init()` until you swap that in.
- **Validated, not just written:** I ran the corrected script end-to-end in the sandbox (200 shortened epochs, with the CIFAR-10 download and W&B network calls stood in for by fakes, since this sandbox can't reach either) and confirmed: device selection, nested config parsing, calibration, the training loop, sigma annealing, PSNR/codebook-utilization reporting, the baseline-delta calculation, and the checkpoint save all work correctly end to end. Output: `Final identity-channel PSNR after 200 epochs: 11.92 dB`, `254/256 levels used`, `Delta vs unquantized ADJSCC baseline (12.00 dB): -0.08 dB` (baseline value there was a placeholder I supplied for the test, not a real number of yours). All 11 tests in `tests/test_adjsccq.py` still pass unchanged.

**Not built this phase (by design, not oversight):**

- Nothing in `coding/` was touched. RS encoding, packetization, and the packet-erasure channel are still Phase 4/5 as scoped.
- No AWGN- or erasure-mode training run exists yet. Per the identity-channel decision, that's future work once the erasure channel and fallback-imputation strategy exist.
- `configs/adjsccq_identity_overfit.yaml` has `report_baseline_psnr_db: null` — fill this in from your existing unquantized ADJSCC checkpoint's PSNR at 10 dB before running the smoke test, so the script can report the quantization-only PSNR cost as a delta rather than a raw number.

---

## 4. What quantization costs you, once you can measure it

The whole point of the identity-channel phase is to get one clean number: **how much PSNR does 256-level scalar quantization cost, independent of any channel noise?** That number doesn't exist yet — it requires an actual CIFAR-10 training run, which this sandbox can't do (no internet access to the CIFAR-10 download host). Once you fix the two `TODO(verify)` spots and run `overfit_test_adjsccq.py` on your GPU environment (Colab/Kaggle, per your usual workflow), you'll have it.

---

## 5. Suggested next steps, in order

1. Send me (or fix yourself) `data/cifar10.py` and `experiments/overfit_test.py` so `overfit_test_adjsccq.py` stops being a scaffold and matches your real conventions exactly.
2. Run the identity-channel overfit smoke test on 10 images, record the PSNR delta vs. the unquantized ADJSCC baseline at the same nominal SNR.
3. If the delta is small, scale to a full-CIFAR-10 training script analogous to `experiments/train_adjscc.py` (SNR/nominal-value sweep, W&B logging, checkpoint artifact).
4. Only after that: start Phase 4 (packetization) and Phase 5 (real RS(255,k) over GF(2⁸), packet-erasure channel) — at which point `dequantize_indices()` and the byte-valued `idx` output from `ADJSCCQEncoder` become load-bearing instead of just exposed.
5. Retrain a second ADJSCC-Q checkpoint under a real erasure channel (not identity) once Phase 5 exists and the fallback-imputation strategy (zero-fill / mean-fill / learned) is decided — the identity-channel checkpoint from this phase should not be pressed into service for that.

---

## 6. Test output (for the record)

```
$ PYTHONPATH=. python3 -m pytest tests/test_adjsccq.py -v

tests/test_adjsccq.py::test_quantizer_output_shape_matches_input PASSED
tests/test_adjsccq.py::test_quantizer_hard_values_are_from_codebook PASSED
tests/test_adjsccq.py::test_quantizer_dequantize_indices_matches_forward_hard_values PASSED
tests/test_adjsccq.py::test_quantizer_gradient_flows_to_centers PASSED
tests/test_adjsccq.py::test_quantizer_sigma_annealing_changes_soft_hard_gap PASSED
tests/test_adjsccq.py::test_quantizer_calibrate_matches_data_range PASSED
tests/test_adjsccq.py::test_adjsccq_encoder_output_shape PASSED
tests/test_adjsccq.py::test_adjsccq_full_round_trip_output_range PASSED
tests/test_adjsccq.py::test_adjsccq_return_attn_and_indices PASSED
tests/test_adjsccq.py::test_adjsccq_calibrate_quantizer_runs PASSED
tests/test_adjsccq.py::test_adjsccq_gradient_reaches_encoder_backbone PASSED

11 passed in 1.60s
```
