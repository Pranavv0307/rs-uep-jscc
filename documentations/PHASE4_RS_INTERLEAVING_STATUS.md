# Phase 4: Interleaved RS Pipeline, Erasure Channel, and Clean Baseline

**Date:** 2026-09-27
**Builds on:** [PHASE3_FIVE_STEPS_IMPLEMENTED.md](PHASE3_FIVE_STEPS_IMPLEMENTED.md), [PROJECT_STATUS_AND_ROADMAP.md](PROJECT_STATUS_AND_ROADMAP.md)
**State:** all changes are uncommitted in the working tree; 79 tests pass.

This document records what changed in this session, why, the current state of the pipeline, and what still needs to be done. Where it disagrees with the two documents above, this one is newer and takes precedence (see [Section 9](#9-superseded-statements-in-older-documents)).

---

## 1. Summary

| Area | Before this session | Now |
| --- | --- | --- |
| Python environment | Tests could not run (no torch/PyYAML) | `.venv` with Python 3.11 and the pinned versions; all tests pass |
| Trained checkpoint | Not checked | ADJSCC-Q identity checkpoint loads; clean baseline measured on the full test set |
| Erasure channel | Only the packetizer's basic `erase_packets()` masking | Separate seeded Bernoulli channel with mask replay for paired comparisons |
| RS packetization | Codewords split into consecutive packets (non-interleaved) | **Packet-interleaved**: every codeword runs across all packets of its tier block |
| `rs_pipeline.py` correctness | Crashed with batch size > 1; index-0 "zero-fill"; decoder used encoder-side permutation | All three fixed; receiver uses only received packets, the scheme, and the side-info lookup table |
| End-to-end run | None | First preview run on 256 test images with the real checkpoint (Section 7) |

---

## 2. Environment

The pinned `requirements-local.txt` (torch 2.4.1, numpy 1.24.4) does not install on Python 3.14, the only interpreter on this machine. A Python 3.11 virtual environment was created instead so the pinned versions match what the checkpoints were trained with.

```bash
# one-time setup (already done on this machine)
python -m pip install --user uv
python -m uv venv --python 3.11 .venv
python -m uv pip install --python .venv/Scripts/python.exe -r requirements-local.txt pytest pyyaml "scipy<1.12" wandb

# activate (Windows)
.venv\Scripts\activate
```

| Component | Version |
| --- | --- |
| Python | 3.11.16 (`.venv`, gitignored) |
| torch / torchvision | 2.4.1+cpu / 0.19.1 |
| numpy | 1.24.4 |
| scipy | 1.11.4 |
| Device | CPU only (no NVIDIA GPU on this machine) |

`pytest`, `PyYAML`, `scipy` and `wandb` are imported by the code but are **not listed** in `requirements-local.txt`. They should be added (see Section 8).

CIFAR-10 is downloaded to `cifar10_data/` (gitignored). The download is slow on this network (~30 min); `tests/test_cifar10.py` triggers it on first run.

**Test command:**

```bash
python -m pytest -q                              # full suite (79 tests)
python -m pytest -q --ignore=tests/test_cifar10.py   # skip the dataset test
```

---

## 3. Checkpoints and Clean Baseline

Checkpoints are in `results/checkpoints/` (not tracked by git):

| Checkpoint | Model |
| --- | --- |
| `adjsccq_identity_train/adjsccq_identity_final.pth` | **ADJSCC-Q, full CIFAR-10, identity channel. This is the one the RS pipeline uses.** |
| `adjsccq_identity_overfit/adjsccq_identity_overfit_final.pth` | ADJSCC-Q overfit run |
| `adjscc_identity_overfit/adjscc_identity_overfit_final.pth` | ADJSCC (unquantized) overfit run |
| `week4_adjscc/adjscc_final.pth` | ADJSCC trained under AWGN |
| `week1_overfit/overfit_final.pth` | Plain Deep JSCC overfit run |

The ADJSCC-Q checkpoint stores `{"encoder", "decoder", "config"}`. It loads with all keys matched via `experiments.eval_adjsccq_clean.load_adjsccq_checkpoint`. Its contract was confirmed:

- latent: `(B, 1024)`
- attention `af5_bottleneck`: `(B, 16)`
- indices: `(B, 1024)`, `torch.long`, values in `[0, 255]`
- quantizer `sigma_q = 0.05` (fully hard)

### Clean (no-erasure) baseline, full 10,000-image test set

Measured on the exact path the RS pipeline uses: `encoder -> byte indices -> dequantize_indices -> decoder`.

| Metric | Value |
| --- | --- |
| PSNR, mean per image | **30.53 dB** (std 2.50) |
| PSNR from dataset-level MSE (training-log convention) | 29.82 dB |
| SSIM, mean | **0.959** (std 0.026) |
| Codebook levels used | 179 / 256 |
| Max gap between index path and direct `z_q` path | 3e-7 (float rounding) |
| AF-module SNR conditioning | 10 dB nominal (no noise; identity channel) |

Saved to `results/eval/adjsccq_clean.json`. Reproduce with:

```bash
python -m experiments.eval_adjsccq_clean
```

`report_baseline_psnr_db` in `configs/adjsccq_identity_train.yaml` is 31.04 dB. If that is the unquantized ADJSCC figure under the same dataset-MSE convention, quantization costs about **1.2 dB**. This was not independently verified.

**Caveat:** this checkpoint was trained with an identity channel. It has never seen erasures or fallback-filled symbols. All erasure results below use it as-is, without retraining.

---

## 4. Pipeline as Implemented Now

```text
CIFAR-10 image (3x32x32)
 -> ADJSCCQEncoder(return_attn=True, return_indices=True), SNR conditioning 10 dB
      idx:       (B, 1024) byte indices, channel-major: symbol i belongs to channel i // 64
      attention: af5_bottleneck (B, 16)
 -> channel_order_from_attention(attention)          (B, 16) ranking = side-info lookup table
 -> permutation_from_channel_order(order, 64)        symbols of top-ranked channels first
 -> encode_blocks(idx, scheme, order)
      per tier: k data packets (row-major fill) + (n-k) parity packets,
                RS(n, k) over GF(256) down every column
 -> BernoulliPacketErasureChannel(p)                 whole packets lost, receiver knows which
 -> decode_blocks(received, scheme, order)
      per tier block: erasure-decode all columns at once
      success -> all data recovered
      failure -> received data packets kept, erased data packets marked failed
      inverse permutation -> original latent order
 -> dequantize_with_fallback(quantizer, idx, failed, 0.0)   failed symbols = 0.0 in latent space
 -> ADJSCC decoder -> reconstructed image -> PSNR / SSIM
```

### Interleaved block layout

```text
packet 0      d d d ... d   \
...                          |  k data packets (systematic)
packet k-1    d d d ... d   /
packet k      p p p ... p   \
...                          |  n - k parity packets
packet n-1    p p p ... p   /
              ^ one RS(n, k) codeword per column (16 columns = packet size)
```

Each packet holds exactly one symbol of every codeword in its block. A lost packet therefore costs every codeword one erasure, and **a block survives any `n - k` lost packets**. Tier blocks are sent back to back: all of tier 0, then tier 1, then tier 2.

### Coding schemes (`coding/rs_pipeline.py`)

All schemes use 16-symbol packets and carry 1024 data symbols (64 data packets).

| Scheme | Tiers `(n, k)` in packets | Packets sent | Symbols sent | Overhead | Loss tolerance |
| --- | --- | --- | --- | --- | --- |
| `NO_RS` | (64, 64) | 64 | 1024 | 0% | none |
| `UNIFORM_RS` | (96, 64) | 96 | 1536 | 50% | any 32 of 96 |
| `IMPORTANCE_AWARE_RS` | high (32, 16), medium (24, 16), low (40, 32) | 32 + 24 + 40 = 96 | 1536 | 50% | high: 16 of 32; medium: 8 of 24; low: 8 of 40 |
| Random-tier control | same as `IMPORTANCE_AWARE_RS`, with `random_channel_order(...)` | 96 | 1536 | 50% | same |

In the importance-aware scheme, the tier data sizes of 16 / 16 / 32 packets correspond exactly to the **top 4 / next 4 / bottom 8 channels**, because each channel is 64 symbols (4 packets). The tier split is 25% / 25% / 50%, not the 20% / 30% / 50% from the original plan.

Uniform and importance-aware send the same number of packets. The erasure channel can therefore apply the **identical loss mask** to both. `NO_RS` has 64 packets, so it cannot share a mask exactly and uses an independent draw with the same seed.

### What the receiver is allowed to use

`decode_blocks` receives only:

1. the received packets and their erased flags (the packet position acts as its sequence number);
2. the `CodingScheme`, agreed in advance;
3. the per-image channel ranking, `(B, 16)`, i.e. 16 bytes of side information per image.

The side information is currently **assumed to be delivered reliably** and is not yet counted in the transmitted-symbol budget. See Section 8.

---

## 5. Bugs Found and Fixed in the Previous `rs_pipeline.py`

| # | Problem | Effect | Fix |
| --- | --- | --- | --- |
| 1 | **Non-interleaved packetization**: each RS codeword occupied consecutive packets (RS(96,64) = 6 packets of 16) | One lost packet erased 16 symbols of a single codeword. The 16 codewords were independent 6-packet groups, each tolerating only 2 losses. At p = 0.2, P(some codeword fails) = 0.81, vs 7e-4 for the interleaved layout with the same redundancy (table below) | Interleaved block layout (Section 4) |
| 2 | `decode_importance_aware` called `layout.restore()` with a 1-row tensor on a B-row layout | `ValueError` for any batch with more than one image; the tests only used B = 1 | Rewritten decoder; tests use B = 2 and B = 3 |
| 3 | Failed codewords filled with **index 0** | Index 0 dequantizes to **-3.32**, the codebook's most extreme value; latent zero is index 186. Failures injected strong artificial values | `dequantize_with_fallback` fills 0.0 in latent space using the failure mask |
| 4 | Decoder read the permutation from the encoder-side `SymbolTierLayout` object | The receiver used information it would not have | Receiver rebuilds the permutation from the transmitted channel ranking |
| 5 | Tier block counts hard-coded as `4, 4, 8` codewords regardless of `tier_configs` | Changing the tier rates silently broke the layout | `CodingScheme` derives everything from the `(n, k)` tiers and validates the latent length |
| 6 | A failed codeword discarded all of its data, including received systematic symbols | Unnecessary loss | Received data packets are always kept |

Theoretical uniform RS(96,64) comparison under independent packet loss (1536 symbols both ways):

| Loss rate | Old layout: expected failed codewords (of 16) | Old layout: P(any failure) | Interleaved: P(block fails) |
| --- | --- | --- | --- |
| 0.1 | 0.25 | 0.23 | 1e-10 |
| 0.2 | 1.58 | 0.81 | 7e-4 |
| 0.3 | 4.09 | 0.99 | 0.20 |
| 0.4 | 7.29 | 1.00 | 0.89 |

---

## 6. Files Changed

### New files

| File | Contents |
| --- | --- |
| `coding/erasure_channel.py` | `sample_erasure_mask`, `apply_erasure_mask`, `BernoulliPacketErasureChannel`, `ChannelOutput`. Seeded independent packet loss. The channel never mutates its input, zeroes erased packet bytes, reports the realized loss rate, and accepts a fixed `erasure_mask` for paired comparisons. Sampling is done on CPU so seeds are device-independent. |
| `experiments/eval_adjsccq_clean.py` | `load_adjsccq_checkpoint(path, device)` (reusable) and the clean-baseline evaluation. Writes `results/eval/adjsccq_clean.json`. |
| `experiments/metrics.py` | `ssim` and `psnr_per_image`, moved out of `toy_demo.py`. |
| `tests/test_erasure_channel.py` | 10 tests: rate 0 and 1, seed reproducibility, whole-packet-only erasure, no input mutation, empirical rate, mask replay, validation, and a paired-mask test across uniform and importance-aware RS. |
| `documentations/PHASE4_RS_INTERLEAVING_STATUS.md` | This document. |

### Modified files

| File | Change |
| --- | --- |
| `coding/rs_pipeline.py` | **Rewritten.** `CodingScheme`; predefined `NO_RS`, `UNIFORM_RS`, `IMPORTANCE_AWARE_RS`; `encode_blocks`, `decode_blocks`, `DecodedBatch` (with `indices`, `failed`, `block_failed`); `dequantize_with_fallback`; `random_channel_order`. The old `encode_uniform` / `decode_uniform` / `encode_importance_aware` / `decode_importance_aware` are **removed**. |
| `coding/reed_solomon.py` | Added numpy-vectorized `encode_block(data_rows, n)` and `decode_block(block, erased_rows, n, k)`. They solve the erasure system once per block for all columns, including a consistency re-encode check. About 7 ms to decode an RS(96,64) block with 30 erasures. Existing `encode` / `decode` are unchanged, and the block columns are verified identical to them. |
| `coding/importance.py` | Added `channel_order_from_attention` (the side-info lookup table) and `permutation_from_channel_order` (receiver-side reconstruction; proven equal to the symbol-level stable sort). |
| `coding/packetizer.py` | Module docstring replaced: it now states that consecutive chunking is only for the no-RS path and points to the interleaved layout, the channel module and the fallback. **Code unchanged.** |
| `experiments/toy_demo.py` | `ssim` / `psnr_per_image` now imported from `experiments/metrics.py`; `sweep_compare.py`, `sweep_demo.py` and `attention_correlation.py` still import them via `toy_demo` and keep working. Unused `torch.nn.functional` import removed. The placeholder erasure logic is **still there** (Section 8). |
| `tests/test_rs_pipeline.py` | Rewritten, 15 tests: equal budgets, batch round trip for all schemes, one symbol per codeword per packet, recover 32 / fail at 33, received data kept on failure, independent tier failure, high tier surviving losses that break uniform, sort matching symbol-level ranking, wrong side info giving the wrong result, seeded random order, no-RS loss, latent-space fallback, and channel integration. |
| `tests/test_reed_solomon.py` | +4 block tests: column equivalence, recovery of `n - k` erased rows for all four tier codes, failure above capacity keeping received rows, and the parity-only-erasure fast path. |
| `tests/test_importance.py` | +1 test: channel ordering reproduces the symbol-level ranking, including ties. |

### Unchanged but now unused by the pipeline

- **`coding/tiering.py`** (`build_symbol_tier_layout`, `build_packet_tier_layout`) is superseded by the channel-order permutation in `rs_pipeline.py`. It is kept, with its tests, for now; remove it or mark it deprecated in a cleanup.
- **`erase_packets()`** in `coding/packetizer.py` is superseded by `coding/erasure_channel.py`. It is still used by `tests/test_packetizer.py`.

### Generated, not tracked by git

`.venv/`, `cifar10_data/`, `results/eval/adjsccq_clean.json`.

---

## 7. First End-to-End Preview (Not a Result)

- 256 CIFAR-10 test images (first batch)
- one channel seed
- the ADJSCC-Q identity checkpoint
- uniform, importance-aware and random-tier share the same 96-packet loss mask

Values are mean PSNR in dB.

| Loss rate p | No RS | Uniform RS | Importance-aware RS | Random-tier RS |
| --- | --- | --- | --- | --- |
| 0.0 | 30.46 | 30.46 | 30.46 | 30.46 |
| 0.2 | 21.92 | **30.46** | 27.56 | 27.77 |
| 0.3 | 20.07 | **28.06** | 22.32 | 22.73 |
| 0.4 | 18.59 | 19.78 | 19.44 | 20.14 |

Block failure rates (fraction of images):

| p | Uniform | UEP high | UEP medium | UEP low |
| --- | --- | --- | --- | --- |
| 0.2 | 0.00 | 0.00 | 0.04 | 0.39 |
| 0.3 | 0.21 | 0.01 | 0.29 | 0.93 |
| 0.4 | 0.89 | 0.13 | 0.70 | 0.99 |

What these preliminary numbers suggest, pending a proper sweep:

1. **The pipeline works.** At p = 0 every method reproduces the clean reconstruction, and RS clearly beats no protection.
2. **Uniform beats this UEP allocation up to p ≈ 0.3.** One large interleaved RS(96,64) block is nearly all-or-nothing and very strong below its ~33% threshold. The UEP low tier, RS(40,32) with 20% redundancy, is too weak and already fails 39% of the time at p = 0.2.
3. **Importance-aware ≈ random-tier, slightly worse, at every p.** Ranking channels by `af5_bottleneck` attention does not yet beat a random ranking. Either attention is not a good proxy for reconstruction sensitivity in this checkpoint, or the tier rates mask the difference. This must be tested directly (Section 8, item 3) before interpreting any UEP result.
4. The UEP advantage, if any, should appear **above the uniform threshold**: at p = 0.4, the high tier fails in only 13% of images vs 89% for uniform. The allocation needs to be tuned for that regime.

---

## 8. What Needs to Be Done Next

In recommended order.

### 8.1 Decisions needed before the final sweep

1. **Side-info transport.** The 16-byte channel ranking per image is currently assumed delivered. Options:
   - (a) keep assuming reliable delivery, and report the 16 bytes (about 1% of 1536) as overhead;
   - (b) send it through the channel, e.g. repeated inside several packets or as a heavily protected header packet, and count it in the budget.

   (a) is recommended for the first results table.
2. **Tier rates.** The current `(32,16), (24,16), (40,32)` was chosen before any results. Choose the candidate allocations **up front**, for example a stronger low tier or a 2-tier split, and document them. Do not tune on the final test set: use a validation subset (for example part of the CIFAR-10 training set) for selection.
3. **Packet size.** 16 is used everywhere. It must divide 64 symbols per channel for tiers to stay channel-aligned (1, 2, 4, 8, 16, 32 or 64 are valid).

### 8.2 Engineering work

1. **Full evaluation script.** For example `experiments/eval_rs_uep.py`, replacing `toy_demo.py`'s placeholder in the benchmark path. It should:
   - run no RS, uniform RS, importance-aware RS and random-tier RS;
   - use the same checkpoint, the same test images and the same paired 96-packet masks for all methods;
   - cover `p ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}`, with a finer grid around the uniform threshold (0.3–0.45);
   - run at least 5 channel seeds (and several random-tier orderings for the control);
   - write one CSV/JSON row per method × p × seed with: mean/std PSNR, mean SSIM, block-failure rate per tier, failed-symbol fraction, realized erasure rate, data/parity/total symbols, overhead, seed, checkpoint;
   - generate plots from that file.

   Runtime estimate on CPU: about 1.5 s per 256 images per method per p, so the full 10k test set × 4 methods × 6 p × 5 seeds takes roughly 1.5–2 hours. Use a fixed 1k-image subset while iterating.
2. **Remove the placeholder.** Delete `apply_erasure_placeholder()` from `experiments/toy_demo.py`, or rewire it to `rs_pipeline`, so `--rs_protection uniform` no longer means repetition coding.
3. **Dependencies.** Add `pytest`, `PyYAML`, `scipy` and `wandb` (with versions) to `requirements-local.txt`, and document that it needs Python ≤ 3.11/3.12.
4. **Cleanup.** Deprecate or remove `coding/tiering.py` and `erase_packets()` once nothing needs them.

### 8.3 Research work

1. **Validate attention as importance. This is the highest priority** given Section 7, item 3. Run a per-channel leave-one-out on the ADJSCC-Q checkpoint:
   - erase each of the 16 channels (fill 0.0 in latent space) and measure the PSNR drop per image;
   - compare the attention ranking with the damage ranking using Spearman and Pearson correlation;
   - use random ranking as a negative control, and report mean damage for the top / middle / bottom attention groups.

   `experiments/attention_correlation.py` exists but targets the old placeholder model path and chunking; update it to use `load_adjsccq_checkpoint`, channel-aligned groups and `dequantize_with_fallback`.
2. **Sensitivity-based importance.** If attention turns out to be a weak proxy, add an oracle or calibrated ranking (the measured per-channel damage, averaged over a training subset) as an extra method. This separates "UEP does not help" from "attention is the wrong signal".
3. **Rate allocation study.** At fixed 96 packets, compare several tier allocations, in particular ones that protect the high tier strongly while keeping the low tier above the target loss rate.
4. **Fallback study.** Compare fallback values: 0.0 (current), codebook mean, and per-channel mean.
5. **Robustness (later).** Burst erasures with a Gilbert-Elliott channel. This needs the tier blocks' packets interleaved with each other as well; they are currently sent back to back.
6. **Retraining (optional, later).** Fine-tune the decoder on erasure/fallback inputs to see whether it improves graceful degradation for all methods.

### 8.4 Reporting

Write the Phase 4/5 results report from the CSV output. Keep engineering conclusions (pipeline correctness, overhead accounting) separate from research conclusions (does attention-driven UEP beat uniform at equal overhead, and in which loss regime).

---

## 9. Superseded Statements in Older Documents

| Document | Statement | Current status |
| --- | --- | --- |
| `PROJECT_STATUS_AND_ROADMAP.md` | RS, tiering and the erasure channel are not implemented; the local environment cannot run tests | All implemented; tests run in `.venv` |
| `PROJECT_STATUS_AND_ROADMAP.md` | Use `reedsolo` | A project-owned GF(256) implementation is used instead (no external RS dependency) |
| `PROJECT_STATUS_AND_ROADMAP.md` §5.5 | "zero-fill" failed codewords | Fill is 0.0 **in latent space**; index 0 would be -3.32 |
| `PHASE3_FIVE_STEPS_IMPLEMENTED.md` | Uniform uses 32-symbol packets, 48 packets; each codeword spans 3 consecutive packets | Uniform uses 16-symbol packets, 96 packets, interleaved (one symbol per codeword per packet) |
| `PHASE3_FIVE_STEPS_IMPLEMENTED.md` | Tier codes RS(128,64) ×4, RS(96,64) ×4, RS(80,64) ×8 across consecutive packets | Same code rates, now as interleaved blocks RS(32,16), RS(24,16), RS(40,32) in packets |
| `PHASE3_FIVE_STEPS_IMPLEMENTED.md` | `encode_uniform` / `decode_uniform` API | Replaced by `encode_blocks` / `decode_blocks` with a `CodingScheme` |
| `PHASE3_FIVE_STEPS_IMPLEMENTED.md` | "Not yet implemented: different RS rates per tier ..." | That list was already stale; per-tier rates exist |
| `packetizer.py` docstring | Open question about packet vs codeword layout | Resolved: interleaved (docstring updated) |

---

## 10. Quick Reference

```python
import torch
from experiments.eval_adjsccq_clean import load_adjsccq_checkpoint, DEFAULT_CHECKPOINT
from coding.importance import channel_order_from_attention
from coding.rs_pipeline import (IMPORTANCE_AWARE_RS, UNIFORM_RS, encode_blocks,
                                decode_blocks, dequantize_with_fallback)
from coding.erasure_channel import sample_erasure_mask, apply_erasure_mask

encoder, decoder, cfg = load_adjsccq_checkpoint(DEFAULT_CHECKPOINT, "cpu")
snr = cfg["af_conditioning"]["nominal_snr_db"]

with torch.no_grad():
    _, attn, idx = encoder(x, snr, return_attn=True, return_indices=True)
order = channel_order_from_attention(attn["af5_bottleneck"])      # side info

mask = sample_erasure_mask(x.shape[0], 96, 0.3, torch.Generator().manual_seed(0))
for scheme, side_info in ((UNIFORM_RS, None), (IMPORTANCE_AWARE_RS, order)):
    packets = encode_blocks(idx, scheme, side_info)
    received = apply_erasure_mask(packets, mask).received          # same losses for both
    decoded = decode_blocks(received, scheme, side_info)
    with torch.no_grad():
        x_hat = decoder(dequantize_with_fallback(encoder.quantizer, decoded.indices, decoded.failed), snr)
```
