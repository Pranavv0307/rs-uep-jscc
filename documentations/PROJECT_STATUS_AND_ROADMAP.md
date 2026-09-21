# Importance-Aware Reed-Solomon Coding for Deep JSCC

## Repository Status and Detailed Implementation Roadmap

**Status date:** 2026-09-20  
**Repository:** `rs-uep-jscc`  
**Research question:** Can learned importance information in an ADJSCC latent representation be used to allocate Reed-Solomon redundancy unequally, improving image reconstruction under packet erasures at the same transmission overhead as uniform protection?

This document is the working source of truth for what exists, what has been validated, what is only a placeholder, and what must be implemented next.

## 1. Executive Status

The project has a working neural baseline, a working ADJSCC attention mechanism, a working DeepJSCC-Q-style scalar quantizer, and packetization plumbing. The central research pipeline is not complete yet.

The current implementation is best described as:

```text
CIFAR-10 image
    -> ADJSCC encoder
    -> optional 256-level scalar quantizer
    -> byte-valued latent indices
    -> packet chunking
    -> placeholder packet erasure only
    -> ADJSCC decoder
```

The following is **not implemented yet**:

```text
byte-valued latent indices
    -> Reed-Solomon encoding over GF(256)
    -> packet-level erasure channel
    -> RS erasure decoding
    -> recovered latent indices
    -> dequantization
    -> ADJSCC decoder
```

The existing `uniform` option in `experiments/toy_demo.py` is a two-copy repetition stand-in. It is not Reed-Solomon and must not be used as evidence for the research hypothesis.

## 2. Repository Audit

The repository contains 35 project files, 27 Python files, and 6 YAML files. The codebase-memory index found 332 code/documentation nodes and 1,141 relationships. The committed history shows the following major additions:

| Commit | Repository milestone |
| --- | --- |
| `1f5713a1` | ADJSCC implementation and multi-SNR toy pipeline |
| `4706fb59` | DeepJSCC-Q implementation |
| `d07eb75b` | DeepJSCC-Q training work |
| `68601527` | Packetizer module and tests |
| `91448fcb` | ADJSCC identity-channel training and overfit configurations |
| `ccd6aef4` | Results directory removed from the repository |

The working tree currently contains an untracked [Problem_Statement.md](Problem_Statement.md). This roadmap is a new documentation file; no existing user changes were reverted.

## 3. What Is Done

### 3.1 Plain Deep JSCC reference

**Files:** [models/deepjscc.py](../models/deepjscc.py), [experiments/overfit_test.py](../experiments/overfit_test.py), [configs/week1_overfit.yaml](../configs/week1_overfit.yaml)

- Encoder and decoder for 32x32 CIFAR-10 images exist.
- The channel bandwidth ratio is configured around `k/n = 1/6`.
- Power normalization exists in the model path.
- AWGN channel integration exists.
- A small-image overfit workflow exists.
- This remains the plain Deep JSCC reference and should be preserved as a baseline.

### 3.2 ADJSCC baseline and attention extraction

**Files:** [models/adjscc.py](../models/adjscc.py), [models/Attention.py](../models/Attention.py), [experiments/train_adjscc.py](../experiments/train_adjscc.py), [configs/week4_adjscc.yaml](../configs/week4_adjscc.yaml)

- ADJSCC is implemented on top of the Deep JSCC architecture.
- Attention is channel recalibration, not transformer self-attention.
- Attention modules receive SNR conditioning.
- The encoder can return attention dictionaries with the `af5_bottleneck` gate.
- `af5_bottleneck` contains one value per bottleneck feature channel and is the current candidate importance signal.
- Training samples SNR per sample over a configured range.
- Evaluation uses a fixed SNR grid for comparable curves.

### 3.3 ADJSCC tests

**File:** [tests/test_adjscc.py](../tests/test_adjscc.py)

The tests cover:

- fixed-SNR shape compatibility;
- per-sample-SNR shape compatibility;
- encoder power normalization;
- decoder output range;
- attention dictionary presence and shape;
- attention values not being exactly uniform at random initialization.

These are contract and smoke tests. They do not establish image-quality performance or prove that attention is a valid importance measure.

### 3.4 DeepJSCC-Q-style quantization

**Files:** [models/quantizer.py](../models/quantizer.py), [models/adjsccq.py](../models/adjsccq.py), [experiments/train_adjsccq_identity.py](../experiments/train_adjsccq_identity.py), [experiments/overfit_test_adjsccq.py](../experiments/overfit_test_adjsccq.py)

- A learnable scalar soft-to-hard quantizer exists.
- The quantizer uses a 256-level codebook by default.
- One quantized scalar maps to one byte index in `[0, 255]`.
- The design intentionally targets one future GF(256) RS symbol per latent scalar.
- A straight-through estimator keeps gradients flowing during training.
- Codebook calibration from empirical latent percentiles exists.
- Sigma annealing from soft to hard behavior exists in the training scripts.
- `ADJSCCQEncoder` can return quantized values, attention values, and symbol indices.
- `ADJSCCQDecoder` reuses the ADJSCC decoder for the identity-channel phase.
- The current training phase is intentionally identity-channel only: quantize, dequantize, decode.

### 3.5 Quantizer tests

**File:** [tests/test_adjsccq.py](../tests/test_adjsccq.py)

The tests cover:

- output and index shapes;
- hard outputs belonging to the codebook;
- index dequantization matching the hard forward output;
- nonzero gradients to codebook centers;
- sigma behavior moving the soft path toward the hard path;
- calibration to empirical percentiles;
- ADJSCC-Q output shape and range;
- attention and index return contracts;
- calibration execution;
- gradient flow through the quantizer into the encoder.

This validates the discrete interface shape, not the complete RS communication system.

### 3.6 Packetization plumbing

**File:** [coding/packetizer.py](../coding/packetizer.py)

- A `PacketizedLatent` data structure exists.
- A `(B, L)` index tensor can be split into fixed-size packets.
- Uneven final packets are padded and later trimmed.
- Packet erasures can be sampled independently and marked.
- Erased packet bytes are zeroed.
- `depacketize()` returns recovered indices plus an erased-symbol mask.
- Packetizer tests cover no-erasure round trips, padding, masking, and erasure behavior.

Important limitation: this module explicitly does not implement RS encoding, RS decoding, GF(256) arithmetic, or a calibrated channel model. It is only packet chunking and packet masking.

### 3.7 Attention/importance investigation

**File:** [experiments/attention_correlation.py](../experiments/attention_correlation.py)

- The script divides latent values into chunks.
- It erases each chunk and measures PSNR degradation.
- It collects the corresponding attention score.
- It computes Pearson correlation and produces a scatter plot.

This is the correct type of preliminary experiment for testing whether attention can serve as a protection signal. It still needs to be run on trained checkpoints and extended with more robust statistics before it can justify an importance-aware policy.

### 3.8 Existing toy comparison

**Files:** [experiments/toy_demo.py](../experiments/toy_demo.py), [experiments/sweep_compare.py](../experiments/sweep_compare.py)

- The end-to-end model/channel/PSNR/SSIM plotting path exists.
- `none` and `uniform` modes can be visually compared.
- The erasure and uniform-protection function is clearly labelled as a placeholder.

This is useful scaffolding only. It is not a valid RS experiment until the placeholder is replaced.

## 4. What Is Partially Done or Not Yet Proven

| Area | Current state | What is missing |
| --- | --- | --- |
| Baseline quality | Scripts and tests exist | Reproducible reported PSNR/SSIM tables from saved checkpoints |
| ADJSCC importance | Attention is exposed | Evidence that attention predicts reconstruction sensitivity across trained models/seeds |
| Quantization | 256-level differentiable scalar quantizer exists | Full-dataset checkpoint and quantization-cost report under a controlled protocol |
| Packetization | Fixed-size byte packets exist | Final packet format and metadata contract for RS codewords |
| Packet erasures | Basic independent masking exists | Explicit channel object, seeded reproducibility, rate verification, and optional burst model |
| Uniform protection | Repetition placeholder exists | Real uniform RS baseline at a defined overhead |
| Importance-aware protection | No implementation | Tiering, redundancy allocation, and equal-overhead accounting |
| End-to-end RS path | Absent | Encoder, channel, decoder, failure/imputation handling, and tests |
| Experimental comparison | Sweep scaffold exists | No-RS vs uniform-RS vs importance-aware-RS results |
| Environment | Local `pytest` cannot collect | Install/use the project environment with `torch` and `PyYAML` |

## 5. Decisions That Should Be Frozen Before Coding RS

These choices must be written into the implementation and experiment configs instead of being silently changed between runs.

### 5.1 Symbol representation

Use the quantizer index tensor, not the floating-point quantized tensor, as the RS input:

```text
ADJSCCQEncoder(..., return_indices=True)
    -> idx: (B, L), torch.long, values 0..255
    -> uint8 byte symbols
    -> RS over GF(256)
```

This follows the existing quantizer design and avoids inventing a second float-to-byte conversion inside the coding module.

### 5.2 Protection unit

The first implementation should protect fixed-size latent packets or packet groups, not individual latent scalars. This keeps the channel model interpretable and matches the project problem statement.

However, the packet layout must not be frozen before the attention-to-latent mapping is understood. A packet should ideally contain symbols from one importance tier, or the tier assignment rule for mixed packets must be explicit. Otherwise the proposed method may accidentally give a packet a protection level that does not match most of its contents.

Recommended first layout:

```text
one image -> L latent bytes -> data packets of P bytes -> RS codeword per tier
```

The implementation must document whether RS is applied:

1. across packets within each importance tier; or
2. across packet columns after arranging packets as a matrix.

Start with independent RS codewords per tier because it is easiest to reason about and test.

### 5.3 First RS parameters

Start with a small, explicit GF(256) configuration that does not exceed the library's supported codeword length. A practical first comparison is:

- uniform: one RS code rate for all data;
- importance-aware: three tiers with three fixed rates;
- same total transmitted symbol count for both methods.

The exact `n` and `k` values must be selected after deciding packet size and total latent length. Do not compare methods by giving one method more transmitted symbols.

### 5.4 Importance signal

Use `attn["af5_bottleneck"]` as the first candidate signal. Since it has one score per latent channel while the latent tensor contains spatial positions, broadcast each channel score over its spatial positions before aggregating scores into packets.

Do not claim that attention is ground-truth importance. The project must compare attention-derived ranking with direct leave-one-group-out PSNR degradation.

This mapping and ranking step comes before finalizing the importance-aware packet/codeword layout. The generic RS wrapper can be implemented independently, but the end-to-end layout must wait until the symbol ordering and tier boundaries are tested.

### 5.5 Failed RS recovery

The receiver must never silently use erased ground-truth data. For a codeword that exceeds its correction capacity, choose and record one fallback:

- zero-fill the unrecovered symbols;
- codebook-mean-fill;
- learned/imputation fallback as a later extension.

Use zero-fill first because it is deterministic and easy to audit. Keep an explicit failure mask so later policies can be compared without changing the channel.

## 6. Exact Work Breakdown

### Sequencing correction

The research-order dependency is:

```text
attention extraction (done)
    -> map attention to latent symbols
    -> sort/rank symbols or packets
    -> define tiers and packet boundaries
    -> choose equal-overhead RS rates
    -> integrate uniform and importance-aware RS
```

The low-level RS encoder/decoder does not depend on attention and can be written first. The final packet/codeword arrangement does depend on attention tiering and must not be treated as settled until the mapping has been validated.

Each item below is intentionally small. Mark an item complete only when its acceptance check passes.

### Phase 0: Reproducible environment

#### 0.1 Select the intended Python interpreter

- [ ] Determine whether the checked-in `venv` is usable.
- [ ] Ensure the selected interpreter has Python, PyTorch, torchvision, NumPy, Pillow, matplotlib, PyYAML, pytest, and the chosen RS library.
- [ ] Use the same interpreter for training and testing.
- [ ] Record the interpreter path and package versions in the experiment report.

Run:

```bash
python --version
python -c "import sys; print(sys.executable)"
python -c "import torch, yaml; print(torch.__version__, yaml.__version__)"
```

Current observed problem: the active `/opt/homebrew/.../python3.14` environment failed collection because `torch` and `yaml` were missing. This is an environment blocker, not evidence that the current source tests fail.

#### 0.2 Install the project dependencies

- [ ] Install [requirements-local.txt](../requirements-local.txt) in the selected environment.
- [ ] Add `pytest`, `PyYAML`, `wandb`, and the selected RS package if they are not already installed.
- [ ] Do not install dependencies into an unrelated global interpreter.

#### 0.3 Establish a clean baseline test command

- [ ] Run `python -m pytest -q` using the selected interpreter.
- [ ] Record the number of passed tests.
- [ ] Keep this command as the required check after every coding phase.

### Phase 1: Baseline model evidence

#### 1.1 Verify plain Deep JSCC shapes

- [ ] Run the plain Deep JSCC tests.
- [ ] Record tensor shapes, `k/n`, device, and output range.

#### 1.2 Verify ADJSCC shapes and attention

- [ ] Run the ADJSCC tests.
- [ ] Confirm `af5_bottleneck` shape is `(B, C_bottleneck)`.
- [ ] Confirm latent shape is `(B, 2k)`.

#### 1.3 Produce a trained ADJSCC checkpoint

- [ ] Run [experiments/train_adjscc.py](../experiments/train_adjscc.py) with [configs/week4_adjscc.yaml](../configs/week4_adjscc.yaml) on a GPU.
- [ ] Save the final checkpoint outside git under the configured results directory.
- [ ] Record validation PSNR at SNRs `0, 5, 10, 15, 20` dB.
- [ ] Record the commit, config, seed, device, and dataset split.

#### 1.4 Produce a reproducible no-extra-protection baseline

- [ ] Run the trained model through the current pipeline with no RS and no placeholder repetition.
- [ ] Use fixed seeds and fixed image selection.
- [ ] Save per-image and aggregate PSNR/SSIM.
- [ ] Treat this as a provisional baseline until the real packet pipeline is available.

### Phase 2: Validate DeepJSCC-Q as the RS input

#### 2.1 Run the quantizer unit tests

- [ ] Confirm 256 hard levels produce indices in `[0, 255]`.
- [ ] Confirm dequantization of an index tensor exactly reconstructs the quantized value.
- [ ] Confirm gradients pass through the encoder.

#### 2.2 Run the identity-channel overfit comparison

- [ ] Run [experiments/overfit_test_adjscc_identity.py](../experiments/overfit_test_adjscc_identity.py).
- [ ] Run [experiments/overfit_test_adjsccq.py](../experiments/overfit_test_adjsccq.py).
- [ ] Use the same seed, images, model ratio, nominal SNR input, learning rate, and epochs.
- [ ] Report unquantized PSNR, quantized PSNR, PSNR delta, and codebook utilization.

#### 2.3 Train the full ADJSCC-Q identity model

- [ ] Run [experiments/train_adjsccq_identity.py](../experiments/train_adjsccq_identity.py) with [configs/adjsccq_identity_train.yaml](../configs/adjsccq_identity_train.yaml).
- [ ] Use a held-out calibration split for the final protocol rather than calibrating on evaluation images.
- [ ] Record codebook utilization and quantization distortion.
- [ ] Do not evaluate this identity-trained checkpoint as if it were packet-erasure robust.

#### 2.4 Freeze the index contract

- [ ] Add or retain a test that asserts `idx.dtype == torch.long`, shape `(B, L)`, and range `[0, 255]`.
- [ ] Convert to `torch.uint8` only at the coding boundary.
- [ ] Define how a recovered byte index is converted back through `dequantize_indices()`.

### Phase 3: Implement real Reed-Solomon coding

#### 3.1 Choose and pin the RS dependency

- [ ] Evaluate a maintained Python GF(256) package, with `reedsolo` as the first candidate.
- [ ] Pin the tested version in the dependency documentation.
- [ ] Confirm the package supports erasure positions, not only unknown-error correction.

#### 3.2 Add an RS wrapper module

Create [coding/reed_solomon.py](../coding/reed_solomon.py) with a narrow API, for example:

```python
encode(data: bytes, n: int, k: int) -> bytes
decode(codeword: bytes, erasure_positions: list[int], n: int, k: int) -> tuple[bytes, bool]
```

The wrapper must:

- validate `0 < k < n`;
- validate the supported GF(256) codeword length;
- return the original data length separately or use fixed-length metadata;
- accept explicit erasure positions;
- report whether decoding succeeded;
- never hide a decode failure;
- preserve deterministic behavior for the same bytes and erasures.

#### 3.3 Write RS unit tests before pipeline integration

Add [tests/test_reed_solomon.py](../tests/test_reed_solomon.py) covering:

- no-erasure round trip;
- one erasure;
- correction at the advertised erasure limit;
- failure above the advertised limit;
- multiple independent codewords;
- invalid `n/k` parameters;
- byte values across the full `0..255` range;
- deterministic output.

#### 3.4 Decide codeword layout

- [ ] Define whether each tier has independent codewords.
- [ ] Define how short final groups are padded.
- [ ] Define how padding is excluded after decoding.
- [ ] Define metadata: original symbol count, tier, code rate, packet size, and seed/version if required.
- [ ] Add round-trip tests for the exact chosen layout.

### Phase 4: Implement the real packet-erasure channel

#### 4.1 Separate channel simulation from packet storage

Keep [coding/packetizer.py](../coding/packetizer.py) responsible for packet representation. Add a separate channel module, such as [coding/erasure_channel.py](../coding/erasure_channel.py), responsible for delivery/loss decisions.

#### 4.2 Implement independent Bernoulli erasures first

- [ ] Input: packetized codewords and `erasure_rate`.
- [ ] Sample one delivery decision per transmitted packet.
- [ ] Return received packets, erased packet positions, and the realized erasure rate.
- [ ] Accept a seeded `torch.Generator` or equivalent RNG.
- [ ] Validate `0 <= erasure_rate <= 1`.
- [ ] Never mutate the input object.

#### 4.3 Test channel statistics and edge cases

- [ ] Rate `0` erases nothing.
- [ ] Rate `1` erases every packet.
- [ ] The same seed gives the same mask.
- [ ] Different seeds can produce different masks.
- [ ] Only whole packets are erased.
- [ ] The empirical rate is close to the requested rate over a large sample.

#### 4.4 Add burst erasures only after the independent model works

The research baseline should first use independent erasures. A Gilbert-Elliott or burst model can be a later robustness experiment and must not be mixed into the first result table.

### Phase 5: Build uniform RS end to end

#### 5.1 Integrate indices, RS, packets, channel, and decode

Implement a pipeline module, for example [coding/rs_pipeline.py](../coding/rs_pipeline.py), with this flow:

```text
image
 -> ADJSCCQ encoder with return_indices=True
 -> tier-independent symbol stream
 -> uniform RS encoder
 -> packetize codewords
 -> erase packets
 -> collect erasure positions
 -> RS erasure decode
 -> recovered symbol indices
 -> dequantize_indices
 -> ADJSCC decoder
```

#### 5.2 Define unrecoverable-codeword behavior

- [ ] Produce an explicit decode-success mask.
- [ ] Apply zero-fill or another documented fallback only to unrecovered symbols.
- [ ] Preserve recovered symbols exactly.
- [ ] Count failed codewords and erased packets in the experiment output.

#### 5.3 Add end-to-end uniform tests

Add tests that verify:

- no erasure gives the same reconstruction path as direct quantized decoding;
- erasures within RS capacity are recovered;
- erasures above capacity trigger the documented fallback;
- the decoder never receives the original erased bytes by accident;
- output shape/range remains valid.

#### 5.4 Remove the placeholder from the production path

- [ ] Replace `apply_erasure_placeholder()` in [experiments/toy_demo.py](../experiments/toy_demo.py) with the real pipeline.
- [ ] Rename CLI options if necessary so `--rs_protection uniform` means actual RS, not repetition.
- [ ] Keep the placeholder only in git history or delete it; do not leave ambiguous behavior in the benchmark path.

### Phase 6: Validate attention as importance

#### 6.1 Define the latent-to-attention mapping

For latent shape `(B, 2k)` and bottleneck spatial area `A`, reshape the real/imaginary flattened representation consistently with the ADJSCC implementation. Map each channel attention value to all latent scalars belonging to that channel.

- [ ] Write a helper that returns an importance tensor with the same logical latent positions as the byte indices.
- [ ] Test the shape and ordering with a synthetic tensor.
- [ ] Document whether real and imaginary components share one channel score.

#### 6.2 Run leave-one-group-out sensitivity analysis

- [ ] Use a trained ADJSCC checkpoint first, without RS, to isolate importance quality.
- [ ] Rank channels or packet groups by attention score.
- [ ] Erase one group at a time.
- [ ] Decode and calculate per-image PSNR/SSIM degradation.
- [ ] Repeat over multiple images and at more than one SNR.
- [ ] Repeat with at least three random seeds if compute allows.

#### 6.3 Compare attention ranking with damage ranking

Report:

- Pearson correlation;
- Spearman rank correlation;
- mean degradation for top, middle, and bottom attention groups;
- confidence intervals or standard error;
- results for random grouping as a negative control.

Acceptance condition: do not call attention useful merely because one scatter plot has a positive slope. The ranking must be more informative than random grouping under a stated test protocol.

### Phase 7: Implement importance-aware UEP

#### 7.1 Start with static three-tier grouping

Use a simple, reproducible first policy:

- high tier: top 20% of importance;
- medium tier: next 30%;
- low tier: bottom 50%.

Keep the grouping at packet level. If a packet crosses a boundary, assign it using the mean importance of its symbols or use a deterministic channel-aligned grouping rule. Record the choice.

#### 7.2 Implement fixed rate allocation

- [ ] Select three RS rates.
- [ ] Compute total transmitted symbols for the candidate allocation.
- [ ] Adjust rates until total overhead matches the uniform baseline within a documented tolerance.
- [ ] Do not choose rates after looking at the final test results.

#### 7.3 Implement importance-to-redundancy policy

The first policy should be fixed tier mapping, not a learned policy:

```text
high importance   -> strongest protection
medium importance -> middle protection
low importance    -> weakest protection
```

Later policies can compare:

- quantile tiers;
- threshold tiers;
- redundancy proportional to normalized importance;
- sensitivity-calibrated importance instead of raw attention.

#### 7.4 Add tiering tests

- [ ] Same input and seed produce the same tier assignment.
- [ ] Every symbol belongs to exactly one tier.
- [ ] Tier counts match the configured proportions within padding rules.
- [ ] Higher importance never receives less redundancy in the monotonic first policy.
- [ ] Total transmitted symbols are reported and match the comparison budget.

#### 7.5 Integrate the proposed pipeline

The final proposed path is:

```text
image
 -> ADJSCCQ encoder: z_q, attention, idx
 -> map attention to latent symbol positions
 -> group packets into high/medium/low tiers
 -> encode each tier with its assigned RS rate
 -> packet-erasure channel
 -> decode each tier with explicit erasure positions
 -> fallback only for failed codewords
 -> reassemble idx
 -> dequantize
 -> ADJSCC decoder
```

Add a test that the complete path works with zero erasures before testing loss.

### Phase 8: Create the controlled evaluation suite

#### 8.1 Freeze the comparison matrix

At minimum evaluate:

| Method | Extra protection |
| --- | --- |
| No RS | none |
| Uniform RS | one RS rate for all symbols |
| Importance-aware RS | three attention-derived tiers |
| Random-tier RS | same tier sizes/rates, random assignment control |

Use the same:

- trained checkpoint;
- CIFAR-10 test images;
- image order;
- random seeds;
- packet size;
- total transmitted-symbol budget;
- decoder and quantizer;
- packet-erasure rates;
- SNR values.

#### 8.2 Choose the first sweep

Use a small grid first:

- packet erasure rates: `0.0, 0.1, 0.2, 0.3, 0.4`;
- SNRs: choose at least `0, 10, 20` dB or clearly justify an erasure-only fixed-SNR study;
- one fixed test subset for quick iteration;
- the full test subset for final results.

#### 8.3 Record metrics

For every method and condition, save:

- mean PSNR;
- PSNR standard deviation or confidence interval;
- mean SSIM;
- failed RS codeword count;
- erased packet count and realized rate;
- transmitted data symbols;
- parity symbols;
- total symbols and overhead percentage;
- runtime if complexity is discussed;
- random seed and checkpoint identifier.

#### 8.4 Write a results file, not only plots

Add a machine-readable CSV or JSON output with one row per method/condition. Plots must be generated from that output so figures and tables cannot silently disagree.

#### 8.5 Add statistical repetitions

- [ ] Run at least 5 channel seeds for final curves.
- [ ] Report mean and uncertainty.
- [ ] Use paired channel masks where appropriate so methods are compared on the same erasure realizations.
- [ ] Keep the test image set fixed across methods.

### Phase 9: Analyze and report the research result

#### 9.1 Answer the core hypothesis

The result must answer:

> At equal transmission overhead, does importance-aware RS produce better reconstruction quality than uniform RS under packet erasures?

Possible outcomes are all valid:

- importance-aware RS wins consistently;
- it wins only in a specific erasure/SNR regime;
- it matches uniform protection;
- it loses because attention is not a reliable importance proxy;
- quantization or tier boundaries remove the expected benefit.

#### 9.2 Separate engineering conclusions from research conclusions

Report these independently:

1. Does the end-to-end RS pipeline work correctly?
2. Does quantization preserve acceptable reconstruction quality?
3. Does attention correlate with latent damage?
4. Does importance-aware redundancy beat uniform redundancy at equal overhead?
5. What additional complexity or failure modes are introduced?

#### 9.3 Update documentation

Extend [documentations/PHASE3_DEEPJSCCQ_REPORT.md](PHASE3_DEEPJSCCQ_REPORT.md) only for the quantization milestone. Create a separate Phase 4/5 report for real RS and UEP results so identity-channel findings are not mixed with packet-erasure findings.

## 7. Immediate Next Steps

The next work session should follow this exact order:

1. Select the intended Python environment and install the missing test dependencies.
2. Run `python -m pytest -q` and record the actual baseline result.
3. Inspect the available checkpoints/results after the results directory cleanup; do not assume a checkpoint exists because a script can write one.
4. Run the existing ADJSCC-Q identity overfit comparison and record quantization cost.
5. Add and test the RS dependency in isolation.
6. Implement [coding/reed_solomon.py](../coding/reed_solomon.py) and [tests/test_reed_solomon.py](../tests/test_reed_solomon.py).
7. Implement and test the attention-to-latent-symbol mapping, including channel, spatial, and real/imaginary ordering.
8. Rank symbols/packets by attention and run sensitivity analysis on a trained checkpoint; use this to validate the tiering unit.
9. Freeze packet boundaries and tier metadata based on the validated mapping.
10. Implement a separate real erasure-channel module and its tests.
11. Integrate uniform RS as the equal-protection baseline using the frozen packet layout.
12. Implement fixed three-tier UEP with equal-overhead accounting.
13. Replace the toy placeholder and build the controlled comparison sweep.

Do not start by tuning tier percentages or RS rates. First validate what the attention scores mean at the packet level, then freeze the layout, then make the uniform RS path correct and measurable. Otherwise an apparent UEP gain could come from a broken baseline, incorrect attention alignment, or unequal overhead.

## 8. Definition of Done

The project is complete only when all of the following are true:

- [ ] A reproducible environment and test command are documented.
- [ ] Plain Deep JSCC, ADJSCC, and ADJSCC-Q baselines have saved metrics.
- [ ] Quantized latent indices are explicitly validated as GF(256)-compatible bytes.
- [ ] RS encode/decode handles known erasure positions and reports failures.
- [ ] Packet erasure is separated from packet storage and is seed-reproducible.
- [ ] Uniform RS works end to end.
- [ ] Attention-to-latent mapping is tested.
- [ ] Attention usefulness is evaluated against random and sensitivity-based controls.
- [ ] Importance-aware tiers work end to end.
- [ ] Uniform and importance-aware methods use the same total transmission budget.
- [ ] No-RS, uniform-RS, importance-aware-RS, and random-tier results are saved as machine-readable data.
- [ ] Results include PSNR, SSIM, erasure rate, RS failures, and overhead.
- [ ] Final claims are limited to the measured regime and do not treat attention as guaranteed semantic importance.

## 9. Current Risks and Difficulty Ratings

Ratings are implementation/research effort estimates for this repository, not claims that the underlying theory is unsolved.

| Issue | Difficulty | Current reason |
| --- | ---: | --- |
| Reproduce Python environment | 3/10 | Dependencies are known, but interpreter mismatch currently blocks tests |
| Plain JSCC baseline | 3/10 | Existing implementation and tests |
| ADJSCC baseline | 4/10 | Existing implementation; trained-checkpoint evidence still needs organizing |
| DeepJSCC-Q quantization | 4/10 | Existing implementation and tests |
| Quantizer-to-GF(256) contract | 3/10 | 256 indices already map naturally to bytes |
| RS wrapper and erasure API | 4/10 | Mature library expected; integration details matter |
| Packet/codeword layout | 5/10 | Padding, metadata, and tier boundaries must be exact |
| Independent erasure channel | 3/10 | Straightforward once packet semantics are frozen |
| Burst-erasure extension | 5/10 | Useful extension, not required for first result |
| Uniform RS end-to-end integration | 5/10 | First full systems integration point |
| Attention-to-latent alignment | 6/10 | Channel/spatial/real-imaginary ordering must be correct |
| Proving attention is useful | 7/10 | Requires sensitivity controls, statistics, and multiple conditions |
| Three-tier importance grouping | 5/10 | Manageable after alignment is correct |
| Equal-overhead redundancy allocation | 7/10 | Main experimental fairness risk |
| Failed-codeword fallback | 5/10 | Must be explicit and consistently measured |
| Full comparison sweep | 6/10 | Many conditions and repeated seeds |
| Showing a positive UEP gain | 8/10 | This is the actual research uncertainty |
| Final report and reproducibility | 5/10 | Mostly disciplined experiment bookkeeping |

**Overall implementation difficulty:** approximately 6/10.  
**Overall research uncertainty:** approximately 8/10.  

The uncertainty is concentrated in whether attention-derived grouping produces a real gain after quantization, packetization, RS constraints, and equal-overhead accounting. The engineering path to test that hypothesis is now well defined.

## 10. Useful Commands

From the repository root:

```bash
# Run all tests with the selected interpreter
python -m pytest -q

# Run only the currently relevant contract tests
python -m pytest -q tests/test_adjscc.py tests/test_adjsccq.py tests/test_packetizer.py

# Inspect command-line options
python -m experiments.overfit_test_adjsccq --help
python -m experiments.toy_demo --help
python -m experiments.sweep_compare --help

# Train the ADJSCC baseline on GPU
python -m experiments.train_adjscc --config configs/week4_adjscc.yaml

# Train the identity-channel ADJSCC-Q model on GPU
python -m experiments.train_adjsccq_identity --config configs/adjsccq_identity_train.yaml

# Run attention sensitivity analysis after a checkpoint exists
python -m experiments.attention_correlation --help
```

The training scripts are intended for a GPU environment such as Colab or Kaggle. The local macOS environment is currently useful for code tests once dependencies are installed, but it should not be assumed to be the final training environment.
