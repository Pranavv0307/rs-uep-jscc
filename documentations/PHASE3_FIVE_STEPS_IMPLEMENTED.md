# Phase 3: Five-Step Implementation Record

**Date:** 2026-09-20
**Scope:** Attention mapping, symbol ordering, GF(256) Reed-Solomon coding, packet/tier layout, and the non-interleaved uniform-RS pipeline.

## What Is Completed

The five agreed coding steps are implemented as reusable modules and tests:

1. Attention scores can be mapped to the flattened ADJSCC latent symbols.
2. Symbol sorting and exact restoration are implemented and tested.
3. A dependency-free GF(256) Reed-Solomon erasure wrapper is implemented and tested.
4. Packet-level importance ranking and high/medium/low tier layout are implemented and reversible.
5. A non-interleaved uniform-RS end-to-end index pipeline is implemented and tested.

The current implementation operates on the quantizer's byte indices. One quantized index is one GF(256) symbol, which is one byte.

## Files Added

### `coding/importance.py`

Provides:

- `attention_to_symbol_importance(attention, spatial_size, latent_length=None)`
- `importance_ranking(importance)`
- `inverse_permutation(permutation)`
- `sort_symbols_by_importance(symbols, importance)`
- `restore_symbol_order(sorted_symbols, permutation)`

The mapping follows the actual ADJSCC encoder operation:

```text
(B, C, H, W) -> view(B, C * H * W)
```

Therefore each attention score in `attn["af5_bottleneck"]` is broadcast across one contiguous spatial block of `H * W` latent symbols.

For the current configuration:

```text
bottleneck channels = 16
spatial size         = 8 x 8
latent symbols       = 16 * 8 * 8 = 1024
```

The helper produces importance with shape `(B, 1024)` from attention with shape `(B, 16)`.

### `tests/test_importance.py`

Tests:

- channel-major spatial expansion;
- latent-length validation;
- per-sample sorting and exact inverse restoration;
- stable descending ranking for equal scores.

### `coding/reed_solomon.py`

Provides a small project-owned GF(256) RS implementation:

```python
encode(data, n, k) -> bytes
decode(codeword, erasure_positions, n, k) -> (data, success)
```

The implementation:

- uses the primitive polynomial `0x11D` for GF(256);
- treats every byte as one RS symbol;
- generates systematic parity symbols;
- supports known erasure positions;
- validates `0 < k < n <= 255`;
- reports failure when erasures exceed `n-k`;
- does not silently treat unknown corruptions as erasures.

### `tests/test_reed_solomon.py`

Tests:

- no-erasure round trip;
- recovery from one complete 32-symbol packet;
- failure above the `RS(96, 64)` erasure capacity;
- full byte-value coverage;
- invalid block-length validation.

### `coding/tiering.py`

Provides packet-level importance layout:

```python
layout = build_packet_tier_layout(
    symbols,
    importance,
    packet_size=32,
    tier_fractions=(0.2, 0.3, 0.5),
)
```

It:

1. groups symbols into packets;
2. averages symbol importance within each packet;
3. sorts packets from highest to lowest importance;
4. assigns high, medium, and low tier IDs;
5. stores the packet permutation;
6. restores sorted packets to the original decoder order.

The symbols are not permanently rearranged. Sorting is only a coding view; `layout.restore()` returns the original latent packet order.

### `tests/test_tiering.py`

Tests:

- packet importance averaging;
- descending packet ranking;
- high/medium/low tier assignment;
- exact restoration of original packet order;
- shape validation.

### `coding/rs_pipeline.py`

Provides the non-interleaved uniform-RS baseline:

```python
encoded = encode_uniform(indices, n=96, k=64, packet_size=32)
recovered, failed = decode_uniform(encoded)
```

Encoding:

```text
(B, L) byte indices
    -> pad to complete k-symbol blocks
    -> RS encode each k-symbol block into n symbols
    -> packetize each contiguous encoded stream
```

Decoding:

```text
packets
    -> depacketize
    -> convert erased packet positions to erased symbol positions
    -> RS erasure decode each codeword
    -> zero-fill failed codewords
    -> trim padding
    -> return recovered indices and failure mask
```

### `tests/test_rs_pipeline.py`

Tests:

- complete `1024`-symbol round trip;
- expected packet count;
- recovery from one lost packet in a codeword.

## Current Concrete Format

The default pipeline uses:

```text
RS code:       RS(96, 64)
packet size:   32 GF(256) symbols
interleaving:  none
latent length: 1024 symbols
```

One codeword is:

```text
64 data bytes + 32 parity bytes = 96 encoded bytes
```

It is transmitted as three contiguous packets:

```text
Packet 0: codeword symbols 0..31
Packet 1: codeword symbols 32..63
Packet 2: codeword symbols 64..95
```

For one 1024-symbol latent vector:

```text
1024 data symbols
-> 16 RS codewords
-> 1536 transmitted symbols
-> 48 packets of 32 symbols
```

The overhead is:

```text
(1536 - 1024) / 1024 = 50%
```

One lost packet erases 32 symbols. `RS(96, 64)` has 32 parity symbols, so one lost packet per codeword is recoverable. Two lost packets from the same codeword exceed the known-erasure capacity.

## What Is Not Yet Complete

These five steps establish the correct coding foundation, but they do not yet constitute the final research result.

Not yet implemented:

- different RS rates for high, medium, and low tiers;
- equal-overhead optimization for the importance-aware allocation;
- an importance-aware RS encode/decode pipeline using separate tier code rates;
- a real trained-checkpoint experiment comparing uniform and importance-aware RS;
- interleaving or burst-erasure experiments;
- replacement of the placeholder in `experiments/toy_demo.py`;
- PSNR/SSIM result tables under packet erasures.

The reason is deliberate: the packet-level attention mapping and reversible layout are now available, but the final tier code rates must be selected with an exact equal-transmission-budget calculation. Implementing unequal rates before that calculation would risk comparing methods with different overhead.

## Validation Performed

Available local validation:

```bash
python3 -m py_compile \
  coding/importance.py \
  coding/reed_solomon.py \
  coding/tiering.py \
  coding/rs_pipeline.py \
  tests/test_importance.py \
  tests/test_reed_solomon.py \
  tests/test_tiering.py \
  tests/test_rs_pipeline.py
```

This passed.

A direct dependency-free RS smoke test also passed for:

- `RS(96, 64)` encode/decode with no loss;
- recovery after erasing one complete 32-symbol packet;
- failure after 33 erasures.

The full pytest suite could not be run in the current shell because `pytest`, PyTorch, and PyYAML are not installed in the active environment. The repository's existing tests still require the project's intended Python environment.

## Next Coding Step

Implement the importance-aware rate allocation on top of this foundation:

1. choose tier data block counts;
2. choose per-tier `(n, k)` rates;
3. prove total transmitted symbols equal the uniform `1536`-symbol budget;
4. encode and decode each tier with its selected rate;
5. restore original packet order;
6. compare failed codewords and reconstruction quality against uniform RS.
