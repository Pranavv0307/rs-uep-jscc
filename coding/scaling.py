"""Scale the 16-channel / 96-packet scheme templates to any latent size.

``SCHEME_CANDIDATES`` in ``coding/rs_pipeline.py`` are written for the
CIFAR-10 model (16 channels, 64 symbols per channel, 16-symbol packets, 64 data
packets, 50% overhead). Track C needs the same *relative* allocation on other
latent shapes:

* different channel counts (the k/n sweep gives 6 / 8 / 16 / 32 channels),
* a different packet size and symbols per channel (Kodak: 128-byte packets,
  24,576 symbols = 192 packets per channel).

Two rules are preserved exactly:

1. Tiers stay **channel-aligned**: every tier holds a whole number of channels.
2. The **total overhead is unchanged**: parity packets = round(overhead * data
   packets), shared between tiers in the same proportions as the template.

For ``channels=16, symbols_per_channel=64, packet_size=16`` the result equals
the original ``SCHEME_CANDIDATES`` entry (checked in the unit tests).

GF(256) limit
-------------
A Reed-Solomon codeword over GF(2^8) has at most 255 symbols, so a tier with
more than 255 packets cannot be one RS block (the roadmap's RS(1536, 768) for
Kodak is impossible as a single code). Such a tier is split into several equal
sub-blocks whose data and parity add up to the tier totals. This keeps the
overhead and the channel alignment but softens the "one big block corrects any
loss up to 1/3" property (finite block length); the report states this.
"""
from math import ceil, floor
from typing import Dict, List, Sequence, Tuple

import numpy as np

from coding.rs_pipeline import SCHEME_CANDIDATES, CodingScheme

MAX_BLOCK = 255


def largest_remainder(total: int, weights: Sequence[float]) -> List[int]:
    """Split the integer ``total`` proportionally to ``weights``; the parts sum to ``total``."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError("weights must have a positive sum")
    raw = total * weights / weights.sum()
    base = np.floor(raw + 1e-9).astype(int)
    remainder = int(total - base.sum())
    for index in np.argsort(-(raw - base), kind="stable")[:remainder]:
        base[index] += 1
    return base.tolist()


def split_block(n: int, k: int, max_block: int = MAX_BLOCK) -> List[Tuple[int, int]]:
    """Split an ``(n, k)`` tier into sub-blocks of at most ``max_block`` packets.

    The sub-block data sizes add up to ``k`` and the parity sizes to ``n - k``.
    """
    if n <= max_block:
        return [(n, k)]
    parity = n - k
    sub_blocks = ceil(n / max_block)
    while True:
        ks = largest_remainder(k, [1] * sub_blocks)
        ps = largest_remainder(parity, [1] * sub_blocks)
        blocks = [(kj + pj, kj) for kj, pj in zip(ks, ps)]
        if max(nj for nj, _ in blocks) <= max_block and min(kj for _, kj in blocks) >= 1:
            return blocks
        sub_blocks += 1
        if sub_blocks > k:
            raise ValueError(f"cannot split tier ({n}, {k}) into blocks of at most {max_block} packets")


def scaled_scheme(
    name: str,
    channels: int,
    symbols_per_channel: int,
    packet_size: int = 16,
    overhead: float = 0.5,
    max_block: int = MAX_BLOCK,
) -> CodingScheme:
    """Build scheme ``name`` for a latent of ``channels`` x ``symbols_per_channel`` symbols."""
    if symbols_per_channel % packet_size:
        raise ValueError(
            f"symbols_per_channel ({symbols_per_channel}) must be a multiple of packet_size "
            f"({packet_size}) so every tier stays channel-aligned"
        )
    packets_per_channel = symbols_per_channel // packet_size
    data_packets = channels * packets_per_channel

    if name == "no_rs":
        raw_tiers = [(data_packets, data_packets)]
    else:
        template = SCHEME_CANDIDATES[name]
        ref_data = [k for _, k in template.tiers]
        ref_parity = [n - k for n, k in template.tiers]
        tier_channels = largest_remainder(channels, ref_data)
        keep = [i for i, count in enumerate(tier_channels) if count > 0]
        tier_channels = [tier_channels[i] for i in keep]
        shares = [ref_parity[i] for i in keep]
        if sum(shares) == 0:
            shares = [1] * len(keep)
        total_parity = int(floor(overhead * data_packets + 0.5))
        parity = largest_remainder(total_parity, shares)
        raw_tiers = [(c * packets_per_channel + p, c * packets_per_channel) for c, p in zip(tier_channels, parity)]

    tiers = tuple(block for n, k in raw_tiers for block in split_block(n, k, max_block))
    return CodingScheme(tiers=tiers, packet_size=packet_size)


def scheme_catalog(
    channels: int,
    symbols_per_channel: int,
    packet_size: int = 16,
    overhead: float = 0.5,
    names: Sequence[str] = (),
    max_block: int = MAX_BLOCK,
) -> Dict[str, CodingScheme]:
    """``{name: scheme}`` for ``no_rs`` plus every requested (default: all) candidate."""
    chosen = list(names) if names else ["no_rs", *SCHEME_CANDIDATES]
    return {
        name: scaled_scheme(name, channels, symbols_per_channel, packet_size, overhead, max_block)
        for name in chosen
    }


def tier_channel_counts(scheme: CodingScheme, symbols_per_channel: int) -> List[int]:
    """Number of whole channels carried by each coding block (for reporting)."""
    return [k * scheme.packet_size // symbols_per_channel for _, k in scheme.tiers]
