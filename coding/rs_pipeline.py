"""Packet-interleaved Reed-Solomon pipeline for quantized latent indices.

Layout
------
The (optionally importance-sorted) latent symbol stream is split into tiers.
Each tier is one block: its data symbols fill ``k`` packets row by row, and
RS(n, k) is applied down every column, adding ``n - k`` parity packets::

    packet 0      d d d ... d   \
    ...                          |  k data packets (systematic)
    packet k-1    d d d ... d   /
    packet k      p p p ... p   \
    ...                          |  n - k parity packets
    packet n-1    p p p ... p   /
                  ^ one RS(n, k) codeword per column

Each packet holds exactly one symbol of every codeword in its block, so a
lost packet is one erasure per codeword and a block survives any ``n - k``
lost packets. Every packet belongs to exactly one tier.

Tiers are transmitted back to back (all packets of tier 0, then tier 1, ...).
Under independent erasures the order does not matter; a burst-erasure study
would need the tier blocks interleaved with each other as well.

What the receiver knows
-----------------------
``decode_blocks`` uses only:
  * the received packets and which of them were erased (packet sequence
    numbers are implicit in the packet position);
  * the ``CodingScheme`` agreed in advance (tier ``(n, k)`` and packet size);
  * the per-image channel ranking (side information, 16 bytes per image),
    which is the lookup table for undoing the importance sort. It is
    currently assumed to be delivered reliably.

Failed blocks
-------------
Data packets that arrived are always kept (the code is systematic). Only
data packets that were erased in a block that could not be recovered are
marked in the returned ``failed`` mask. Their index value is meaningless;
use :func:`dequantize_with_fallback` so they are replaced in latent space,
not with an arbitrary codebook index.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from coding.importance import inverse_permutation, permutation_from_channel_order
from coding.packetizer import PacketizedLatent
from coding.reed_solomon import decode_block, encode_block


@dataclass(frozen=True)
class CodingScheme:
    """Static coding parameters shared by transmitter and receiver.

    ``tiers`` lists ``(n, k)`` per tier in packets, most important first.
    ``n == k`` means the tier is sent without parity.
    """

    tiers: Tuple[Tuple[int, int], ...]
    packet_size: int = 16

    def __post_init__(self):
        if self.packet_size <= 0:
            raise ValueError("packet_size must be positive")
        if not self.tiers:
            raise ValueError("at least one tier is required")
        for n, k in self.tiers:
            if not 0 < k <= n <= 255:
                raise ValueError("every tier must satisfy 0 < k <= n <= 255")

    @property
    def data_symbols(self) -> int:
        return sum(k for _, k in self.tiers) * self.packet_size

    @property
    def total_packets(self) -> int:
        return sum(n for n, _ in self.tiers)

    @property
    def transmitted_symbols(self) -> int:
        return self.total_packets * self.packet_size

    @property
    def overhead(self) -> float:
        return self.transmitted_symbols / self.data_symbols - 1.0


# 1024 latent symbols, 16-symbol packets -> 64 data packets per image.
NO_RS = CodingScheme(tiers=((64, 64),))
UNIFORM_RS = CodingScheme(tiers=((96, 64),))
# 16 / 16 / 32 data packets = top 4 / next 4 / bottom 8 channels; 96 packets total.
IMPORTANCE_AWARE_RS = CodingScheme(tiers=((32, 16), (24, 16), (40, 32)))

# Candidate fixed-budget allocations for the validation study. Data packets
# are multiples of four so each tier remains aligned to whole 64-symbol
# latent channels when packet_size is 16.
SCHEME_CANDIDATES = {
    "uniform": UNIFORM_RS,
    "two_tier_25_75": CodingScheme(tiers=((32, 16), (64, 48))),
    "two_tier_50_50": CodingScheme(tiers=((48, 32), (48, 32))),
    "two_tier_75_25": CodingScheme(tiers=((64, 48), (32, 16))),
    "three_tier_25_25_50": IMPORTANCE_AWARE_RS,
    "three_tier_25_50_25": CodingScheme(tiers=((32, 16), (48, 32), (16, 16))),
    "three_tier_50_25_25": CodingScheme(tiers=((48, 32), (24, 16), (24, 16))),
}


@dataclass
class DecodedBatch:
    indices: torch.Tensor       # (B, L) long, original latent order
    failed: torch.Tensor        # (B, L) bool, True where no valid symbol was recovered
    block_failed: torch.Tensor  # (B, num_tiers) bool, True where a tier block was unrecoverable


def _coding_permutation(channel_order: Optional[torch.Tensor], batch_size: int, length: int, device) -> torch.Tensor:
    if channel_order is None:
        return torch.arange(length, device=device).expand(batch_size, length)
    if channel_order.shape[0] != batch_size or length % channel_order.shape[1]:
        raise ValueError("channel_order must have shape (B, C) with C dividing the latent length")
    return permutation_from_channel_order(channel_order.to(device), length // channel_order.shape[1])


def encode_blocks(
    indices: torch.Tensor,
    scheme: CodingScheme,
    channel_order: Optional[torch.Tensor] = None,
) -> PacketizedLatent:
    """Sort by channel ranking (if given), RS-encode each tier across packets.

    Returns ``(B, scheme.total_packets, scheme.packet_size)`` uint8 packets.
    """
    if indices.dim() != 2 or int(indices.min()) < 0 or int(indices.max()) > 255:
        raise ValueError("indices must be a (B, L) tensor of byte values")
    batch_size, length = indices.shape
    if length != scheme.data_symbols:
        raise ValueError(f"scheme carries {scheme.data_symbols} data symbols, latent has {length}")

    permutation = _coding_permutation(channel_order, batch_size, length, indices.device)
    sorted_symbols = indices.gather(1, permutation).cpu().numpy().astype(np.uint8)

    rows = []
    for sample in sorted_symbols:
        packets, offset = [], 0
        for n, k in scheme.tiers:
            data_rows = sample[offset:offset + k * scheme.packet_size].reshape(k, scheme.packet_size)
            offset += k * scheme.packet_size
            packets.append(data_rows if n == k else encode_block(data_rows, n))
        rows.append(np.concatenate(packets, axis=0))

    packets = torch.from_numpy(np.stack(rows)).to(indices.device)
    return PacketizedLatent(
        packets=packets,
        erased=torch.zeros(batch_size, scheme.total_packets, dtype=torch.bool, device=indices.device),
        n_symbols=scheme.transmitted_symbols,
        packet_size=scheme.packet_size,
        orig_shape=torch.Size((batch_size, scheme.transmitted_symbols)),
    )


def decode_blocks(
    received: PacketizedLatent,
    scheme: CodingScheme,
    channel_order: Optional[torch.Tensor] = None,
) -> DecodedBatch:
    """RS-decode every tier block and restore the original latent order."""
    batch_size, packet_count, packet_size = received.packets.shape
    if packet_count != scheme.total_packets or packet_size != scheme.packet_size:
        raise ValueError("received packets do not match the coding scheme")

    packets = received.packets.cpu().numpy()
    erased = received.erased.cpu().numpy()
    sorted_rows, failed_rows, block_failed_rows = [], [], []
    for sample_packets, sample_erased in zip(packets, erased):
        data, failed, block_failed, offset = [], [], [], 0
        for n, k in scheme.tiers:
            block = sample_packets[offset:offset + n]
            erased_rows = np.flatnonzero(sample_erased[offset:offset + n])
            offset += n
            if n == k:
                recovered, success = block, not erased_rows.size
            else:
                recovered, success = decode_block(block, erased_rows, n, k)
            lost = np.zeros(k, dtype=bool)
            if not success:
                lost[erased_rows[erased_rows < k]] = True
            data.append(recovered.reshape(-1))
            failed.append(np.repeat(lost, packet_size))
            block_failed.append(not success)
        sorted_rows.append(np.concatenate(data))
        failed_rows.append(np.concatenate(failed))
        block_failed_rows.append(block_failed)

    device = received.packets.device
    sorted_indices = torch.from_numpy(np.stack(sorted_rows)).to(device=device, dtype=torch.long)
    sorted_failed = torch.from_numpy(np.stack(failed_rows)).to(device)
    permutation = _coding_permutation(channel_order, batch_size, scheme.data_symbols, device)
    restore = inverse_permutation(permutation)
    return DecodedBatch(
        indices=sorted_indices.gather(1, restore),
        failed=sorted_failed.gather(1, restore),
        block_failed=torch.tensor(block_failed_rows, dtype=torch.bool, device=device),
    )


def dequantize_with_fallback(quantizer, indices: torch.Tensor, failed: torch.Tensor,
                             fill_value: float = 0.0) -> torch.Tensor:
    """Dequantize indices, replacing unrecovered symbols with ``fill_value``.

    The fallback is applied in latent space. Filling the byte index instead
    (e.g. index 0) would inject the codebook's most extreme value.
    """
    latent = quantizer.dequantize_indices(indices)
    return latent.masked_fill(failed.to(latent.device), fill_value)


def random_channel_order(batch_size: int, channels: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Random per-image channel ranking for the random-tier control."""
    return torch.stack([torch.randperm(channels, generator=generator) for _ in range(batch_size)])
