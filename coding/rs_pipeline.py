"""Non-interleaved uniform RS pipeline for quantized latent indices."""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch

from coding.packetizer import PacketizedLatent, depacketize, packetize
from coding.reed_solomon import decode, encode
from coding.tiering import SymbolTierLayout, build_symbol_tier_layout


@dataclass
class EncodedRSBatch:
    packetized: PacketizedLatent
    original_length: int
    codeword_n: int
    codeword_k: int
    packet_size: int


@dataclass
class EncodedTieredRSBatch:
    packetized: PacketizedLatent
    original_length: int
    tier_layout: SymbolTierLayout
    tier_configs: Tuple[Tuple[int, int], ...]
    packet_size: int


def _encode_stream(values: Sequence[int], n: int, k: int) -> list:
    if len(values) % k:
        raise ValueError("tier data length must be divisible by codeword_k")
    encoded = bytearray()
    for start in range(0, len(values), k):
        encoded.extend(encode(values[start:start + k], n=n, k=k))
    return list(encoded)


def _validate_rs_layout(n: int, k: int, packet_size: int) -> None:
    if n % packet_size != 0:
        raise ValueError("codeword_n must be divisible by packet_size without interleaving")
    if k >= n or k <= 0:
        raise ValueError("codeword_k must satisfy 0 < k < codeword_n")


def encode_uniform(
    indices: torch.Tensor,
    n: int = 96,
    k: int = 64,
    packet_size: int = 32,
) -> EncodedRSBatch:
    """RS encode each sample, keeping every codeword contiguous in packets."""
    if indices.dim() != 2 or int(indices.min()) < 0 or int(indices.max()) > 255:
        raise ValueError("indices must be a (B, L) byte-valued tensor")
    _validate_rs_layout(n, k, packet_size)
    batch_size, original_length = indices.shape
    block_count = (original_length + k - 1) // k
    padded_length = block_count * k
    padded = torch.zeros(batch_size, padded_length, dtype=torch.long, device=indices.device)
    padded[:, :original_length] = indices

    encoded_rows = []
    for row in padded.detach().cpu().tolist():
        encoded = bytearray()
        for start in range(0, padded_length, k):
            encoded.extend(encode(row[start:start + k], n=n, k=k))
        encoded_rows.append(list(encoded))
    encoded_tensor = torch.tensor(encoded_rows, dtype=torch.long, device=indices.device)
    packetized = packetize(encoded_tensor, packet_size=packet_size)
    return EncodedRSBatch(packetized, original_length, n, k, packet_size)


def decode_uniform(
    batch: EncodedRSBatch,
    fill_failed_codeword: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode contiguous codewords and return ``(indices, failed_mask)``."""
    recovered_packets, erased_symbols = depacketize(batch.packetized)
    batch_size, encoded_length = recovered_packets.shape
    codeword_count = encoded_length // batch.codeword_n
    recovered_rows = []
    failed_rows = []

    for row_index in range(batch_size):
        row_values = recovered_packets[row_index].tolist()
        row_erased = erased_symbols[row_index].tolist()
        data_values = []
        failed_values = []
        for codeword_index in range(codeword_count):
            start = codeword_index * batch.codeword_n
            end = start + batch.codeword_n
            decoded, success = decode(
                row_values[start:end],
                [position for position, erased in enumerate(row_erased[start:end]) if erased],
                n=batch.codeword_n,
                k=batch.codeword_k,
            )
            if success:
                data_values.extend(decoded)
                failed_values.extend([False] * batch.codeword_k)
            else:
                data_values.extend([fill_failed_codeword] * batch.codeword_k)
                failed_values.extend([True] * batch.codeword_k)
        recovered_rows.append(data_values[:batch.original_length])
        failed_rows.append(failed_values[:batch.original_length])

    return (
        torch.tensor(recovered_rows, dtype=torch.long, device=recovered_packets.device),
        torch.tensor(failed_rows, dtype=torch.bool, device=recovered_packets.device),
    )


def encode_importance_aware(
    indices: torch.Tensor,
    importance: torch.Tensor,
    tier_configs: Tuple[Tuple[int, int], ...] = ((128, 64), (96, 64), (80, 64)),
    packet_size: int = 16,
) -> EncodedTieredRSBatch:
    """Sort symbols, RS encode tiers, then create transport packets.

    The default configuration uses data lengths (256, 256, 512) and sends
    512 + 384 + 640 = 1536 symbols, exactly matching uniform RS(96,64).
    """
    if indices.shape != importance.shape or indices.dim() != 2:
        raise ValueError("indices and importance must have shape (B, L)")
    if len(tier_configs) != 3:
        raise ValueError("tier_configs must contain three (n, k) pairs")
    tier_lengths = tuple(config[1] * (4 if tier_index < 2 else 8)
                        for tier_index, config in enumerate(tier_configs))
    if sum(tier_lengths) != indices.shape[1]:
        raise ValueError("default tier block counts require a 1024-symbol latent")
    for n, k in tier_configs:
        _validate_rs_layout(n, k, packet_size)

    layout = build_symbol_tier_layout(indices, importance, tier_lengths)
    encoded_rows = []
    for row in layout.sorted_symbols.detach().cpu().tolist():
        encoded_row = []
        for tier_index, (n, k) in enumerate(tier_configs):
            start, end = layout.tier_slices[tier_index]
            encoded_row.extend(_encode_stream(row[start:end], n, k))
        encoded_rows.append(encoded_row)
    encoded_tensor = torch.tensor(encoded_rows, dtype=torch.long, device=indices.device)
    if encoded_tensor.shape[1] % packet_size:
        raise ValueError("encoded tier stream must align to packet_size")
    packetized = packetize(encoded_tensor, packet_size=packet_size)
    return EncodedTieredRSBatch(
        packetized=packetized,
        original_length=indices.shape[1],
        tier_layout=layout,
        tier_configs=tier_configs,
        packet_size=packet_size,
    )


def decode_importance_aware(
    batch: EncodedTieredRSBatch,
    fill_failed_codeword: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode tier streams, then restore original latent symbol order."""
    received, erased = depacketize(batch.packetized)
    recovered_rows = []
    failed_rows = []
    for row_values, row_erased in zip(received.tolist(), erased.tolist()):
        offset = 0
        sorted_data = []
        sorted_failed = []
        for tier_index, (n, k) in enumerate(batch.tier_configs):
            start, end = batch.tier_layout.tier_slices[tier_index]
            tier_data_length = end - start
            codeword_count = tier_data_length // k
            tier_encoded_length = codeword_count * n
            tier_values = row_values[offset:offset + tier_encoded_length]
            tier_erased = row_erased[offset:offset + tier_encoded_length]
            offset += tier_encoded_length
            for codeword_index in range(codeword_count):
                code_start = codeword_index * n
                code_end = code_start + n
                decoded, success = decode(
                    tier_values[code_start:code_end],
                    [position for position, value in enumerate(tier_erased[code_start:code_end]) if value],
                    n=n,
                    k=k,
                )
                if success:
                    sorted_data.extend(decoded)
                    sorted_failed.extend([False] * k)
                else:
                    sorted_data.extend([fill_failed_codeword] * k)
                    sorted_failed.extend([True] * k)
        sorted_tensor = torch.tensor([sorted_data], dtype=torch.long, device=received.device)
        failed_tensor = torch.tensor([sorted_failed], dtype=torch.bool, device=received.device)
        restored = batch.tier_layout.restore(sorted_tensor)
        restored_failed = batch.tier_layout.restore(failed_tensor)
        recovered_rows.append(restored[0, :batch.original_length])
        failed_rows.append(restored_failed[0, :batch.original_length])
    return torch.stack(recovered_rows), torch.stack(failed_rows)