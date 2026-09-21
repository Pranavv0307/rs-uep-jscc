"""Non-interleaved uniform RS pipeline for quantized latent indices."""

from dataclasses import dataclass
from typing import Tuple

import torch

from coding.packetizer import PacketizedLatent, depacketize, packetize
from coding.reed_solomon import decode, encode


@dataclass
class EncodedRSBatch:
    packetized: PacketizedLatent
    original_length: int
    codeword_n: int
    codeword_k: int
    packet_size: int


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