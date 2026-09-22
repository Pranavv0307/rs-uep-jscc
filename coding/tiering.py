"""Symbol-level importance ranking and reversible tier layout."""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch

@dataclass
class SymbolTierLayout:
    """Sorted symbol view used by tier coding, with restore metadata."""

    sorted_symbols: torch.Tensor
    sorted_importance: torch.Tensor
    symbol_permutation: torch.Tensor
    tier_ids: torch.Tensor
    tier_slices: Dict[int, Tuple[int, int]]

    def restore(self, sorted_symbols: torch.Tensor) -> torch.Tensor:
        """Restore sorted symbols to the original latent order."""
        if sorted_symbols.shape != self.sorted_symbols.shape:
            raise ValueError("sorted_symbols must match the layout shape")
        restored = torch.empty_like(sorted_symbols)
        restored.scatter_(1, self.symbol_permutation, sorted_symbols)
        return restored


@dataclass
class PacketTierLayout:
    """Compatibility view for experiments that already have packets."""

    sorted_symbols: torch.Tensor
    sorted_packet_importance: torch.Tensor
    packet_permutation: torch.Tensor
    tier_ids: torch.Tensor
    tier_slices: Dict[int, Tuple[int, int]]
    packet_size: int

    def restore(self, sorted_symbols: torch.Tensor) -> torch.Tensor:
        if sorted_symbols.shape != self.sorted_symbols.shape:
            raise ValueError("sorted_symbols must match the layout shape")
        restored = torch.empty_like(sorted_symbols)
        sorted_packets = sorted_symbols.reshape(sorted_symbols.shape[0], -1, self.packet_size)
        restored_packets = restored.reshape(restored.shape[0], -1, self.packet_size)
        restored_packets.scatter_(
            1,
            self.packet_permutation.unsqueeze(-1).expand_as(sorted_packets),
            sorted_packets,
        )
        return restored.reshape_as(sorted_symbols)


def build_symbol_tier_layout(
    symbols: torch.Tensor,
    importance: torch.Tensor,
    tier_lengths: Sequence[int],
) -> SymbolTierLayout:
    """Sort symbols by importance and assign contiguous symbol tiers.

    ``tier_lengths`` are data-symbol counts in descending importance order.
    Keeping these lengths explicit lets the RS layer choose code rates and
    prove the total transmission budget before any packets are created.
    """
    if symbols.shape != importance.shape or symbols.dim() != 2:
        raise ValueError("symbols and importance must have the same shape (B, L)")
    if len(tier_lengths) != 3 or any(length <= 0 for length in tier_lengths):
        raise ValueError("tier_lengths must contain three positive values")
    if sum(tier_lengths) != symbols.shape[1]:
        raise ValueError("tier_lengths must sum to the latent symbol count")

    permutation = torch.argsort(importance, dim=1, descending=True, stable=True)
    sorted_symbols = symbols.gather(1, permutation)
    sorted_importance = importance.gather(1, permutation)
    boundaries = (0, tier_lengths[0], tier_lengths[0] + tier_lengths[1], symbols.shape[1])
    tier_ids = torch.empty_like(permutation)
    tier_ids[:, boundaries[0]:boundaries[1]] = 0
    tier_ids[:, boundaries[1]:boundaries[2]] = 1
    tier_ids[:, boundaries[2]:boundaries[3]] = 2
    return SymbolTierLayout(
        sorted_symbols=sorted_symbols,
        sorted_importance=sorted_importance,
        symbol_permutation=permutation,
        tier_ids=tier_ids,
        tier_slices={
            0: (boundaries[0], boundaries[1]),
            1: (boundaries[1], boundaries[2]),
            2: (boundaries[2], boundaries[3]),
        },
    )


def packet_importance(importance: torch.Tensor, packet_size: int) -> torch.Tensor:
    """Average symbol importance within each packet."""
    if importance.dim() != 2:
        raise ValueError("importance must have shape (B, L)")
    if packet_size <= 0 or importance.shape[1] % packet_size:
        raise ValueError("latent length must be divisible by packet_size")
    return importance.reshape(importance.shape[0], -1, packet_size).mean(dim=-1)


def build_packet_tier_layout(
    symbols: torch.Tensor,
    importance: torch.Tensor,
    packet_size: int = 32,
    tier_fractions: Sequence[float] = (0.2, 0.3, 0.5),
) -> PacketTierLayout:
    """Sort packets by mean importance and assign deterministic tiers.

    The symbols are only reordered in the returned coding view. ``restore``
    returns them to the original decoder order after coding and recovery.
    """
    if symbols.shape != importance.shape or symbols.dim() != 2:
        raise ValueError("symbols and importance must have the same shape (B, L)")
    if len(tier_fractions) != 3 or abs(sum(tier_fractions) - 1.0) > 1e-6:
        raise ValueError("tier_fractions must contain three values summing to 1")
    if any(fraction <= 0 for fraction in tier_fractions):
        raise ValueError("tier fractions must be positive")

    packets = symbols.reshape(symbols.shape[0], -1, packet_size)
    scores = packet_importance(importance, packet_size)
    permutation = torch.argsort(scores, dim=1, descending=True, stable=True)
    sorted_packets = packets.gather(1, permutation.unsqueeze(-1).expand_as(packets))
    sorted_scores = scores.gather(1, permutation)
    packet_count = packets.shape[1]

    first_count = max(1, round(packet_count * tier_fractions[0]))
    second_count = max(1, round(packet_count * tier_fractions[1]))
    if first_count + second_count >= packet_count:
        second_count = packet_count - first_count - 1
    boundaries = (0, first_count, first_count + second_count, packet_count)
    tier_ids = torch.empty_like(permutation)
    tier_ids[:, boundaries[0]:boundaries[1]] = 0
    tier_ids[:, boundaries[1]:boundaries[2]] = 1
    tier_ids[:, boundaries[2]:boundaries[3]] = 2

    return PacketTierLayout(
        sorted_symbols=sorted_packets.reshape_as(symbols),
        sorted_packet_importance=sorted_scores,
        packet_permutation=permutation,
        tier_ids=tier_ids,
        tier_slices={
            0: (boundaries[0], boundaries[1]),
            1: (boundaries[1], boundaries[2]),
            2: (boundaries[2], boundaries[3]),
        },
        packet_size=packet_size,
    )