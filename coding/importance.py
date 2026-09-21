"""Map ADJSCC bottleneck attention scores onto flattened latent symbols."""

from typing import Optional

import torch


def attention_to_symbol_importance(
    attention: torch.Tensor,
    spatial_size: int,
    latent_length: Optional[int] = None,
) -> torch.Tensor:
    """Broadcast one bottleneck-channel score over its spatial symbols.

    ADJSCC flattens ``(B, C, H, W)`` with ``view(B, -1)``. This function uses
    the same channel-major ordering, so each channel score occupies one
    contiguous block of ``H * W`` latent positions.
    """
    if attention.dim() != 2:
        raise ValueError(f"attention must have shape (B, C), got {tuple(attention.shape)}")
    if spatial_size <= 0:
        raise ValueError("spatial_size must be positive")

    symbols_per_channel = spatial_size * spatial_size
    importance = attention.unsqueeze(-1).expand(-1, -1, symbols_per_channel)
    importance = importance.reshape(attention.shape[0], -1)
    if latent_length is not None and importance.shape[1] != latent_length:
        raise ValueError(
            f"attention expands to {importance.shape[1]} symbols, "
            f"but latent_length is {latent_length}"
        )
    return importance


def importance_ranking(importance: torch.Tensor) -> torch.Tensor:
    """Return per-sample symbol positions from highest to lowest importance."""
    if importance.dim() != 2:
        raise ValueError(f"importance must have shape (B, L), got {tuple(importance.shape)}")
    return torch.argsort(importance, dim=1, descending=True, stable=True)


def inverse_permutation(permutation: torch.Tensor) -> torch.Tensor:
    """Return the inverse of a batch of index permutations."""
    if permutation.dim() != 2:
        raise ValueError("permutation must have shape (B, L)")
    inverse = torch.empty_like(permutation)
    positions = torch.arange(permutation.shape[1], device=permutation.device)
    inverse.scatter_(1, permutation, positions.expand_as(permutation))
    return inverse


def sort_symbols_by_importance(
    symbols: torch.Tensor,
    importance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort symbols for coding and return the permutation used."""
    if symbols.shape != importance.shape:
        raise ValueError("symbols and importance must have the same shape")
    permutation = importance_ranking(importance)
    return symbols.gather(1, permutation), permutation


def restore_symbol_order(sorted_symbols: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """Undo :func:`sort_symbols_by_importance` without changing values."""
    if sorted_symbols.shape != permutation.shape:
        raise ValueError("sorted_symbols and permutation must have the same shape")
    restored = torch.empty_like(sorted_symbols)
    restored.scatter_(1, permutation, sorted_symbols)
    return restored