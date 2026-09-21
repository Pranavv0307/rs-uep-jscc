import torch

from coding.importance import (
    attention_to_symbol_importance,
    importance_ranking,
    restore_symbol_order,
    sort_symbols_by_importance,
)


def test_attention_maps_channel_major_spatial_blocks():
    attention = torch.tensor([[0.1, 0.8, 0.3]])
    importance = attention_to_symbol_importance(attention, spatial_size=2)
    expected = torch.tensor([[0.1, 0.1, 0.1, 0.1,
                              0.8, 0.8, 0.8, 0.8,
                              0.3, 0.3, 0.3, 0.3]])
    assert torch.allclose(importance, expected)


def test_attention_mapping_checks_latent_length():
    attention = torch.ones(2, 3)
    try:
        attention_to_symbol_importance(attention, spatial_size=2, latent_length=13)
        assert False, "expected a latent-length validation error"
    except ValueError:
        pass


def test_sort_and_restore_are_exact_inverses_per_batch():
    symbols = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    importance = torch.tensor([[0.2, 0.9, 0.1, 0.7], [0.8, 0.1, 0.6, 0.3]])
    sorted_symbols, permutation = sort_symbols_by_importance(symbols, importance)

    assert sorted_symbols.tolist() == [[11, 13, 10, 12], [20, 22, 23, 21]]
    assert torch.equal(restore_symbol_order(sorted_symbols, permutation), symbols)


def test_ranking_is_descending_and_stable_for_ties():
    importance = torch.tensor([[0.5, 0.9, 0.5, 0.1]])
    assert importance_ranking(importance).tolist() == [[1, 0, 2, 3]]