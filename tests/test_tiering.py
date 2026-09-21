import torch

from coding.tiering import build_packet_tier_layout, packet_importance


def test_packet_importance_averages_symbol_scores():
    importance = torch.tensor([[1.0, 3.0, 10.0, 14.0]])
    assert torch.equal(packet_importance(importance, 2), torch.tensor([[2.0, 12.0]]))


def test_tiering_sorts_packets_and_restores_original_order():
    symbols = torch.tensor([[10, 11, 20, 21, 30, 31, 40, 41]])
    importance = torch.tensor([[0.1, 0.1, 0.8, 0.8, 0.5, 0.5, 0.2, 0.2]])
    layout = build_packet_tier_layout(
        symbols, importance, packet_size=2, tier_fractions=(0.25, 0.25, 0.5)
    )

    assert layout.sorted_symbols.tolist() == [[20, 21, 30, 31, 40, 41, 10, 11]]
    assert torch.equal(layout.restore(layout.sorted_symbols), symbols)
    assert layout.tier_ids.tolist() == [[0, 1, 2, 2]]


def test_tiering_rejects_mixed_shapes():
    try:
        build_packet_tier_layout(torch.zeros(1, 8), torch.zeros(1, 4), packet_size=2)
        assert False, "expected shape validation"
    except ValueError:
        pass