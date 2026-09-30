import pytest
import torch

from coding.erasure_channel import BernoulliPacketErasureChannel, apply_erasure_mask
from coding.importance import attention_to_symbol_importance, channel_order_from_attention, importance_ranking
from coding.rs_pipeline import (
    IMPORTANCE_AWARE_RS,
    NO_RS,
    UNIFORM_RS,
    CodingScheme,
    decode_blocks,
    dequantize_with_fallback,
    encode_blocks,
    random_channel_order,
    SCHEME_CANDIDATES,
)
from models.quantizer import ScalarSoftToHardQuantizer


def _latent(batch_size=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, 256, (batch_size, 1024), generator=generator)
    attention = torch.rand(batch_size, 16, generator=generator)
    return indices, channel_order_from_attention(attention), attention


def _erase(packetized, rows):
    mask = torch.zeros_like(packetized.erased)
    for sample, packet in rows:
        mask[sample, packet] = True
    return apply_erasure_mask(packetized, mask).received


def test_default_schemes_share_the_transmission_budget():
    assert UNIFORM_RS.total_packets == IMPORTANCE_AWARE_RS.total_packets == 96
    assert UNIFORM_RS.transmitted_symbols == IMPORTANCE_AWARE_RS.transmitted_symbols == 1536
    assert UNIFORM_RS.data_symbols == IMPORTANCE_AWARE_RS.data_symbols == NO_RS.data_symbols == 1024
    assert UNIFORM_RS.overhead == pytest.approx(0.5)
    assert NO_RS.overhead == 0.0


def test_candidate_schemes_share_the_fixed_budget_and_data_shape():
    assert set(SCHEME_CANDIDATES) == {
        "uniform",
        "two_tier_25_75",
        "two_tier_50_50",
        "two_tier_75_25",
        "three_tier_25_25_50",
        "three_tier_25_50_25",
        "three_tier_50_25_25",
    }
    for scheme in SCHEME_CANDIDATES.values():
        assert scheme.data_symbols == 1024
        assert scheme.total_packets == 96
        assert scheme.packet_size == 16
        assert scheme.transmitted_symbols == 1536


@pytest.mark.parametrize("scheme", list(SCHEME_CANDIDATES.values()))
def test_candidate_schemes_round_trip_for_a_batch(scheme):
    indices, order, _ = _latent(batch_size=2)
    packets = encode_blocks(indices, scheme, order)
    decoded = decode_blocks(packets, scheme, order)
    assert torch.equal(decoded.indices, indices)
    assert not decoded.failed.any()
    assert not decoded.block_failed.any()


@pytest.mark.parametrize("scheme", [NO_RS, UNIFORM_RS, IMPORTANCE_AWARE_RS])
def test_round_trip_without_erasure_for_a_batch(scheme):
    indices, order, _ = _latent()
    packets = encode_blocks(indices, scheme, order)
    assert packets.packets.shape == (3, scheme.total_packets, 16)
    decoded = decode_blocks(packets, scheme, order)
    assert torch.equal(decoded.indices, indices)
    assert not decoded.failed.any()
    assert not decoded.block_failed.any()


def test_every_packet_carries_one_symbol_of_every_codeword():
    indices, _, _ = _latent(batch_size=1)
    packets = encode_blocks(indices, UNIFORM_RS)
    # Systematic: the first 64 packets are the data, row by row.
    assert torch.equal(packets.packets[0, :64].reshape(-1).long(), indices[0])
    # Each column is one RS(96, 64) codeword that spans all 96 packets.
    from coding.reed_solomon import encode
    column = packets.packets[0, :, 5].tolist()
    assert bytes(column) == encode(column[:64], 96, 64)


def test_uniform_recovers_any_32_lost_packets_and_fails_at_33():
    indices, _, _ = _latent(batch_size=1)
    packets = encode_blocks(indices, UNIFORM_RS)
    lost = torch.randperm(96, generator=torch.Generator().manual_seed(1))

    decoded = decode_blocks(_erase(packets, [(0, int(p)) for p in lost[:32]]), UNIFORM_RS)
    assert torch.equal(decoded.indices, indices)
    assert not decoded.failed.any()

    decoded = decode_blocks(_erase(packets, [(0, int(p)) for p in lost[:33]]), UNIFORM_RS)
    assert decoded.block_failed.all()


def test_failed_block_keeps_received_data_packets():
    indices, _, _ = _latent(batch_size=1)
    packets = encode_blocks(indices, UNIFORM_RS)
    lost_data = list(range(0, 40))  # 40 data packets lost > 32 parity
    decoded = decode_blocks(_erase(packets, [(0, p) for p in lost_data]), UNIFORM_RS)
    expected_failed = torch.zeros(1, 1024, dtype=torch.bool)
    expected_failed[0, :40 * 16] = True
    assert decoded.block_failed.all()
    assert torch.equal(decoded.failed, expected_failed)
    assert torch.equal(decoded.indices[~decoded.failed], indices[~expected_failed])


def test_tiers_fail_independently():
    indices, order, _ = _latent(batch_size=1)
    packets = encode_blocks(indices, IMPORTANCE_AWARE_RS, order)
    # Low tier occupies packets 56..95 and tolerates 8 losses; lose 9 of them.
    received = _erase(packets, [(0, p) for p in range(56, 65)])
    decoded = decode_blocks(received, IMPORTANCE_AWARE_RS, order)
    assert decoded.block_failed.tolist() == [[False, False, True]]
    # The failed symbols all belong to the 8 lowest-ranked channels.
    low_channels = order[0, 8:]
    failed_channels = decoded.failed[0].reshape(16, 64).any(dim=1).nonzero().flatten()
    assert set(failed_channels.tolist()) <= set(low_channels.tolist())
    assert torch.equal(decoded.indices[~decoded.failed], indices[~decoded.failed])


def test_high_tier_survives_losses_that_break_uniform():
    indices, order, _ = _latent(batch_size=1)
    uniform = encode_blocks(indices, UNIFORM_RS)
    tiered = encode_blocks(indices, IMPORTANCE_AWARE_RS, order)
    # Same 33 packet positions for both schemes: one more than uniform tolerates.
    # For the tiered scheme they fall in the medium (32..55) and low (56..95) tiers.
    lost = [(0, p) for p in range(32, 65)]
    assert decode_blocks(_erase(uniform, lost), UNIFORM_RS).block_failed.all()
    decoded = decode_blocks(_erase(tiered, lost), IMPORTANCE_AWARE_RS, order)
    assert decoded.block_failed.tolist() == [[False, True, True]]
    high_channels = order[0, :4]
    for channel in high_channels.tolist():
        span = slice(channel * 64, (channel + 1) * 64)
        assert not decoded.failed[0, span].any()
        assert torch.equal(decoded.indices[0, span], indices[0, span])


def test_sorting_matches_symbol_level_importance_ranking():
    indices, order, attention = _latent(batch_size=2)
    importance = attention_to_symbol_importance(attention, spatial_size=8)
    permutation = importance_ranking(importance)
    packets = encode_blocks(indices, IMPORTANCE_AWARE_RS, order)
    high_tier_data = packets.packets[:, :16].reshape(2, -1).long()
    assert torch.equal(high_tier_data, indices.gather(1, permutation)[:, :256])


def test_decoder_needs_the_channel_order():
    indices, order, _ = _latent(batch_size=1)
    packets = encode_blocks(indices, IMPORTANCE_AWARE_RS, order)
    wrong = decode_blocks(packets, IMPORTANCE_AWARE_RS, torch.arange(16).unsqueeze(0))
    assert not torch.equal(wrong.indices, indices)


def test_random_channel_order_is_a_seeded_permutation():
    first = random_channel_order(4, 16, torch.Generator().manual_seed(3))
    second = random_channel_order(4, 16, torch.Generator().manual_seed(3))
    assert torch.equal(first, second)
    assert torch.equal(first.sort(dim=1).values, torch.arange(16).expand(4, 16))


def test_no_rs_loses_exactly_the_erased_packets():
    indices, _, _ = _latent(batch_size=1)
    packets = encode_blocks(indices, NO_RS)
    decoded = decode_blocks(_erase(packets, [(0, 3)]), NO_RS)
    assert decoded.failed[0].nonzero().flatten().tolist() == list(range(48, 64))
    assert torch.equal(decoded.indices[~decoded.failed], indices[~decoded.failed])


def test_fallback_is_applied_in_latent_space():
    quantizer = ScalarSoftToHardQuantizer(num_levels=256, init_range=(-3.0, 3.0))
    indices = torch.zeros(1, 4, dtype=torch.long)
    failed = torch.tensor([[False, True, False, True]])
    latent = dequantize_with_fallback(quantizer, indices, failed)
    assert latent[0, 1] == 0.0 and latent[0, 3] == 0.0
    assert latent[0, 0].item() == pytest.approx(-3.0)


def test_channel_output_decodes_for_all_schemes():
    indices, order, _ = _latent(batch_size=2)
    for scheme, channel_order in ((UNIFORM_RS, None), (IMPORTANCE_AWARE_RS, order)):
        packets = encode_blocks(indices, scheme, channel_order)
        received = BernoulliPacketErasureChannel(0.1)(packets, torch.Generator().manual_seed(0)).received
        decoded = decode_blocks(received, scheme, channel_order)
        assert torch.equal(decoded.indices[~decoded.failed], indices[~decoded.failed])


def test_scheme_validation():
    with pytest.raises(ValueError):
        CodingScheme(tiers=((10, 20),))
    with pytest.raises(ValueError):
        encode_blocks(torch.zeros(1, 1000, dtype=torch.long), UNIFORM_RS)
