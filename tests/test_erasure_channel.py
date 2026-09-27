import pytest
import torch

from coding.erasure_channel import (
    BernoulliPacketErasureChannel,
    apply_erasure_mask,
    sample_erasure_mask,
)
from coding.packetizer import depacketize, packetize
from coding.importance import channel_order_from_attention
from coding.rs_pipeline import IMPORTANCE_AWARE_RS, UNIFORM_RS, decode_blocks, encode_blocks


def _packets(batch_size=2, length=256, packet_size=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, 256, (batch_size, length), generator=generator)
    return indices, packetize(indices, packet_size=packet_size)


def test_rate_zero_erases_nothing():
    indices, packetized = _packets()
    output = BernoulliPacketErasureChannel(0.0)(packetized, torch.Generator().manual_seed(1))
    assert not output.received.erased.any()
    assert output.realized_rate == 0.0
    assert torch.equal(depacketize(output.received)[0], indices)


def test_rate_one_erases_every_packet():
    _, packetized = _packets()
    output = BernoulliPacketErasureChannel(1.0)(packetized, torch.Generator().manual_seed(1))
    assert output.received.erased.all()
    assert output.realized_rate == 1.0
    assert not output.received.packets.any()


def test_same_seed_gives_same_mask():
    first = sample_erasure_mask(4, 96, 0.3, torch.Generator().manual_seed(7))
    second = sample_erasure_mask(4, 96, 0.3, torch.Generator().manual_seed(7))
    assert torch.equal(first, second)


def test_different_seeds_can_give_different_masks():
    first = sample_erasure_mask(4, 96, 0.3, torch.Generator().manual_seed(7))
    second = sample_erasure_mask(4, 96, 0.3, torch.Generator().manual_seed(8))
    assert not torch.equal(first, second)


def test_only_whole_packets_are_erased():
    indices, packetized = _packets()
    output = BernoulliPacketErasureChannel(0.5)(packetized, torch.Generator().manual_seed(3))
    recovered, erased_symbols = depacketize(output.received)
    per_packet = erased_symbols.reshape(erased_symbols.shape[0], -1, packetized.packet_size)
    assert torch.equal(per_packet.all(dim=-1), per_packet.any(dim=-1))
    assert torch.equal(per_packet[:, :, 0], output.erasure_mask)
    assert torch.equal(recovered[~erased_symbols], indices[~erased_symbols])
    assert not recovered[erased_symbols].any()


def test_input_is_not_mutated():
    _, packetized = _packets()
    packets_before = packetized.packets.clone()
    BernoulliPacketErasureChannel(1.0)(packetized, torch.Generator().manual_seed(1))
    assert torch.equal(packetized.packets, packets_before)
    assert not packetized.erased.any()


def test_empirical_rate_is_close_to_requested_rate():
    mask = sample_erasure_mask(200, 500, 0.2, torch.Generator().manual_seed(0))
    assert abs(mask.float().mean().item() - 0.2) < 0.005


def test_replayed_mask_is_applied_exactly():
    _, packetized = _packets()
    mask = torch.zeros_like(packetized.erased)
    mask[0, 3] = True
    mask[1, 0] = True
    output = BernoulliPacketErasureChannel(0.9)(packetized, erasure_mask=mask)
    assert torch.equal(output.received.erased, mask)
    assert output.realized_rate == pytest.approx(2 / mask.numel())


def test_invalid_rate_and_mask_shape_are_rejected():
    _, packetized = _packets()
    with pytest.raises(ValueError):
        BernoulliPacketErasureChannel(1.5)
    with pytest.raises(ValueError):
        sample_erasure_mask(1, 4, -0.1)
    with pytest.raises(ValueError):
        apply_erasure_mask(packetized, torch.zeros(1, 1, dtype=torch.bool))


def test_uniform_and_tiered_rs_share_a_paired_mask():
    generator = torch.Generator().manual_seed(0)
    indices = torch.randint(0, 256, (1, 1024), generator=generator)
    order = channel_order_from_attention(torch.rand(1, 16, generator=generator))
    uniform = encode_blocks(indices, UNIFORM_RS)
    tiered = encode_blocks(indices, IMPORTANCE_AWARE_RS, order)
    assert uniform.erased.shape == tiered.erased.shape

    mask = sample_erasure_mask(1, 96, 0.05, torch.Generator().manual_seed(4))
    channel = BernoulliPacketErasureChannel(0.05)
    uniform_out = decode_blocks(channel(uniform, erasure_mask=mask).received, UNIFORM_RS)
    tiered_out = decode_blocks(channel(tiered, erasure_mask=mask).received, IMPORTANCE_AWARE_RS, order)
    assert torch.equal(uniform_out.indices, indices) and not uniform_out.failed.any()
    assert torch.equal(tiered_out.indices, indices) and not tiered_out.failed.any()
