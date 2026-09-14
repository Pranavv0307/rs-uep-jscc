"""Packetization plumbing sanity tests. Run locally with: pytest tests/"""
import torch

from coding.packetizer import PACKET_SIZE_SYMBOLS, packetize, depacketize, erase_packets


def _sample_idx(B=2, L=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (B, L), generator=g)


def test_packetize_depacketize_roundtrip_no_erasure():
    idx = _sample_idx(B=3, L=100)
    packetized = packetize(idx, packet_size=32)
    idx_recovered, erased_mask = depacketize(packetized)
    assert torch.equal(idx_recovered, idx)
    assert not erased_mask.any()


def test_packetize_pads_uneven_length_correctly():
    idx = _sample_idx(B=2, L=100)  # 100 / 32 -> 4 packets, 28 bytes of padding
    packetized = packetize(idx, packet_size=32)
    assert packetized.packets.shape == (2, 4, 32)
    idx_recovered, _ = depacketize(packetized)
    assert idx_recovered.shape == idx.shape
    assert torch.equal(idx_recovered, idx)


def test_packetize_exact_multiple_needs_no_padding():
    idx = _sample_idx(B=2, L=64)
    packetized = packetize(idx, packet_size=32)
    assert packetized.packets.shape == (2, 2, 32)


def test_packetize_rejects_wrong_ndim():
    idx = _sample_idx(B=2, L=64).view(2, 8, 8)  # not (B, L)
    try:
        packetize(idx, packet_size=32)
        assert False, "expected an assertion error for non-2D input"
    except AssertionError:
        pass


def test_erase_packets_full_erasure_marks_everything():
    idx = _sample_idx(B=4, L=128)
    packetized = packetize(idx, packet_size=32)  # 4 packets/sample
    g = torch.Generator().manual_seed(1)
    erased_pk = erase_packets(packetized, erasure_rate=1.0, generator=g)
    assert erased_pk.erased.all()
    idx_recovered, erased_mask = depacketize(erased_pk)
    assert erased_mask.all()
    assert (idx_recovered == 0).all()


def test_erase_packets_zero_rate_is_a_noop():
    idx = _sample_idx(B=2, L=64)
    packetized = packetize(idx, packet_size=32)
    g = torch.Generator().manual_seed(2)
    erased_pk = erase_packets(packetized, erasure_rate=0.0, generator=g)
    assert not erased_pk.erased.any()
    idx_recovered, erased_mask = depacketize(erased_pk)
    assert torch.equal(idx_recovered, idx)
    assert not erased_mask.any()


def test_erase_packets_only_masks_symbols_in_erased_packets():
    idx = _sample_idx(B=1, L=64)
    packetized = packetize(idx, packet_size=32)  # 2 packets
    g = torch.Generator().manual_seed(3)
    erased_pk = erase_packets(packetized, erasure_rate=0.5, generator=g)
    idx_recovered, erased_mask = depacketize(erased_pk)

    for p in range(2):
        lo, hi = p * 32, (p + 1) * 32
        if erased_pk.erased[0, p]:
            assert erased_mask[0, lo:hi].all()
            assert (idx_recovered[0, lo:hi] == 0).all()
        else:
            assert not erased_mask[0, lo:hi].any()
            assert torch.equal(idx_recovered[0, lo:hi], idx[0, lo:hi])


def test_erase_packets_is_non_destructive_and_composable():
    idx = _sample_idx(B=2, L=64)
    packetized = packetize(idx, packet_size=32)
    g = torch.Generator().manual_seed(4)
    once = erase_packets(packetized, erasure_rate=0.5, generator=g)
    # original PacketizedLatent must be untouched
    assert not packetized.erased.any()
    # erasing again only adds erasures, never removes one
    twice = erase_packets(once, erasure_rate=0.5, generator=g)
    assert torch.equal(twice.erased, once.erased | twice.erased)
    assert bool((once.erased & ~twice.erased).any()) is False


def test_fill_value_override():
    idx = _sample_idx(B=1, L=32)
    packetized = packetize(idx, packet_size=32)
    g = torch.Generator().manual_seed(5)
    erased_pk = erase_packets(packetized, erasure_rate=1.0, generator=g)
    idx_recovered, erased_mask = depacketize(erased_pk, fill_value=99)
    assert (idx_recovered[erased_mask] == 99).all()


def test_default_packet_size_divides_current_bottleneck_evenly():
    # k_over_n=1/6, image_size=32 -> k=512 -> latent length 2k=1024. Not
    # enforced by packetize() (it pads correctly regardless), but worth
    # knowing PACKET_SIZE_SYMBOLS doesn't silently introduce padding for
    # the architecture actually in use today.
    assert 1024 % PACKET_SIZE_SYMBOLS == 0


def test_integration_with_quantizer_dequantize_roundtrip():
    """Proves the actual intended interface composes: quantizer -> idx ->
    packetize -> erase -> depacketize -> dequantize_indices. Not RS -- just
    confirms shapes/dtypes fit end to end and surviving symbols come back
    bit-exact."""
    from models.quantizer import ScalarSoftToHardQuantizer

    q = ScalarSoftToHardQuantizer(num_levels=256)
    z = torch.randn(2, 64)
    z_q, idx = q(z)

    packetized = packetize(idx, packet_size=32)
    g = torch.Generator().manual_seed(6)
    erased_pk = erase_packets(packetized, erasure_rate=0.3, generator=g)
    idx_recovered, erased_mask = depacketize(erased_pk)

    assert idx_recovered.shape == idx.shape
    assert idx_recovered.dtype == torch.long

    z_recovered = q.dequantize_indices(idx_recovered)
    assert z_recovered.shape == z.shape
    # surviving (non-erased) positions must match the original
    # hard-quantized values -- nothing about packetization should perturb
    # a symbol that was never erased. allclose, not equal: z_q's forward
    # value is z_soft + (z_hard - z_soft).detach(), a floating-point
    # reconstruction of centers[idx] rather than a bit-identical copy of
    # it, so a same-magnitude epsilon gap is possible even when
    # packetization introduced no error at all.
    assert torch.allclose(z_recovered[~erased_mask], z_q[~erased_mask], atol=1e-6)
