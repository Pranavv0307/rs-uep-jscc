"""Automated ADJSCC sanity tests. Run locally with: pytest tests/"""
import torch
from models.adjscc import ADJSCCEncoder, ADJSCCDecoder
from models.channel import AWGNChannel


def test_shapes_roundtrip_fixed_snr():
    enc = ADJSCCEncoder(k_over_n=1 / 6)
    dec = ADJSCCDecoder(k=enc.k)
    chan = AWGNChannel(snr_db=10)
    x = torch.rand(2, 3, 32, 32)
    z = enc(x, snr_db=10)
    x_hat = dec(chan(z, snr_db=10), snr_db=10)
    assert x_hat.shape == x.shape


def test_shapes_roundtrip_per_sample_snr():
    enc = ADJSCCEncoder(k_over_n=1 / 6)
    dec = ADJSCCDecoder(k=enc.k)
    chan = AWGNChannel()
    x = torch.rand(4, 3, 32, 32)
    snr_batch = torch.tensor([0.0, 5.0, 10.0, 20.0])
    z = enc(x, snr_batch)
    x_hat = dec(chan(z, snr_batch), snr_batch)
    assert x_hat.shape == x.shape


def test_power_constraint():
    enc = ADJSCCEncoder(k_over_n=1 / 6)
    x = torch.rand(8, 3, 32, 32)
    z = enc(x, snr_db=10)
    k = z.shape[1] / 2
    avg_power = (z ** 2).sum(dim=1).mean() / k
    assert abs(avg_power.item() - 1.0) < 1e-3


def test_output_range():
    enc = ADJSCCEncoder(k_over_n=1 / 6)
    dec = ADJSCCDecoder(k=enc.k)
    chan = AWGNChannel(snr_db=10)
    x = torch.rand(2, 3, 32, 32)
    x_hat = dec(chan(enc(x, snr_db=10), snr_db=10), snr_db=10)
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0


def test_attention_weights_returned_and_shaped():
    enc = ADJSCCEncoder(k_over_n=1 / 6)
    x = torch.rand(3, 3, 32, 32)
    z, attn = enc(x, snr_db=10, return_attn=True)
    assert "af5_bottleneck" in attn
    # bottleneck gate has one value per transmitted channel
    n_bottleneck_channels = z.shape[1] // (enc._out_spatial ** 2)
    assert attn["af5_bottleneck"].shape == (3, n_bottleneck_channels)


def test_attention_weights_not_uniform():
    """Loose smoke test for Week 5's 'weights are meaningfully
    differentiated, not near-uniform' check -- an untrained network with
    random init should already show some spread; if this is ever exactly
    flat, something in AFModule is wired wrong (e.g. gate not reaching
    the FC layer)."""
    enc = ADJSCCEncoder(k_over_n=1 / 6)
    x = torch.rand(16, 3, 32, 32)
    _, attn = enc(x, snr_db=10, return_attn=True)
    gate = attn["af5_bottleneck"]
    assert gate.std(dim=1).mean().item() > 1e-4