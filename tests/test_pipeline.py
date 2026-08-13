"""Automated pipeline sanity tests. Run locally with: pytest tests/"""
import torch
from models.deepjscc import DeepJSCCEncoder, DeepJSCCDecoder
from models.channel import AWGNChannel


def test_shapes_roundtrip():
    enc = DeepJSCCEncoder(k_over_n=1 / 6)
    dec = DeepJSCCDecoder(k=enc.k)
    chan = AWGNChannel(snr_db=10)
    x = torch.rand(2, 3, 32, 32)
    x_hat = dec(chan(enc(x)))
    assert x_hat.shape == x.shape


def test_power_constraint():
    enc = DeepJSCCEncoder(k_over_n=1 / 6)
    x = torch.rand(8, 3, 32, 32)
    z = enc(x)
    k = z.shape[1] / 2
    avg_power = (z ** 2).sum(dim=1).mean() / k
    assert abs(avg_power.item() - 1.0) < 1e-3


def test_output_range():
    enc = DeepJSCCEncoder(k_over_n=1 / 6)
    dec = DeepJSCCDecoder(k=enc.k)
    chan = AWGNChannel(snr_db=10)
    x = torch.rand(2, 3, 32, 32)
    x_hat = dec(chan(enc(x)))
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0
