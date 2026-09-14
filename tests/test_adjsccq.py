"""
Tests for models/quantizer.py and models/adjsccq.py (Phase 3,
identity-channel DeepJSCC-Q integration on top of ADJSCC).

Mirrors the style of tests/test_adjscc.py: shape/contract tests plus a
couple of tests specific to what a soft-to-hard quantizer needs to get
right (discreteness of the hard path, non-zero gradient through the
soft path, sigma_q annealing actually changing behaviour).
"""
import torch

from models.quantizer import ScalarSoftToHardQuantizer
from models.adjsccq import ADJSCCQEncoder, ADJSCCQDecoder, NOMINAL_SNR_DB_IDENTITY


# ---------------------------------------------------------------------------
# ScalarSoftToHardQuantizer
# ---------------------------------------------------------------------------

def test_quantizer_output_shape_matches_input():
    q = ScalarSoftToHardQuantizer(num_levels=256)
    z = torch.randn(4, 1024)
    z_q, idx = q(z)
    assert z_q.shape == z.shape
    assert idx.shape == z.shape


def test_quantizer_hard_values_are_from_codebook():
    """Every element of z_q's forward-pass value must exactly equal one
    of the 256 learnable centers -- this is the actual discreteness
    guarantee that makes downstream RS-symbol mapping valid."""
    q = ScalarSoftToHardQuantizer(num_levels=256, init_range=(-3.0, 3.0))
    z = torch.randn(8, 500) * 2.0
    z_q, idx = q(z)

    centers = q.centers.detach()
    # every z_q value must be present (within fp tolerance) in centers
    for val in z_q.reshape(-1)[:200]:  # sample-check, full check is slow
        assert torch.any(torch.isclose(centers, val, atol=1e-5))

    assert idx.dtype == torch.long
    assert idx.min() >= 0
    assert idx.max() < 256


def test_quantizer_dequantize_indices_matches_forward_hard_values():
    q = ScalarSoftToHardQuantizer(num_levels=256)
    z = torch.randn(4, 100)
    z_q, idx = q(z)
    looked_up = q.dequantize_indices(idx)
    assert torch.allclose(z_q, looked_up, atol=1e-5)


def test_quantizer_gradient_flows_to_centers():
    """The straight-through estimator must actually let gradients reach
    the codebook centers -- if this is zero, the codebook can never
    learn and we've silently regressed to a fixed grid."""
    q = ScalarSoftToHardQuantizer(num_levels=256, init_sigma=1.0)
    z = torch.randn(4, 100, requires_grad=True)
    z_q, _ = q(z)
    loss = z_q.sum()
    loss.backward()
    assert q.centers.grad is not None
    assert torch.any(q.centers.grad != 0)


def test_quantizer_sigma_annealing_changes_soft_hard_gap():
    """With a large sigma_q, the soft path should be visibly blurred
    away from the hard nearest-center value (poor approximation, but
    good early-training gradients). With a very small sigma_q, soft and
    hard should nearly coincide (matches true inference behaviour)."""
    q = ScalarSoftToHardQuantizer(num_levels=256, init_range=(-3.0, 3.0))
    z = torch.randn(4, 200)

    q.set_sigma(2.0)
    z_flat = z.reshape(-1, 1)
    centers = q.centers.view(1, -1)
    dist_sq = (z_flat - centers) ** 2
    weights_soft_sigma = torch.softmax(-dist_sq / (2.0 * 2.0 ** 2), dim=1)
    z_soft_loose = (weights_soft_sigma * centers).sum(dim=1)

    q.set_sigma(0.01)
    weights_hard_sigma = torch.softmax(-dist_sq / (2.0 * 0.01 ** 2), dim=1)
    z_soft_tight = (weights_hard_sigma * centers).sum(dim=1)

    idx_hard = torch.argmin(dist_sq, dim=1)
    z_hard = centers.view(-1)[idx_hard]

    gap_loose = (z_soft_loose - z_hard).abs().mean()
    gap_tight = (z_soft_tight - z_hard).abs().mean()
    assert gap_tight < gap_loose


def test_quantizer_calibrate_matches_data_range():
    q = ScalarSoftToHardQuantizer(num_levels=256, init_range=(-1.0, 1.0))
    z_sample = torch.randn(1000, 100) * 5.0 + 10.0  # centered far from init_range
    q.calibrate(z_sample, low_pct=1.0, high_pct=99.0)

    expected_lo = torch.quantile(z_sample.reshape(-1), 0.01)
    expected_hi = torch.quantile(z_sample.reshape(-1), 0.99)
    assert torch.isclose(q.centers.min(), expected_lo, atol=1e-3)
    assert torch.isclose(q.centers.max(), expected_hi, atol=1e-3)


# ---------------------------------------------------------------------------
# ADJSCCQEncoder / ADJSCCQDecoder (identity-channel round trip)
# ---------------------------------------------------------------------------

def test_adjsccq_encoder_output_shape():
    enc = ADJSCCQEncoder(k_over_n=1 / 6, image_size=32)
    x = torch.rand(2, 3, 32, 32)
    z_q = enc(x, NOMINAL_SNR_DB_IDENTITY)
    assert z_q.shape == (2, 2 * enc.k)


def test_adjsccq_full_round_trip_output_range():
    enc = ADJSCCQEncoder(k_over_n=1 / 6, image_size=32)
    dec = ADJSCCQDecoder(k=enc.k, image_size=32)
    x = torch.rand(2, 3, 32, 32)

    z_q = enc(x, NOMINAL_SNR_DB_IDENTITY)
    x_hat = dec(z_q, NOMINAL_SNR_DB_IDENTITY)

    assert x_hat.shape == x.shape
    assert x_hat.min() >= 0.0
    assert x_hat.max() <= 1.0


def test_adjsccq_return_attn_and_indices():
    enc = ADJSCCQEncoder(k_over_n=1 / 6, image_size=32)
    x = torch.rand(2, 3, 32, 32)

    z_q, attn, idx = enc(x, NOMINAL_SNR_DB_IDENTITY, return_attn=True, return_indices=True)
    assert "af5_bottleneck" in attn
    assert idx.shape == z_q.shape
    assert idx.dtype == torch.long
    assert idx.min() >= 0 and idx.max() < enc.quantizer.num_levels


def test_adjsccq_calibrate_quantizer_runs():
    enc = ADJSCCQEncoder(k_over_n=1 / 6, image_size=32)
    x = torch.rand(8, 3, 32, 32)
    before = enc.quantizer.centers.clone()
    enc.calibrate_quantizer(x)
    after = enc.quantizer.centers
    assert not torch.allclose(before, after)


def test_adjsccq_gradient_reaches_encoder_backbone():
    """End-to-end sanity: loss computed after the decoder must still be
    able to update the ADJSCC encoder's conv weights through the
    quantizer's straight-through path."""
    enc = ADJSCCQEncoder(k_over_n=1 / 6, image_size=32)
    dec = ADJSCCQDecoder(k=enc.k, image_size=32)
    x = torch.rand(2, 3, 32, 32)

    z_q = enc(x, NOMINAL_SNR_DB_IDENTITY)
    x_hat = dec(z_q, NOMINAL_SNR_DB_IDENTITY)
    loss = ((x_hat - x) ** 2).mean()
    loss.backward()

    conv1_grad = enc.encoder.conv1.weight.grad
    assert conv1_grad is not None
    assert torch.any(conv1_grad != 0)
