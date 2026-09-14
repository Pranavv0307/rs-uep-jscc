"""
Scalar soft-to-hard quantizer, following the differentiable
soft-to-hard mechanism used by DeepJSCC-Q (Tung, Gunduz & Simeone 2022,
arXiv:2111.13042), adapted from DeepJSCC-Q's 2D QAM constellation to a
1D scalar codebook.

Why 1D instead of 2D: DeepJSCC-Q maps latent values to points on a
learned complex-plane constellation (I/Q pairs) because its downstream
channel is a literal wireless modulation scheme. This project's
downstream stage is Reed-Solomon over GF(2^8), which consumes
one-dimensional byte symbols, not complex points. Collapsing the
codebook to 1D means each quantized latent scalar maps directly to one
RS symbol (one byte, num_levels=256), with no bit-packing or I/Q
splitting step needed at the RS interface. This is a deliberate
simplification relative to the source paper, not an oversight -- see
PROJECT_CONTEXT.md Phase 3 notes.

Design decisions baked into this module (confirmed in the Week 3
planning conversation, see PROJECT_CONTEXT.md):
  - num_levels=256 so each scalar maps 1:1 onto a GF(2^8) RS symbol.
  - ONE codebook, shared across every latent dimension -- mirrors how
    a QAM constellation is one fixed point-set reused for every
    transmitted symbol, not a per-dimension codebook. This also keeps
    the parameter count trivial (256 scalars total, regardless of k).
  - Codebook centers are LEARNABLE (nn.Parameter), matching DeepJSCC-Q's
    actual contribution (a learned constellation, not a fixed uniform
    grid).
  - The softness/temperature parameter sigma_q is intentionally NOT a
    learnable nn.Parameter. It is a plain buffer that an external
    training loop anneals on a schedule (soft early in training so
    gradients flow well; hard by the end of training so the train-time
    forward pass matches the true hard-quantized inference behaviour).
    Making sigma_q itself learnable would let gradient descent cheat by
    just keeping it soft forever, which defeats the point.

NOT yet implemented here (explicitly out of scope for this file):
  - Any channel/corruption model. This module only quantizes and
    dequantizes; it does not simulate AWGN, erasure, or RS
    encoding/decoding. See models/adjsccq.py's module docstring for the
    identity-channel decision made for this phase.
  - Byte packing / GF(2^8) arithmetic. `idx` returned by forward() is a
    plain torch.long tensor in [0, 256); turning that into actual RS
    codewords is Phase 5 (coding/) work, not this module's job.
"""
import torch
import torch.nn as nn


class ScalarSoftToHardQuantizer(nn.Module):
    """
    Learnable 1D scalar codebook with a differentiable soft-to-hard
    quantizer and a straight-through gradient estimator.

    Forward pass numerically returns the HARD (nearest-center)
    quantized value -- i.e. exactly what would actually be transmitted
    -- while gradients flow backward through the SOFT (softmax-weighted)
    assignment, so the codebook centers and the upstream encoder remain
    trainable despite the hard argmin being non-differentiable.
    """

    def __init__(self, num_levels: int = 256, init_range: tuple = (-3.0, 3.0),
                 init_sigma: float = 1.0):
        super().__init__()
        self.num_levels = num_levels

        centers_init = torch.linspace(float(init_range[0]), float(init_range[1]), num_levels)
        self.centers = nn.Parameter(centers_init)

        # Buffer, not Parameter: saved/loaded with the checkpoint (so a
        # reloaded model remembers what hardness it was last trained at)
        # but never receives a gradient and is never touched by the
        # optimizer -- only by set_sigma(), called externally by the
        # training loop's annealing schedule.
        self.register_buffer("sigma_q", torch.tensor(float(init_sigma)))

    @torch.no_grad()
    def calibrate(self, z_sample: torch.Tensor, low_pct: float = 1.0, high_pct: float = 99.0):
        """
        Re-initialize codebook centers from the empirical range of a
        representative sample of UNQUANTIZED encoder output z, instead
        of the constructor's fixed init_range guess.

        Call this once, before training starts, on one or a few
        calibration batches pushed through the (freshly-initialized)
        encoder with quantization bypassed. This matters because
        power_normalize's constraint (Eq. 1 in Bourtsoulatze et al.) is
        a GLOBAL L2-norm constraint over the whole 2k-length vector, not
        a per-element range constraint -- so the actual per-scalar range
        of z depends on k, P, and how the encoder happens to distribute
        energy across dimensions, and is not something we should just
        guess at with a hard-coded [-3, 3].

        Percentiles (default 1st/99th) are used rather than raw
        min/max so that a handful of outlier latent values don't stretch
        the whole codebook and waste levels on rarely-used extremes.
        """
        flat = z_sample.detach().reshape(-1).to(self.centers.device)
        lo = torch.quantile(flat, low_pct / 100.0)
        hi = torch.quantile(flat, high_pct / 100.0)
        new_centers = torch.linspace(lo.item(), hi.item(), self.num_levels, device=self.centers.device)
        self.centers.copy_(new_centers)

    @torch.no_grad()
    def set_sigma(self, sigma: float):
        """Called once per epoch (or per step) by the training loop's
        annealing schedule. sigma_q is the softmax temperature over
        negative squared distance to each center: large sigma_q ->
        soft/blurry assignment (good early-training gradients), small
        sigma_q -> assignment approaches a hard argmin (matches true
        inference behaviour)."""
        self.sigma_q.fill_(float(sigma))

    def forward(self, z: torch.Tensor):
        """
        z: arbitrary-shaped tensor of continuous latent values.

        Returns:
          z_q:  same shape as z. Forward-pass values equal the HARD
                (nearest-center) quantized values -- exactly what a
                real system would transmit as an RS symbol and
                dequantize on the other end. Gradients flowing into z_q
                during backward() are the SOFT assignment's gradients
                (straight-through estimator).
          idx:  same shape as z, dtype torch.long, values in
                [0, num_levels). This is the hard nearest-center index
                for each latent scalar -- i.e. the future RS symbol byte
                value. Detached from the graph (argmin is
                non-differentiable and idx is for export/inspection, not
                further backprop).
        """
        orig_shape = z.shape
        z_flat = z.reshape(-1, 1)                                       # (N, 1)
        centers = self.centers.view(1, -1)                              # (1, L)
        dist_sq = (z_flat - centers) ** 2                               # (N, L)

        # Hard assignment: this is what is actually "sent over the wire".
        idx_flat = torch.argmin(dist_sq, dim=1)                         # (N,)
        z_hard_flat = centers.view(-1)[idx_flat]                        # (N,)

        # Soft assignment: this is what gradients actually flow through.
        sigma = self.sigma_q.clamp_min(1e-4)
        weights = torch.softmax(-dist_sq / (2.0 * sigma ** 2), dim=1)   # (N, L)
        z_soft_flat = (weights * centers).sum(dim=1)                    # (N,)

        # Straight-through estimator: forward value = hard, backward
        # gradient = soft's gradient. The (z_hard - z_soft).detach() term
        # numerically corrects z_soft up to z_hard on the forward pass
        # while contributing exactly zero gradient on the backward pass.
        z_q_flat = z_soft_flat + (z_hard_flat - z_soft_flat).detach()

        z_q = z_q_flat.view(orig_shape)
        idx = idx_flat.view(orig_shape).detach()
        return z_q, idx

    def dequantize_indices(self, idx: torch.Tensor) -> torch.Tensor:
        """Look up center values from hard indices. This is the function
        Phase 5's RS-decoding path will call after recovering (or
        imputing, for symbols RS failed to recover) byte-valued symbols,
        to turn them back into decoder-ready floats. Provided now so the
        quantizer's public interface is already the one Phase 5 will
        need, even though nothing calls this yet."""
        return self.centers[idx]
