"""
Attention Feature (AF) module, following Xu et al. 2022 (ADJSCC),
arXiv:2109.14467 -- "attention" here refers to feature-recalibration
attention (squeeze-and-excitation-style), NOT self-attention/transformer
attention. Each AF module:

  1. Global-average-pools the incoming feature map to one scalar per
     channel  ->  (B, C)
  2. Concatenates the current channel SNR (in dB) as one extra scalar
     ->  (B, C+1)
  3. Passes that through a small two-layer MLP ending in a sigmoid
     ->  per-channel gate in (0, 1)
  4. Rescales the feature map channel-wise by that gate.

This is what lets a single encoder/decoder pair work across a *range*
of SNRs instead of being trained for one fixed value: the gate learns
to suppress/boost channels differently depending on how noisy the
channel currently is.

NOTE: re-check this against Fig. 2 / Sec. III-B of the ADJSCC paper
(open the PDF, page 3-4) before treating it as byte-for-byte -- the
squeeze-pool-gate structure is standard and matches the paper's text,
but the exact hidden_dim reduction ratio isn't pinned down here.
"""
import torch
import torch.nn as nn


class AFModule(nn.Module):
    def __init__(self, num_channels: int, hidden_dim: int = None, snr_scale: float = 20.0):
        super().__init__()
        self.num_channels = num_channels
        # SNR values are divided by snr_scale before being fed to the MLP so
        # they sit roughly in the same O(1) range as the pooled features
        # (pooled feature values from a power-normalized bottleneck are
        # small too) -- purely a conditioning-numerics choice, not from the
        # paper. Revisit if training is unstable.
        self.snr_scale = snr_scale
        hidden_dim = hidden_dim or max(num_channels // 4, 4)

        self.fc = nn.Sequential(
            nn.Linear(num_channels + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_channels),
            nn.Sigmoid(),
        )

    def _prep_snr(self, snr_db, batch_size: int, device) -> torch.Tensor:
        """Accepts a python float/int (same SNR for the whole batch) or a
        (B,) / (B, 1) tensor (per-sample SNR, needed once training sweeps
        multiple SNRs per batch). Always returns shape (B, 1)."""
        if not torch.is_tensor(snr_db):
            snr_db = torch.full((batch_size, 1), float(snr_db), device=device)
        else:
            snr_db = snr_db.to(device=device, dtype=torch.float32).view(batch_size, 1)
        return snr_db / self.snr_scale

    def forward(self, x: torch.Tensor, snr_db):
        """
        x: (B, C, H, W)
        snr_db: python scalar, or tensor broadcastable to (B,)
        Returns: (gated_x, gate) where gate has shape (B, C) -- gate is
        what Week 5's "extract per-channel attention weights" deliverable
        reads from.
        """
        B, C, H, W = x.shape
        assert C == self.num_channels, (
            f"AFModule built for {self.num_channels} channels, got {C}"
        )
        pooled = x.mean(dim=[2, 3])                      # (B, C)
        snr_norm = self._prep_snr(snr_db, B, x.device)    # (B, 1)
        context = torch.cat([pooled, snr_norm], dim=1)    # (B, C+1)
        gate = self.fc(context)                           # (B, C)
        gated = x * gate.view(B, C, 1, 1)
        return gated, gate