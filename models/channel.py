"""
AWGN channel layer: eta_n(z) = z + n, n ~ CN(0, sigma^2 I_k).
Erasure channel still arrives Week 7 (Phase 3) -- unchanged from Week 1.

CHANGE FOR ADJSCC (Week 4): DeepJSCC (Week 2-3) trained at one fixed
snr_db baked in at __init__. ADJSCC needs to train across a *range* of
SNRs -- and ideally a different SNR per sample within a batch, not just
per batch, so the AF modules see enough SNR diversity to learn the
conditioning. forward() now accepts an optional snr_db override
(scalar or (B,) tensor); if omitted, falls back to the value passed at
construction, so Week 1-3 call sites (fixed snr_db, no override) are
untouched.
"""
import torch
import torch.nn as nn


class AWGNChannel(nn.Module):
    def __init__(self, snr_db: float = None):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, z: torch.Tensor, snr_db=None) -> torch.Tensor:
        snr = snr_db if snr_db is not None else self.snr_db
        if snr is None:
            raise ValueError("snr_db must be given at __init__ or forward()")

        signal_power = 1.0
        if torch.is_tensor(snr):
            # per-sample SNR: (B,) -> (B, 1) so it broadcasts against z's
            # (B, k) shape regardless of k
            snr = snr.to(device=z.device, dtype=z.dtype).view(-1, 1)
            snr_linear = 10 ** (snr / 10)
        else:
            snr_linear = 10 ** (snr / 10)

        noise_power = signal_power / snr_linear
        noise_std = (noise_power / 2) ** 0.5
        noise = torch.randn_like(z) * noise_std
        return z + noise