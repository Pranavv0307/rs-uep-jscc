"""
AWGN channel layer: eta_n(z) = z + n, n ~ CN(0, sigma^2 I_k).
AWGN only this week — erasure channel arrives Week 7.
"""
import torch
import torch.nn as nn


class AWGNChannel(nn.Module):
    def __init__(self, snr_db: float):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        snr_linear = 10 ** (self.snr_db / 10)
        signal_power = 1.0
        noise_power = signal_power / snr_linear
        noise_std = (noise_power / 2) ** 0.5
        noise = torch.randn_like(z) * noise_std
        return z + noise
