"""Masked latent inpainting head (Track B, Step B.2).

When a Reed-Solomon block cannot be recovered, the receiver knows exactly which
latent symbols are missing (the ``failed`` mask). Instead of feeding zeros to
the decoder, this head lets the surviving channels vote on what the missing
symbols should be:

* each channel is a token of ``D`` symbols (D = 64 for an 8x8 latent grid);
* failed symbols are replaced by a learnable mask token and the failure mask is
  concatenated to the token, so the network knows what is missing;
* ``layers`` transformer-encoder blocks (multi-head self-attention across the
  channel tokens) mix information between channels;
* received symbols are **never changed**: the output only overwrites symbols
  that were actually lost (RS is systematic, so what arrived is exact).

It sits between ``dequantize_with_fallback`` and the decoder, and uses no
information the receiver does not have: the failure mask is the receiver's own
RS decoding outcome (roadmap Rule 2).
"""
import torch
import torch.nn as nn


class LatentInpainter(nn.Module):
    def __init__(self, channels: int = 16, symbols_per_channel: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        if symbols_per_channel % heads:
            raise ValueError("symbols_per_channel must be divisible by the number of heads")
        self.channels = channels
        self.symbols_per_channel = symbols_per_channel
        dim = symbols_per_channel
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.embed = nn.Linear(2 * dim, dim)
        self.position = nn.Parameter(torch.randn(1, channels, dim) * 0.02)
        block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=4 * dim, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, latent: torch.Tensor, failed: torch.Tensor) -> torch.Tensor:
        """``latent`` (B, C*D) floats with zeros where lost; ``failed`` (B, C*D) bool."""
        batch = latent.shape[0]
        values = latent.view(batch, self.channels, self.symbols_per_channel)
        lost = failed.view(batch, self.channels, self.symbols_per_channel)
        lost_f = lost.to(values.dtype)
        filled = values * (1.0 - lost_f) + self.mask_token.view(1, 1, -1) * lost_f
        tokens = self.embed(torch.cat([filled, lost_f], dim=-1)) + self.position
        predicted = self.out(self.encoder(tokens))
        return torch.where(lost, predicted, values).reshape(batch, -1)
