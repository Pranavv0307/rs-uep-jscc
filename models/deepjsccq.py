"""DeepJSCC-Q: the plain DeepJSCC backbone with the ADJSCC-Q quantiser (G1 attention ablation).

This is ADJSCC-Q (``models/adjsccq.py``) with the ten AF attention modules
(five in the encoder, five in the decoder) removed and nothing else changed:

* the same convolutions (kernel 5; encoder strides 2,2,1,1,1 and widths
  16,32,32,32,C; decoder strides 1,1,1,2,2 and widths 32,32,32,16,3),
* the same PReLU activations, power normalisation and sigmoid output,
* the same learnable 256-level scalar quantiser (one byte per latent value),
* the same latent shape: 16 channels x 8 x 8 = 1024 bytes at k/n = 1/6,

so every Reed-Solomon scheme, packet mask and evaluation script applies
unchanged. See ``docs/02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md`` (AD-02 to AD-04).

Interface parity with ADJSCC-Q
------------------------------
The evaluation code calls models with the ADJSCC-Q signature::

    encoder(x, snr_db, return_attn=..., return_indices=...)
    encoder.encoder(x, snr_db, return_attn=...)          # backbone only
    encoder.calibrate_quantizer(x, snr_db=..., ...)
    decoder(z, snr_db, return_attn=..., spatial_hw=...)

The classes below accept exactly these arguments. ``snr_db`` is accepted and
ignored (there is no SNR-conditioned layer), and the attention dictionary is
always empty. ``has_attention = False`` lets callers refuse the ``attention``
importance source with a clear message instead of a ``KeyError``.

``models/deepjscc.py`` is not edited (it stays the frozen Week-2/3 reference).
The backbone classes are *subclassed*, so their parameter names (``net.0.weight``
...) and initialisation are exactly those of the original DeepJSCC classes.
"""
import torch
import torch.nn as nn

from models.deepjscc import DeepJSCCDecoder, DeepJSCCEncoder
from models.quantizer import ScalarSoftToHardQuantizer


class DeepJSCCBackbone(DeepJSCCEncoder):
    """``DeepJSCCEncoder`` with the ``ADJSCCEncoder`` call signature (``snr_db`` ignored)."""

    has_attention = False

    @property
    def num_channels(self) -> int:
        """Latent channels = output channels of the bottleneck convolution."""
        return self.net[-1].out_channels

    def forward(self, x: torch.Tensor, snr_db=None, return_attn: bool = False):
        z = super().forward(x)
        if return_attn:
            return z, {}
        return z


class DeepJSCCQEncoder(nn.Module):
    """DeepJSCC backbone followed by ``ScalarSoftToHardQuantizer``.

    Mirrors ``ADJSCCQEncoder`` line for line: same constructor arguments, same
    ``calibrate_quantizer``, same return conventions of ``forward``.
    """

    has_attention = False

    def __init__(self, k_over_n: float = 1 / 6, image_size: int = 32,
                 num_levels: int = 256, init_range: tuple = (-3.0, 3.0),
                 init_sigma: float = 1.0):
        super().__init__()
        self.encoder = DeepJSCCBackbone(k_over_n=k_over_n, image_size=image_size)
        self.k = self.encoder.k
        self.quantizer = ScalarSoftToHardQuantizer(
            num_levels=num_levels, init_range=init_range, init_sigma=init_sigma
        )

    @property
    def num_channels(self) -> int:
        return self.encoder.num_channels

    @torch.no_grad()
    def calibrate_quantizer(self, x: torch.Tensor, snr_db=None,
                            low_pct: float = 1.0, high_pct: float = 99.0):
        """Re-centre the codebook on the empirical range of the unquantised latent
        (same procedure as ``ADJSCCQEncoder.calibrate_quantizer``)."""
        z = self.encoder(x)
        self.quantizer.calibrate(z, low_pct=low_pct, high_pct=high_pct)

    def forward(self, x: torch.Tensor, snr_db=None, return_attn: bool = False,
                return_indices: bool = False):
        z = self.encoder(x)
        z_q, idx = self.quantizer(z)
        attn = {}
        if return_attn and return_indices:
            return z_q, attn, idx
        if return_attn:
            return z_q, attn
        if return_indices:
            return z_q, idx
        return z_q


class DeepJSCCQDecoder(DeepJSCCDecoder):
    """``DeepJSCCDecoder`` with the ``ADJSCCDecoder`` call signature.

    Adds ``spatial_hw`` (the original class hard-codes the square training grid)
    so the fully convolutional decoder can also run at other resolutions, e.g.
    Kodak, exactly like ``ADJSCCDecoder``.
    """

    has_attention = False

    def forward(self, z_hat_flat: torch.Tensor, snr_db=None, return_attn: bool = False,
                spatial_hw=None):
        batch = z_hat_flat.shape[0]
        height, width = spatial_hw if spatial_hw is not None else (self.spatial, self.spatial)
        channels = z_hat_flat.shape[1] // (height * width)
        x_hat = self.net(z_hat_flat.view(batch, channels, height, width))
        if return_attn:
            return x_hat, {}
        return x_hat
