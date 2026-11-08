"""1-D speech JSCC-Q (speech study S1): the image models' design ported to waveforms.

Two architectures, registered in ``models/registry.py``:

* ``adjsccq1d``  -- conv backbone with an AF attention module after every layer (as ADJSCC-Q),
* ``deepjsccq1d`` -- the same backbone without attention (as DeepJSCC-Q, the G1 ablation),

so the speech study crosses modality with the attention ablation instead of
confounding them (``docs/02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md``, AD-14 to AD-17).

Geometry (defaults)
-------------------
A 16 kHz mono segment of 16,384 samples (1.024 s) is encoded by five Conv1d layers
(kernel 9, strides 4,4,4,4 then kernel 5, stride 1; widths 16,32,32,32,C -- the
image backbone's widths) into ``C = 16`` channels x ``T = 64`` steps = **1024
latent values**, power-normalised exactly as the image models and quantised by the
same 256-level codebook. That is 1024 bytes per segment (8 kbit/s), the same
16 x 64 layout as the CIFAR models, so every Reed-Solomon scheme, packet mask and
the side-info rule apply unchanged. The decoder mirrors it with ConvTranspose1d
(kernel 8, stride 4, padding 2 doubles exactly x4) and a ``tanh`` output for
waveforms in [-1, 1].

Interface
---------
Identical to ADJSCC-Q (``encoder(x, snr_db, return_attn, return_indices)``,
``encoder.encoder``, ``encoder.quantizer``, ``encoder.k``, ``num_channels``,
``has_attention``, ``calibrate_quantizer``; ``decoder(z, snr_db, return_attn, spatial_hw)``)
plus ``symbols_per_channel`` and ``segment_length``. Inputs are ``(B, 1, L)``.
``spatial_hw`` is accepted and ignored: the number of latent steps follows from the
latent length. The attention variant uses the unmodified ``models/Attention.AFModule``
on a ``(B, C, L, 1)`` view, so its gate is the same squeeze-and-excitation gate.
"""
from typing import Sequence

import torch
import torch.nn as nn

from models.Attention import AFModule
from models.adjsccq import NOMINAL_SNR_DB_IDENTITY
from models.deepjscc import power_normalize
from models.quantizer import ScalarSoftToHardQuantizer

TOTAL_STRIDE = 256
DEFAULT_SEGMENT_LENGTH = 16384
DEFAULT_LATENT_CHANNELS = 16
DEFAULT_WIDTHS = (16, 32, 32, 32)


def _gate(af: AFModule, x: torch.Tensor, snr_db):
    """Apply a 2-D AF module to a 1-D feature map through a trailing singleton axis."""
    gated, gate = af(x.unsqueeze(-1), snr_db)
    return gated.squeeze(-1), gate


def _check_geometry(segment_length: int, widths: Sequence[int]):
    if segment_length % TOTAL_STRIDE:
        raise ValueError(f"segment_length must be a multiple of {TOTAL_STRIDE}, got {segment_length}")
    if len(widths) != 4:
        raise ValueError(f"widths must have 4 entries (hidden layers), got {widths}")


class SpeechJSCCEncoder(nn.Module):
    def __init__(self, segment_length: int = DEFAULT_SEGMENT_LENGTH, latent_channels: int = DEFAULT_LATENT_CHANNELS,
                 attention: bool = True, widths: Sequence[int] = DEFAULT_WIDTHS):
        super().__init__()
        _check_geometry(segment_length, widths)
        self.segment_length = segment_length
        self.latent_steps = segment_length // TOTAL_STRIDE
        self.attention = attention
        # 2k real latent values, as in the image models (power_normalize uses k = length / 2).
        self.k = latent_channels * self.latent_steps // 2
        w0, w1, w2, w3 = widths
        layers = [(1, w0, 9, 4, 4), (w0, w1, 9, 4, 4), (w1, w2, 9, 4, 4), (w2, w3, 9, 4, 4),
                  (w3, latent_channels, 5, 1, 2)]
        self.convs = nn.ModuleList(nn.Conv1d(i, o, k, stride=s, padding=p) for i, o, k, s, p in layers)
        self.afs = nn.ModuleList(AFModule(o) for _, o, *_ in layers) if attention else None
        self.acts = nn.ModuleList(nn.PReLU() for _ in range(4))  # no activation after the bottleneck

    @property
    def has_attention(self) -> bool:
        return self.attention

    @property
    def num_channels(self) -> int:
        return self.convs[-1].out_channels

    def forward(self, x: torch.Tensor, snr_db=NOMINAL_SNR_DB_IDENTITY, return_attn: bool = False):
        snr_db = NOMINAL_SNR_DB_IDENTITY if snr_db is None else snr_db
        attn = {}
        h = x
        for index, conv in enumerate(self.convs):
            h = conv(h)
            if self.afs is not None:
                h, gate = _gate(self.afs[index], h, snr_db)
                if return_attn:
                    attn["af5_bottleneck" if index == 4 else f"af{index + 1}"] = gate.detach()
            if index < 4:
                h = self.acts[index](h)
        z = power_normalize(h.reshape(h.shape[0], -1), P=1.0)  # channel-major, like the image models
        if return_attn:
            return z, attn
        return z


class SpeechJSCCDecoder(nn.Module):
    def __init__(self, segment_length: int = DEFAULT_SEGMENT_LENGTH, latent_channels: int = DEFAULT_LATENT_CHANNELS,
                 attention: bool = True, widths: Sequence[int] = DEFAULT_WIDTHS):
        super().__init__()
        _check_geometry(segment_length, widths)
        self.segment_length = segment_length
        self.latent_channels = latent_channels
        self.attention = attention
        w0, w1, w2, w3 = widths
        layers = [(latent_channels, w3, 5, 1, 2), (w3, w2, 8, 4, 2), (w2, w1, 8, 4, 2), (w1, w0, 8, 4, 2),
                  (w0, 1, 8, 4, 2)]
        self.deconvs = nn.ModuleList(nn.ConvTranspose1d(i, o, k, stride=s, padding=p) for i, o, k, s, p in layers)
        self.afs = nn.ModuleList(AFModule(o) for _, o, *_ in layers) if attention else None
        self.acts = nn.ModuleList(nn.PReLU() for _ in range(4))
        self.out_act = nn.Tanh()

    @property
    def has_attention(self) -> bool:
        return self.attention

    def forward(self, z_hat_flat: torch.Tensor, snr_db=NOMINAL_SNR_DB_IDENTITY, return_attn: bool = False,
                spatial_hw=None):
        snr_db = NOMINAL_SNR_DB_IDENTITY if snr_db is None else snr_db
        batch = z_hat_flat.shape[0]
        steps = z_hat_flat.shape[1] // self.latent_channels
        h = z_hat_flat.reshape(batch, self.latent_channels, steps)
        attn = {}
        for index, deconv in enumerate(self.deconvs):
            h = deconv(h)
            if self.afs is not None:
                h, gate = _gate(self.afs[index], h, snr_db)
                if return_attn:
                    attn[f"af{index + 1}"] = gate.detach()
            if index < 4:
                h = self.acts[index](h)
        x_hat = self.out_act(h)
        if return_attn:
            return x_hat, attn
        return x_hat


class SpeechJSCCQEncoder(nn.Module):
    """Speech backbone followed by the shared 256-level quantiser (mirrors ``ADJSCCQEncoder``)."""

    def __init__(self, segment_length: int = DEFAULT_SEGMENT_LENGTH, latent_channels: int = DEFAULT_LATENT_CHANNELS,
                 attention: bool = True, widths: Sequence[int] = DEFAULT_WIDTHS, num_levels: int = 256,
                 init_range: tuple = (-3.0, 3.0), init_sigma: float = 1.0):
        super().__init__()
        self.encoder = SpeechJSCCEncoder(segment_length, latent_channels, attention, widths)
        self.k = self.encoder.k
        self.quantizer = ScalarSoftToHardQuantizer(num_levels=num_levels, init_range=init_range,
                                                   init_sigma=init_sigma)

    @property
    def has_attention(self) -> bool:
        return self.encoder.attention

    @property
    def num_channels(self) -> int:
        return self.encoder.num_channels

    @property
    def symbols_per_channel(self) -> int:
        return self.encoder.latent_steps

    @property
    def segment_length(self) -> int:
        return self.encoder.segment_length

    @torch.no_grad()
    def calibrate_quantizer(self, x: torch.Tensor, snr_db=NOMINAL_SNR_DB_IDENTITY,
                            low_pct: float = 1.0, high_pct: float = 99.0):
        z = self.encoder(x, snr_db)
        self.quantizer.calibrate(z, low_pct=low_pct, high_pct=high_pct)

    def forward(self, x: torch.Tensor, snr_db=NOMINAL_SNR_DB_IDENTITY, return_attn: bool = False,
                return_indices: bool = False):
        if return_attn:
            z, attn = self.encoder(x, snr_db, return_attn=True)
        else:
            z, attn = self.encoder(x, snr_db), None
        z_q, idx = self.quantizer(z)
        if return_attn and return_indices:
            return z_q, attn, idx
        if return_attn:
            return z_q, attn
        if return_indices:
            return z_q, idx
        return z_q


# As for ADJSCC-Q, the decoder consumes the dequantised latent directly.
SpeechJSCCQDecoder = SpeechJSCCDecoder
