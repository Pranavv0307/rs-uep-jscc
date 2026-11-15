"""1-D text JSCC-Q (text study T1): the image / speech design ported to character sequences.

Two architectures, registered in ``models/registry.py``:

* ``adjsccq_text``   -- conv backbone with an AF attention module after every layer (as ADJSCC-Q),
* ``deepjsccq_text`` -- the same backbone without attention (as DeepJSCC-Q, the G1 ablation),

so the text study crosses modality with the attention ablation instead of confounding
them, exactly like the speech study (``docs/02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md``, AD-22 to AD-28).

Geometry (defaults)
-------------------
A fixed window of ``L = 2048`` characters from text8 (27 symbols: space + a-z) is
embedded (``embed_dim = 32``) and encoded by five Conv1d layers (strides 4,2,2,2 with
kernel ``2s + 1``, then kernel 5 stride 1; widths 32,64,64,64,C) into ``C = 16``
channels x ``T = 64`` steps = **1024 latent values**, power-normalised exactly as the
image and speech models and quantised by the same 256-level codebook. That is 1024 bytes
per window (4 bits per character), the same 16 x 64 layout as CIFAR and LibriSpeech, so
every Reed-Solomon scheme, packet mask and the side-info rule apply unchanged. The
decoder mirrors it with ConvTranspose1d (kernel ``2s``, stride ``s``, padding ``s/2``:
exactly x``s``) and ends in **per-position logits over the vocabulary**, shape
``(B, V, L)``; the reconstruction is their argmax. No softmax/tanh is applied in the model:
training uses cross-entropy on the logits.

Interface
---------
Identical to ADJSCC-Q / the speech models (``encoder(x, snr_db, return_attn, return_indices)``,
``encoder.encoder``, ``encoder.quantizer``, ``encoder.k``, ``num_channels``,
``has_attention``, ``calibrate_quantizer``, ``symbols_per_channel``, ``segment_length``;
``decoder(z, snr_db, return_attn, spatial_hw)``). Inputs are integer symbol ids
``(B, L)`` (any integer dtype; cast to ``long`` for the embedding). The attention variant
uses the unmodified ``models/Attention.AFModule`` on a ``(B, C, L, 1)`` view, so its gate
is the same squeeze-and-excitation gate as in every other model.
"""
from math import prod
from typing import Sequence

import torch
import torch.nn as nn

from models.Attention import AFModule
from models.adjsccq import NOMINAL_SNR_DB_IDENTITY
from models.deepjscc import power_normalize
from models.quantizer import ScalarSoftToHardQuantizer

DEFAULT_SEGMENT_LENGTH = 2048
DEFAULT_LATENT_CHANNELS = 16
DEFAULT_WIDTHS = (32, 64, 64, 64)
DEFAULT_STRIDES = (4, 2, 2, 2)
DEFAULT_EMBED_DIM = 32
DEFAULT_VOCAB_SIZE = 27   # text8: space + a-z (data/text.py, VOCAB)


def _gate(af: AFModule, x: torch.Tensor, snr_db):
    """Apply a 2-D AF module to a 1-D feature map through a trailing singleton axis."""
    gated, gate = af(x.unsqueeze(-1), snr_db)
    return gated.squeeze(-1), gate


def _check_geometry(segment_length: int, widths: Sequence[int], strides: Sequence[int]):
    if len(widths) != 4:
        raise ValueError(f"widths must have 4 entries (hidden layers), got {widths}")
    if len(strides) != 4 or any(int(s) < 2 or int(s) % 2 for s in strides):
        raise ValueError(f"strides must be 4 even integers >= 2, got {strides}")
    total = prod(int(s) for s in strides)
    if segment_length % total:
        raise ValueError(f"segment_length must be a multiple of the total stride {total}, got {segment_length}")


class TextJSCCEncoder(nn.Module):
    def __init__(self, segment_length: int = DEFAULT_SEGMENT_LENGTH, latent_channels: int = DEFAULT_LATENT_CHANNELS,
                 attention: bool = True, widths: Sequence[int] = DEFAULT_WIDTHS,
                 strides: Sequence[int] = DEFAULT_STRIDES, vocab_size: int = DEFAULT_VOCAB_SIZE,
                 embed_dim: int = DEFAULT_EMBED_DIM):
        super().__init__()
        _check_geometry(segment_length, widths, strides)
        self.segment_length = segment_length
        self.latent_steps = segment_length // prod(int(s) for s in strides)
        self.attention = attention
        self.vocab_size = vocab_size
        # 2k real latent values, as in the image models (power_normalize uses k = length / 2).
        self.k = latent_channels * self.latent_steps // 2
        self.embed = nn.Embedding(vocab_size, embed_dim)
        w0, w1, w2, w3 = widths
        s0, s1, s2, s3 = (int(s) for s in strides)
        layers = [(embed_dim, w0, 2 * s0 + 1, s0, s0), (w0, w1, 2 * s1 + 1, s1, s1), (w1, w2, 2 * s2 + 1, s2, s2),
                  (w2, w3, 2 * s3 + 1, s3, s3), (w3, latent_channels, 5, 1, 2)]
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
        h = self.embed(x.long()).transpose(1, 2)   # (B, L) ids -> (B, E, L)
        for index, conv in enumerate(self.convs):
            h = conv(h)
            if self.afs is not None:
                h, gate = _gate(self.afs[index], h, snr_db)
                if return_attn:
                    attn["af5_bottleneck" if index == 4 else f"af{index + 1}"] = gate.detach()
            if index < 4:
                h = self.acts[index](h)
        z = power_normalize(h.reshape(h.shape[0], -1), P=1.0)  # channel-major, like the image / speech models
        if return_attn:
            return z, attn
        return z


class TextJSCCDecoder(nn.Module):
    def __init__(self, segment_length: int = DEFAULT_SEGMENT_LENGTH, latent_channels: int = DEFAULT_LATENT_CHANNELS,
                 attention: bool = True, widths: Sequence[int] = DEFAULT_WIDTHS,
                 strides: Sequence[int] = DEFAULT_STRIDES, vocab_size: int = DEFAULT_VOCAB_SIZE,
                 embed_dim: int = DEFAULT_EMBED_DIM):
        super().__init__()
        _check_geometry(segment_length, widths, strides)
        self.segment_length = segment_length
        self.latent_channels = latent_channels
        self.attention = attention
        self.vocab_size = vocab_size
        w0, w1, w2, w3 = widths
        s0, s1, s2, s3 = (int(s) for s in strides)
        layers = [(latent_channels, w3, 5, 1, 2), (w3, w2, 2 * s3, s3, s3 // 2), (w2, w1, 2 * s2, s2, s2 // 2),
                  (w1, w0, 2 * s1, s1, s1 // 2), (w0, vocab_size, 2 * s0, s0, s0 // 2)]
        # Stride-1 first layer: ConvTranspose1d(k=5, s=1, p=2) keeps the length, as in the speech decoder.
        self.deconvs = nn.ModuleList(nn.ConvTranspose1d(i, o, k, stride=s, padding=p) for i, o, k, s, p in layers)
        self.afs = nn.ModuleList(AFModule(o) for _, o, *_ in layers) if attention else None
        self.acts = nn.ModuleList(nn.PReLU() for _ in range(4))

    @property
    def has_attention(self) -> bool:
        return self.attention

    def forward(self, z_hat_flat: torch.Tensor, snr_db=NOMINAL_SNR_DB_IDENTITY, return_attn: bool = False,
                spatial_hw=None):
        """Logits ``(B, V, L)``; ``spatial_hw`` is accepted and ignored (1-D latent)."""
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
        if return_attn:
            return h, attn
        return h


class TextJSCCQEncoder(nn.Module):
    """Text backbone followed by the shared 256-level quantiser (mirrors ``ADJSCCQEncoder``)."""

    def __init__(self, segment_length: int = DEFAULT_SEGMENT_LENGTH, latent_channels: int = DEFAULT_LATENT_CHANNELS,
                 attention: bool = True, widths: Sequence[int] = DEFAULT_WIDTHS,
                 strides: Sequence[int] = DEFAULT_STRIDES, vocab_size: int = DEFAULT_VOCAB_SIZE,
                 embed_dim: int = DEFAULT_EMBED_DIM, num_levels: int = 256, init_range: tuple = (-3.0, 3.0),
                 init_sigma: float = 1.0):
        super().__init__()
        self.encoder = TextJSCCEncoder(segment_length, latent_channels, attention, widths, strides, vocab_size,
                                       embed_dim)
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

    @property
    def vocab_size(self) -> int:
        return self.encoder.vocab_size

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
TextJSCCQDecoder = TextJSCCDecoder
