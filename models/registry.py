"""Which JSCC architecture a config or checkpoint describes, and how to build it (G0, S1).

``config["model"]["arch"]`` selects the architecture. Checkpoints written before
G0 have no such key; they are ADJSCC-Q (``DEFAULT_ARCH``), so every existing
checkpoint, config and Kaggle launcher keeps working unchanged.

Every architecture registered here must provide the ADJSCC-Q interface (see
``models/deepjsccq.py``): ``encoder(x, snr_db, return_attn, return_indices)``,
``encoder.encoder`` (backbone), ``encoder.quantizer``, ``encoder.k``,
``encoder.num_channels``, ``encoder.calibrate_quantizer``, ``encoder.has_attention``
and ``decoder(z, snr_db, spatial_hw=...)``. ``tests/test_deepjsccq.py`` and
``tests/test_speech.py`` check this. See ``documentations/ARCHITECTURE_DECISIONS.md``
(AD-01, AD-14).

Each entry also names its *modality* (``image``: inputs ``(B, 3, H, W)``, config keys
``k_over_n`` / ``image_size``; ``speech``: inputs ``(B, 1, L)``, config keys
``segment_length`` / ``latent_channels`` / ``widths``).
"""
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn

from models.adjsccq import ADJSCCQDecoder, ADJSCCQEncoder
from models.deepjsccq import DeepJSCCQDecoder, DeepJSCCQEncoder
from models.speech_jscc import (
    DEFAULT_LATENT_CHANNELS,
    DEFAULT_SEGMENT_LENGTH,
    DEFAULT_WIDTHS,
    SpeechJSCCQDecoder,
    SpeechJSCCQEncoder,
)

DEFAULT_ARCH = "adjsccq"


class ArchSpec(NamedTuple):
    encoder: type
    decoder: type
    modality: str            # "image" or "speech"
    options: dict            # extra constructor arguments shared by encoder and decoder


ARCHITECTURES = {
    "adjsccq": ArchSpec(ADJSCCQEncoder, ADJSCCQDecoder, "image", {}),        # Phases 0-5
    "deepjsccq": ArchSpec(DeepJSCCQEncoder, DeepJSCCQDecoder, "image", {}),  # G1: no attention
    "adjsccq1d": ArchSpec(SpeechJSCCQEncoder, SpeechJSCCQDecoder, "speech", {"attention": True}),    # S1
    "deepjsccq1d": ArchSpec(SpeechJSCCQEncoder, SpeechJSCCQDecoder, "speech", {"attention": False}),  # S1
}


def model_arch(cfg: dict) -> str:
    """Architecture name of a config; ``adjsccq`` when the key is absent (pre-G0 checkpoints)."""
    arch = (cfg.get("model") or {}).get("arch") or DEFAULT_ARCH
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown model.arch {arch!r}; known: {sorted(ARCHITECTURES)}")
    return arch


def model_modality(cfg: dict) -> str:
    return ARCHITECTURES[model_arch(cfg)].modality


def build_jscc(cfg: dict, device: Optional[torch.device] = None) -> Tuple[nn.Module, nn.Module]:
    """Construct ``(encoder, decoder)`` for ``cfg``.

    The encoder is built before the decoder with the same arguments the
    training script always used, so for ``adjsccq`` the random initialisation
    is bit-identical to the pre-G0 code for the same seed (unit-tested).
    """
    spec = ARCHITECTURES[model_arch(cfg)]
    model_cfg = cfg["model"]
    quantizer_cfg = cfg.get("quantizer", {})
    quantizer_args = dict(
        num_levels=quantizer_cfg.get("num_levels", 256),
        init_range=tuple(quantizer_cfg.get("init_range", (-3.0, 3.0))),
        init_sigma=quantizer_cfg.get("sigma_start", 1.0),
    )
    if spec.modality == "image":
        encoder = spec.encoder(k_over_n=model_cfg["k_over_n"], image_size=model_cfg["image_size"], **quantizer_args)
        decoder = spec.decoder(k=encoder.k, image_size=model_cfg["image_size"])
    else:
        geometry = dict(
            segment_length=int(model_cfg.get("segment_length", DEFAULT_SEGMENT_LENGTH)),
            latent_channels=int(model_cfg.get("latent_channels", DEFAULT_LATENT_CHANNELS)),
            widths=tuple(model_cfg.get("widths", DEFAULT_WIDTHS)),
            **spec.options,
        )
        encoder = spec.encoder(**geometry, **quantizer_args)
        decoder = spec.decoder(**geometry)
    if device is not None:
        encoder, decoder = encoder.to(device), decoder.to(device)
    return encoder, decoder
