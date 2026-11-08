"""Load trained JSCC models (optionally with an inpainting head) and run them.

A *bundle* is everything the receiver and transmitter need: the encoder (with
its quantizer), the decoder, an optional :class:`LatentInpainter`, and the
config stored in the checkpoint. All evaluation scripts go through
:func:`reconstruct`, so a model with an inpainter is evaluated exactly like one
without, just with the extra receiver-side step.

G0: the architecture is read from ``config["model"]["arch"]`` through
``models/registry.py`` (ADJSCC-Q when absent, i.e. every pre-G0 checkpoint), so
the same bundle serves ADJSCC-Q and DeepJSCC-Q. Scripts must use the bundle
properties below (``channels``, ``has_attention``, ``arch``) rather than reaching
into model internals such as ``encoder.encoder.af5``.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.adjsccq import NOMINAL_SNR_DB_IDENTITY, SensitivityHead
from models.inpainting import LatentInpainter
from models.registry import build_jscc, model_arch, model_modality


def _symbols_per_channel(encoder: nn.Module, cfg: dict) -> int:
    """Latent symbols per channel: the speech models report it; image models use (S/4)^2 as before."""
    steps = getattr(encoder, "symbols_per_channel", None)
    if steps is not None:
        return int(steps)
    return (int(cfg["model"]["image_size"]) // 4) ** 2


@dataclass
class ModelBundle:
    encoder: nn.Module   # any registered encoder (models/registry.py)
    decoder: nn.Module   # its decoder
    inpainter: Optional[LatentInpainter]
    config: dict
    path: str
    device: torch.device
    split_seed: int = 42

    @property
    def arch(self) -> str:
        return model_arch(self.config)

    @property
    def has_attention(self) -> bool:
        """True when the model has a bottleneck attention gate (the ``attention`` importance source)."""
        return bool(getattr(self.encoder, "has_attention", False))

    @property
    def channels(self) -> int:
        return self.encoder.num_channels

    @property
    def modality(self) -> str:
        """``image`` (inputs (B, 3, H, W)) or ``speech`` (inputs (B, 1, L))."""
        return model_modality(self.config)

    @property
    def image_size(self) -> int:
        """Image models only."""
        return int(self.config["model"]["image_size"])

    @property
    def symbols_per_channel(self) -> int:
        return _symbols_per_channel(self.encoder, self.config)

    def example_input(self, batch: int = 1) -> torch.Tensor:
        """A zero input of the shape the encoder expects (e.g. for counting MACs)."""
        if self.modality == "speech":
            return torch.zeros(batch, 1, self.encoder.segment_length)
        return torch.zeros(batch, 3, self.image_size, self.image_size)

    @property
    def snr_db(self) -> float:
        return float(self.config.get("af_conditioning", {}).get("nominal_snr_db", NOMINAL_SNR_DB_IDENTITY))

    def eval(self) -> "ModelBundle":
        for module in (self.encoder, self.decoder, self.inpainter):
            if module is not None:
                module.eval()
        return self


def _torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:  # noqa: BLE001 - older checkpoints may pickle non-tensor objects
        return torch.load(path, map_location=device, weights_only=False)


def build_models(cfg: dict, device):
    """``(encoder, decoder)`` of the architecture named in ``cfg`` (ADJSCC-Q by default).

    Kept under this name because ``experiments/train_highres.py`` imports it.
    """
    return build_jscc(cfg, device)


def load_model_bundle(path: str, device) -> ModelBundle:
    device = torch.device(device)
    checkpoint = _torch_load(path, device)
    cfg = checkpoint["config"]
    encoder, decoder = build_models(cfg, device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    inpainter = None
    if "inpainter" in checkpoint:
        channels = encoder.num_channels
        spc = _symbols_per_channel(encoder, cfg)
        meta = checkpoint.get("inpainter_config", {})
        inpainter = LatentInpainter(channels, spc, meta.get("heads", 4), meta.get("layers", 2)).to(device)
        inpainter.load_state_dict(checkpoint["inpainter"])
    split_seed = int(checkpoint.get("split", {}).get("seed", cfg.get("split_seed", 42)))
    return ModelBundle(encoder, decoder, inpainter, cfg, path, device, split_seed).eval()


def load_sensitivity_head(path: str, device) -> SensitivityHead:
    """Load a head saved by ``experiments/train_sensitivity_head.py``."""
    checkpoint = _torch_load(path, torch.device(device))
    head = SensitivityHead(checkpoint["in_channels"], checkpoint["hidden_dim"], checkpoint["use_variance"])
    head.load_state_dict(checkpoint["state_dict"])
    return head.to(device).eval()


@torch.no_grad()
def encode_full(encoder: nn.Module, x: torch.Tensor, snr_db: float):
    """Latent ``z`` (continuous), hard-quantised ``z_q``, byte indices, attention gates.

    ``attn`` is an empty dict for a model without attention (DeepJSCC-Q).
    """
    z, attn = encoder.encoder(x, snr_db, return_attn=True)
    z_q, idx = encoder.quantizer(z)
    return {"z": z, "z_q": z_q, "idx": idx, "attn": attn}


def reconstruct(bundle: ModelBundle, latent: torch.Tensor, failed: Optional[torch.Tensor],
                snr_db: float, spatial_hw=None) -> torch.Tensor:
    """Decoder output from a (possibly erased) latent.

    ``failed`` is the receiver's RS-decoding outcome (``True`` = symbol lost,
    already replaced by 0.0 in ``latent``). It is only used by the inpainter.
    """
    if bundle.inpainter is not None and failed is not None and bool(failed.any()):
        latent = bundle.inpainter(latent, failed)
    if spatial_hw is None:
        return bundle.decoder(latent, snr_db)
    return bundle.decoder(latent, snr_db, spatial_hw=spatial_hw)
