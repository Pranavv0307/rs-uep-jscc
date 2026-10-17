"""Load trained ADJSCC-Q models (optionally with an inpainting head) and run them.

A *bundle* is everything the receiver and transmitter need: the encoder (with
its quantizer), the decoder, an optional :class:`LatentInpainter`, and the
config stored in the checkpoint. All evaluation scripts go through
:func:`reconstruct`, so a model with an inpainter is evaluated exactly like one
without, just with the extra receiver-side step.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from models.adjsccq import ADJSCCQDecoder, ADJSCCQEncoder, NOMINAL_SNR_DB_IDENTITY, SensitivityHead
from models.inpainting import LatentInpainter


@dataclass
class ModelBundle:
    encoder: ADJSCCQEncoder
    decoder: ADJSCCQDecoder
    inpainter: Optional[LatentInpainter]
    config: dict
    path: str
    device: torch.device
    split_seed: int = 42

    @property
    def channels(self) -> int:
        return self.encoder.encoder.af5.num_channels

    @property
    def image_size(self) -> int:
        return int(self.config["model"]["image_size"])

    @property
    def symbols_per_channel(self) -> int:
        return (self.image_size // 4) ** 2

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
    encoder = ADJSCCQEncoder(
        k_over_n=cfg["model"]["k_over_n"],
        image_size=cfg["model"]["image_size"],
        num_levels=cfg["quantizer"]["num_levels"],
    ).to(device)
    decoder = ADJSCCQDecoder(k=encoder.k, image_size=cfg["model"]["image_size"]).to(device)
    return encoder, decoder


def load_model_bundle(path: str, device) -> ModelBundle:
    device = torch.device(device)
    checkpoint = _torch_load(path, device)
    cfg = checkpoint["config"]
    encoder, decoder = build_models(cfg, device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    inpainter = None
    if "inpainter" in checkpoint:
        channels = encoder.encoder.af5.num_channels
        spc = (cfg["model"]["image_size"] // 4) ** 2
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
def encode_full(encoder: ADJSCCQEncoder, x: torch.Tensor, snr_db: float):
    """Latent ``z`` (continuous), hard-quantised ``z_q``, byte indices, attention gates."""
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
