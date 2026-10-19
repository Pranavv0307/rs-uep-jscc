"""Every way of ranking the latent channels, behind one interface.

A *source* turns one image batch into a ``(B, C)`` channel order (most to least
important). The same sources feed the erasure sweeps, the proxy-quality study
(A.2) and the transfer study (A.1), so they are compared like for like.

=====================  =====================================================  ==========================
source                 definition                                             needs at the transmitter
=====================  =====================================================  ==========================
``attention``          ADJSCC bottleneck gate (per image)                     encoder only
``energy``             mean z^2 per channel                                   latent only
``variance``           variance of z per channel                              latent only
``l1``                 mean |z| per channel = (1/2) mean |d||z||^2 / dz|      latent only
                       (analytic gradient saliency of roadmap A.2)
``taylor``             |sum_hw (dMSE/dz) * z| per channel: first-order        decoder copy + the image
                       estimate of the damage of zeroing the channel
``predicted``          trained SensitivityHead (A.3)                          encoder + tiny head
``oracle_perimage``    true per-image LOO damage                              image + decoder
                                                                              (upper bound, not deployable)
``static:<name>``      one frozen ranking from a JSON file (train-split        nothing per image
                       oracle, another dataset's, another model's, ...)
``random``             random order per image (control)                       handled in the sweep loop
=====================  =====================================================  ==========================

Statistics are computed on the latent *after* power normalisation, which is what
is quantised and sent (roadmap A.2, "what NOT to do"). No labels are ever used.
"""
from typing import Dict, Optional, Sequence

import torch

from experiments.model_io import ModelBundle, encode_full
from experiments.sensitivity import loo_damage_batch

STATIC_PREFIX = "static:"
PER_IMAGE_SOURCES = ("attention", "energy", "variance", "l1", "taylor", "predicted", "oracle_perimage")


def descending_order(scores: torch.Tensor) -> torch.Tensor:
    return torch.argsort(scores, dim=1, descending=True, stable=True)


def taylor_scores(bundle: ModelBundle, z_q: torch.Tensor, x: torch.Tensor, snr_db: float,
                  spatial_hw=None) -> torch.Tensor:
    """|gradient . latent| per channel: first-order estimate of the loss increase from zeroing it."""
    channels = bundle.channels
    with torch.enable_grad():
        latent = z_q.detach().clone().requires_grad_(True)
        if spatial_hw is None:
            recon = bundle.decoder(latent, snr_db)
        else:
            recon = bundle.decoder(latent, snr_db, spatial_hw=spatial_hw)
        per_image = ((recon - x) ** 2).mean(dim=[1, 2, 3])
        (gradient,) = torch.autograd.grad(per_image.sum(), latent)
    contribution = (gradient * latent.detach()).reshape(latent.shape[0], channels, -1).sum(dim=-1)
    return contribution.abs()


@torch.no_grad()
def compute_orders(
    bundle: ModelBundle,
    x: torch.Tensor,
    sources: Sequence[str],
    snr_db: float,
    rankings: Optional[Dict[str, Sequence[int]]] = None,
    head: Optional[torch.nn.Module] = None,
    info: Optional[dict] = None,
    spatial_hw=None,
) -> Dict[str, torch.Tensor]:
    """``{source: (B, C) long order}`` on ``x``'s device. ``random`` is not handled here."""
    info = info or encode_full(bundle.encoder, x, snr_db)
    z = info["z"]
    batch, channels = z.shape[0], bundle.channels
    per_channel = z.reshape(batch, channels, -1)
    orders: Dict[str, torch.Tensor] = {}
    for source in sources:
        if source == "attention":
            scores = info["attn"]["af5_bottleneck"].reshape(batch, channels)
        elif source == "energy":
            scores = per_channel.pow(2).mean(dim=-1)
        elif source == "variance":
            scores = per_channel.var(dim=-1, unbiased=False)
        elif source == "l1":
            scores = per_channel.abs().mean(dim=-1)
        elif source == "taylor":
            scores = taylor_scores(bundle, info["z_q"], x, snr_db, spatial_hw)
        elif source == "predicted":
            if head is None:
                raise ValueError("source 'predicted' needs a trained sensitivity head (--head_checkpoint)")
            scores = head(z)
        elif source == "oracle_perimage":
            latent = bundle.encoder.quantizer.dequantize_indices(info["idx"])
            scores, _ = loo_damage_batch(bundle, latent, x, snr_db, spatial_hw)
        elif source.startswith(STATIC_PREFIX):
            name = source[len(STATIC_PREFIX):]
            if not rankings or name not in rankings:
                raise ValueError(f"no ranking named {name!r}; provide it with --rankings {name}=<file>")
            orders[source] = torch.tensor(list(rankings[name]), device=z.device).unsqueeze(0).expand(batch, -1)
            continue
        else:
            raise ValueError(f"unknown importance source {source!r}")
        orders[source] = descending_order(scores)
    return orders
