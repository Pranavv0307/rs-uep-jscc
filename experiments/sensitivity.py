"""Leave-one-channel-out (LOO) sensitivity, the ground truth behind every UEP ranking.

For one image, zeroing latent channel ``c`` and decoding gives a PSNR drop
``d_c``; the 16-vector ``d`` says how much each channel matters *for that image*.
Averaged over many training images it is the (static) oracle ranking. Used by:

* the oracle ranking and its cross-seed / cross-dataset comparison (Phase 1, A.1),
* per-image labels for the sensitivity head (A.3),
* the per-image oracle upper bound and the proxy-quality study (A.2),
* the attention-vs-damage figure.

Quantisation noise interacts with erasures, so the **quantised index path** is
always used (roadmap A.1, "what NOT to do").
"""
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.metrics import psnr_per_image
from experiments.model_io import ModelBundle, encode_full, reconstruct


@torch.no_grad()
def loo_damage_batch(bundle: ModelBundle, latent: torch.Tensor, x: torch.Tensor, snr_db: float,
                     spatial_hw=None):
    """``(damage (B, C), clean_psnr (B,))`` for a batch of dequantised latents."""
    channels = bundle.channels
    width = latent.shape[1] // channels
    clean = reconstruct(bundle, latent, None, snr_db, spatial_hw)
    clean_psnr = psnr_per_image(clean, x)
    damage = torch.empty(latent.shape[0], channels, device=latent.device)
    for channel in range(channels):
        failed = torch.zeros_like(latent, dtype=torch.bool)
        failed[:, channel * width:(channel + 1) * width] = True
        recon = reconstruct(bundle, latent.masked_fill(failed, 0.0), failed, snr_db, spatial_hw)
        damage[:, channel] = clean_psnr - psnr_per_image(recon, x)
    return damage, clean_psnr


@torch.no_grad()
def damage_for_tensor(bundle: ModelBundle, images: torch.Tensor, batch_size: int = 250,
                      progress: Optional[Callable] = None) -> Dict[str, np.ndarray]:
    """LOO damage for every image: ``{"damage": (N, C), "clean_psnr": (N,)}`` as float32 numpy."""
    snr_db = bundle.snr_db
    damages, cleans = [], []
    starts = range(0, len(images), batch_size)
    for start in (progress(starts) if progress else starts):
        x = images[start:start + batch_size].to(bundle.device)
        info = encode_full(bundle.encoder, x, snr_db)
        latent = bundle.encoder.quantizer.dequantize_indices(info["idx"])
        damage, clean = loo_damage_batch(bundle, latent, x, snr_db)
        damages.append(damage.cpu())
        cleans.append(clean.cpu())
    return {
        "damage": torch.cat(damages).numpy().astype(np.float32),
        "clean_psnr": torch.cat(cleans).numpy().astype(np.float32),
    }


def ranking_from_damage(damage: np.ndarray) -> list:
    """Channels from most to least damaging, using the mean damage over images."""
    mean = np.asarray(damage, dtype=np.float64).mean(axis=0)
    return np.argsort(-mean, kind="stable").tolist()


def pairwise_ranking_loss(predicted: torch.Tensor, target: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Hinge loss over all channel pairs: ``max(0, -y_ij (p_i - p_j) + margin)``.

    ``y_ij = sign(target_i - target_j)``; pairs with equal targets are ignored.
    """
    dp = predicted.unsqueeze(2) - predicted.unsqueeze(1)
    y = torch.sign(target.unsqueeze(2) - target.unsqueeze(1))
    active = (y != 0).to(predicted.dtype)
    return (F.relu(-y * dp + margin) * active).sum() / active.sum().clamp_min(1.0)


def count_macs(model: nn.Module, forward: Callable[[], object]) -> int:
    """Multiply-accumulates of one forward pass (conv, transposed conv, linear)."""
    total = [0]

    def conv_hook(module, inputs, output):
        total[0] += output[0].numel() * (module.in_channels // module.groups) * int(np.prod(module.kernel_size))

    def deconv_hook(module, inputs, output):
        total[0] += inputs[0][0].numel() * (module.out_channels // module.groups) * int(np.prod(module.kernel_size))

    def linear_hook(module, inputs, output):
        total[0] += module.in_features * module.out_features

    handles = []
    for module in model.modules():
        if isinstance(module, nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(deconv_hook))
        elif isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            forward()
    finally:
        for handle in handles:
            handle.remove()
    return int(total[0])
