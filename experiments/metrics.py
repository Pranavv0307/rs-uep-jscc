"""Per-image image-quality metrics shared by the evaluation scripts."""
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = (g1d / g1d.sum()).unsqueeze(1)
    g2d = g1d @ g1d.t()
    return g2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_components(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
                     data_range: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-image ``(ssim, contrast-structure)`` using a Gaussian window."""
    channels = img1.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean(dim=[1, 2, 3]), cs_map.mean(dim=[1, 2, 3])


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, data_range: float = 1.0) -> torch.Tensor:
    """Standard windowed SSIM (Wang et al. 2004), self-contained in torch so
    there is no extra dependency beyond what the rest of the repo already
    requires. Returns one SSIM value per image in the batch."""
    return _ssim_components(img1, img2, window_size, data_range)[0]


def ms_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, data_range: float = 1.0,
            weights: Sequence[float] = MS_SSIM_WEIGHTS) -> torch.Tensor:
    """Multi-scale SSIM (Wang et al. 2003), one value per image.

    Needs the smallest scale to be at least ``window_size`` pixels wide. If the
    image is too small for all five scales (e.g. 32x32 CIFAR), fewer scales are
    used and the weights are renormalised, so the value stays in [0, 1].
    """
    min_side = min(img1.shape[-2:])
    scales = len(weights)
    while scales > 1 and min_side // (2 ** (scales - 1)) < window_size:
        scales -= 1
    w = torch.tensor(weights[:scales], dtype=img1.dtype, device=img1.device)
    w = w / w.sum()

    result = torch.ones(img1.shape[0], dtype=img1.dtype, device=img1.device)
    for level in range(scales):
        ssim_value, cs_value = _ssim_components(img1, img2, window_size, data_range)
        if level < scales - 1:
            result = result * cs_value.clamp_min(1e-6) ** w[level]
            img1, img2 = F.avg_pool2d(img1, 2), F.avg_pool2d(img2, 2)
        else:
            result = result * ssim_value.clamp_min(1e-6) ** w[level]
    return result


def psnr_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mse = ((x_hat - x) ** 2).mean(dim=[1, 2, 3])
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


# --------------------------------------------------------------------------
# Modality-aware helpers (speech study S1). Images (B, 3, H, W) always take the
# exact pre-S1 code path above; audio is (B, 1, L) in [-1, 1].
# --------------------------------------------------------------------------
def is_audio(x: torch.Tensor) -> bool:
    return x.dim() == 3


def per_item_mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return ((x_hat - x) ** 2).mean(dim=[1, 2, 3])
    return ((x_hat - x) ** 2).flatten(1).mean(dim=1)


def snr_db_per_item(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Per-segment SNR, 10 log10(||x||^2 / ||x - x_hat||^2).

    The difference of two SNRs of the same segment equals the difference of the
    corresponding PSNRs (the signal power cancels), so leave-one-out damage and
    paired gains in dB mean the same thing for speech and images (AD-16).
    """
    signal = x.flatten(1).pow(2).sum(dim=1)
    noise = (x_hat - x).flatten(1).pow(2).sum(dim=1)
    return 10 * torch.log10(signal.clamp_min(eps) / noise.clamp_min(eps))


def si_sdr_db_per_item(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Scale-invariant SDR (Le Roux et al. 2019) per segment, zero-mean version."""
    reference = x.flatten(1)
    estimate = x_hat.flatten(1)
    reference = reference - reference.mean(dim=1, keepdim=True)
    estimate = estimate - estimate.mean(dim=1, keepdim=True)
    scale = (estimate * reference).sum(dim=1, keepdim=True) / reference.pow(2).sum(dim=1, keepdim=True).clamp_min(eps)
    target = scale * reference
    residual = estimate - target
    return 10 * torch.log10(target.pow(2).sum(dim=1).clamp_min(eps) / residual.pow(2).sum(dim=1).clamp_min(eps))


def quality_db_per_item(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The pipeline's primary per-item quality in dB: PSNR for images (unchanged), SNR for audio.

    Stored under the existing ``psnr_db_*`` result keys so selection, reporting and
    comparison code work for both modalities; speech result files say so in their
    metadata (``quality_metric``).
    """
    if x.dim() == 4:
        return psnr_per_image(x_hat, x)
    return snr_db_per_item(x_hat, x)


_LPIPS = {}


def lpips_per_image(x_hat: torch.Tensor, x: torch.Tensor, net: str = "alex") -> Optional[torch.Tensor]:
    """LPIPS distance per image, or ``None`` if the optional ``lpips`` package is missing.

    Inputs are in [0, 1]; they are mapped to [-1, 1] as LPIPS expects.
    """
    try:
        import lpips
    except ImportError:
        return None
    key = (net, str(x.device))
    if key not in _LPIPS:
        _LPIPS[key] = lpips.LPIPS(net=net, verbose=False).to(x.device).eval()
    with torch.no_grad():
        return _LPIPS[key](x_hat * 2 - 1, x * 2 - 1).flatten()
