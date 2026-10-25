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
