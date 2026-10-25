"""Shared machinery for every packet-erasure experiment.

One implementation of "send these latent indices through coding + a lossy
channel and reconstruct", used by the erasure sweep, the burst study and the
Kodak study, so Rules 2-4 of the roadmap are enforced in exactly one place:

* the receiver sees only received packets and the failure mask (Rule 2);
* every method is handed slices of the **same** master loss realisation (Rule 3);
* side-information bytes follow ``coding.rs_pipeline.side_info_bytes`` (Rule 4).

It also provides the parallel runner. Cells (one seed x one loss setting) are
independent and individually seeded, so results are bit-identical for any
number of workers.
"""
import importlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from coding.erasure_channel import apply_erasure_mask
from coding.rs_pipeline import (
    CodingScheme,
    decode_blocks,
    dequantize_with_fallback,
    encode_blocks,
    random_channel_order,
    side_info_bytes,
)
from experiments.importance_sources import compute_orders
from experiments.metrics import lpips_per_image, ms_ssim, psnr_per_image, ssim
from experiments.model_io import ModelBundle, encode_full, reconstruct

INTERLEAVE_TAG = {"sequential": "", "round_robin": "#rr"}

# Metrics aggregated across channel seeds (mean / std / 95% CI).
SUMMARY_METRICS = (
    "psnr_db_mean_per_image", "psnr_db_median", "psnr_db_p5", "ssim_mean", "ms_ssim_mean", "lpips_mean",
    "failed_symbol_fraction", "any_block_failure_rate", "realized_erasure_rate",
)


# --------------------------------------------------------------------------
# Method specification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MethodSpec:
    name: str
    scheme: str                    # key into the scheme catalog
    source: Optional[str] = None   # importance source; None = no ranking, no sorting
    interleave: str = "sequential"


def display_source(source: str) -> str:
    """``static:oracle`` -> ``oracle``; ``static:cifar100`` -> ``static_cifar100``; others unchanged."""
    if source == "static:oracle":
        return "oracle"
    return source.replace("static:", "static_")


def canonical_source(token: str) -> str:
    """Accept the short names used on the command line."""
    return "static:oracle" if token in ("oracle", "oracle_static") else token


def make_method_specs(schemes: Sequence[str], sources: Sequence[str], include_no_rs: bool = False,
                      include_random: bool = False, interleaves: Sequence[str] = ("sequential",)) -> List[MethodSpec]:
    """``no_rs`` (optional), ``uniform`` (always: it is the paired baseline), then scheme@source."""
    specs: List[MethodSpec] = []
    if include_no_rs:
        specs.append(MethodSpec("no_rs", "no_rs"))
    specs.append(MethodSpec("uniform", "uniform"))
    for scheme in schemes:
        if scheme in ("uniform", "no_rs"):
            continue
        for source in list(sources) + (["random"] if include_random else []):
            for mode in interleaves:
                label = f"{scheme}@{display_source(source)}{INTERLEAVE_TAG[mode]}"
                specs.append(MethodSpec(label, scheme, source, mode))
    return specs


def method_order(spec: MethodSpec, orders: Dict[str, torch.Tensor], start: int, stop: int, channels: int,
                 generator: torch.Generator, device) -> Optional[torch.Tensor]:
    if spec.source is None:
        return None
    if spec.source == "random":
        return random_channel_order(stop - start, channels, generator).to(device)
    return orders[spec.source][start:stop].to(device)


# --------------------------------------------------------------------------
# One transmission
# --------------------------------------------------------------------------
@torch.no_grad()
def transmit(bundle: ModelBundle, idx: torch.Tensor, scheme: CodingScheme, order: Optional[torch.Tensor],
             packet_mask: torch.Tensor, snr_db: float, spatial_hw=None):
    """Encode -> erase -> decode -> dequantise (0.0 fallback) -> reconstruct."""
    packets = encode_blocks(idx, scheme, order)
    received = apply_erasure_mask(packets, packet_mask).received
    decoded = decode_blocks(received, scheme, order)
    latent = dequantize_with_fallback(bundle.encoder.quantizer, decoded.indices, decoded.failed)
    recon = reconstruct(bundle, latent, decoded.failed, snr_db, spatial_hw)
    return recon, decoded, int(received.erased.sum()), int(received.erased.numel())


class CellAccumulator:
    """Running per-image metrics of one method in one cell."""

    def __init__(self, spec: MethodSpec, scheme: CodingScheme, channels: int, constants: dict):
        self.spec, self.scheme, self.channels, self.constants = spec, scheme, channels, constants
        self.psnr: List[np.ndarray] = []
        self.ssim_sum = self.ms_ssim_sum = self.lpips_sum = 0.0
        self.has_ms, self.has_lpips = False, False
        self.images = self.failed_symbols = self.erased = self.total = self.images_with_failure = 0
        self.block_failures = np.zeros(len(scheme.tiers), dtype=np.int64)

    def add(self, recon, x, decoded, erased, total, extra_metrics: bool = False):
        psnr = psnr_per_image(recon, x)
        self.psnr.append(psnr.cpu().numpy())
        self.ssim_sum += ssim(recon, x).sum().item()
        if extra_metrics:
            self.ms_ssim_sum += ms_ssim(recon, x).sum().item()
            self.has_ms = True
            lp = lpips_per_image(recon, x)
            if lp is not None:
                self.lpips_sum += lp.sum().item()
                self.has_lpips = True
        batch = x.shape[0]
        self.images += batch
        self.failed_symbols += int(decoded.failed.sum())
        self.block_failures += decoded.block_failed.sum(dim=0).cpu().numpy()
        self.images_with_failure += int(decoded.block_failed.any(dim=1).sum())
        self.erased += erased
        self.total += total

    def finish(self):
        psnr = np.concatenate(self.psnr)
        side = side_info_bytes(self.spec.name, self.channels)
        transmitted = self.scheme.transmitted_symbols + side
        data = self.scheme.data_symbols
        row = {
            "method": self.spec.name,
            "scheme": self.spec.scheme,
            "importance": self.spec.source,
            "interleave": self.spec.interleave,
            "scheme_tiers_packets": [list(tier) for tier in self.scheme.tiers],
            "packet_size": self.scheme.packet_size,
            "data_symbols": data,
            "coding_symbols": self.scheme.transmitted_symbols,
            "coding_overhead_fraction": self.scheme.overhead,
            "side_info_bytes_per_image": side,
            "side_info_bytes_total": self.images * side,
            "total_transmitted_bytes_per_image": transmitted,
            "total_overhead_fraction_including_side_info": transmitted / data - 1.0,
            "images": self.images,
            "psnr_db_mean_per_image": float(psnr.mean()),
            "psnr_db_std_per_image": float(psnr.std()),  # image-to-image spread, not an error bar over seeds
            "psnr_db_median": float(np.median(psnr)),
            "psnr_db_p5": float(np.percentile(psnr, 5)),
            "ssim_mean": self.ssim_sum / self.images,
            "failed_symbol_fraction": self.failed_symbols / (self.images * data),
            "block_failure_rate_by_tier": (self.block_failures / self.images).tolist(),
            "any_block_failure_rate": self.images_with_failure / self.images,
            "realized_erasure_rate": self.erased / max(self.total, 1),
            **self.constants,
        }
        if self.has_ms:
            row["ms_ssim_mean"] = self.ms_ssim_sum / self.images
        if self.has_lpips:
            row["lpips_mean"] = self.lpips_sum / self.images
        return row, psnr.astype(np.float32)


# --------------------------------------------------------------------------
# Dataset-level precomputation (independent of seed and loss rate)
# --------------------------------------------------------------------------
@torch.no_grad()
def precompute(bundle: ModelBundle, images: torch.Tensor, sources: Sequence[str], batch_size: int,
               rankings: Optional[dict] = None, head=None, spatial_hw=None, progress=None) -> dict:
    """Byte indices, importance orders and clean PSNR for every image.

    Orders and indices do not depend on the channel seed or the loss rate, so
    they are computed once and shared by all cells.
    """
    snr_db = bundle.snr_db
    idx_parts, clean_parts = [], []
    order_parts: Dict[str, list] = {s: [] for s in sources}
    starts = range(0, len(images), batch_size)
    for start in (progress(starts) if progress else starts):
        x = images[start:start + batch_size].to(bundle.device)
        info = encode_full(bundle.encoder, x, snr_db)
        idx_parts.append(info["idx"].to(torch.uint8).cpu())
        orders = compute_orders(bundle, x, [s for s in sources if s != "random"], snr_db, rankings, head,
                                info=info, spatial_hw=spatial_hw)
        for source, order in orders.items():
            order_parts[source].append(order.cpu())
        recon = reconstruct(bundle, bundle.encoder.quantizer.dequantize_indices(info["idx"]), None, snr_db,
                            spatial_hw)
        clean_parts.append(psnr_per_image(recon, x).cpu())
    return {
        "idx": torch.cat(idx_parts),
        "orders": {s: torch.cat(parts) for s, parts in order_parts.items() if parts},
        "clean_psnr": torch.cat(clean_parts).numpy(),
    }


def make_context(bundle: ModelBundle, images: torch.Tensor, prep: dict, schemes: Dict[str, CodingScheme],
                 batch_size: int, spatial_hw=None) -> SimpleNamespace:
    return SimpleNamespace(bundle=bundle, X=images, prep=prep, schemes=schemes, batch_size=batch_size,
                           snr_db=bundle.snr_db, channels=bundle.channels, device=bundle.device,
                           spatial_hw=spatial_hw)


def batch_ranges(num_images: int, batch_size: int):
    """``(batch_index, start, stop)``; batch_index seeds the loss masks (reproducible)."""
    for batch_index, start in enumerate(range(0, num_images, batch_size)):
        yield batch_index, start, min(start + batch_size, num_images)


# --------------------------------------------------------------------------
# Parallel runner
# --------------------------------------------------------------------------
_WORKER: dict = {}


def module_function(module_name: str, function_name: str) -> Callable:
    """Resolve ``module.function`` through a normal import.

    A function defined in a script run with ``python -m`` is pickled as
    ``__main__.f``; going through the importable module name avoids that.
    """
    return getattr(importlib.import_module(module_name), function_name)


def _worker_init(builder: Callable, payload: dict) -> None:
    torch.set_num_threads(1)
    _WORKER["ctx"] = builder(payload)


def _worker_call(function: Callable, task):
    return function(_WORKER["ctx"], task)


def run_tasks(function: Callable, tasks: Sequence, builder: Callable, payload: dict, workers: int = 1,
              description: str = "cells", on_result: Optional[Callable] = None) -> list:
    """Run ``function(ctx, task)`` for every task; results come back in task order."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None
    results = [None] * len(tasks)
    bar = tqdm(total=len(tasks), desc=description) if tqdm else None

    def finish(index, value):
        results[index] = value
        if on_result is not None:
            on_result(tasks[index], value)
        if bar is not None:
            bar.update(1)

    if workers <= 1:
        context = builder(payload)
        for index, task in enumerate(tasks):
            finish(index, function(context, task))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=_worker_init, initargs=(builder, payload)) as pool:
            futures = {pool.submit(_worker_call, function, task): index for index, task in enumerate(tasks)}
            for future in as_completed(futures):
                finish(futures[future], future.result())
    if bar is not None:
        bar.close()
    return results
