"""Step D.2: classical separation (image codec + Reed-Solomon) over the same packet losses.

Each test image is compressed to fit the Deep JSCC data budget (default 1024
bytes: 2-byte length header + payload, zero padded), protected with the same
``RS(96, 64)`` block (identical 50% overhead, identical 1536 transmitted bytes)
and sent through **the same master loss masks** as the Deep JSCC methods. When the
loss exceeds what the code can repair, the codec sees a damaged bitstream:

* the decoder is run on whatever arrived (truncated / zero-filled streams are
  decoded as far as possible, the most favourable behaviour for the baseline);
* if it fails, the receiver shows mid-grey (PSNR is then computed against that).

Codecs: ``jpeg`` and ``jpeg2000`` and ``webp`` through Pillow (always available
when Pillow was built with them), and ``bpg`` if ``bpgenc``/``bpgdec`` are on PATH
(the roadmap's choice; install from https://bellard.org/bpg/). The rate search
picks the highest quality whose stream fits the budget.

    python -m experiments.eval_digital_baseline --split val --codecs jpeg2000,jpeg,webp \\
        --seeds 0,1,2,3,4 --output_dir results/phase4/digital
"""
import argparse
import io
import itertools
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np
import torch

from coding.erasure_channel import apply_erasure_mask, sample_master_mask, slice_master_mask
from coding.rs_pipeline import NO_RS, UNIFORM_RS, decode_blocks, encode_blocks
from data.datasets import is_official_test, load_image_tensor
from experiments import wandb_utils
from experiments.metrics import psnr_per_image, ssim
from experiments.pipeline_core import SUMMARY_METRICS, module_function, run_tasks
from experiments.reporting import write_sweep_outputs
from experiments.reproducibility import build_provenance, log_test_access, seed_everything

MODULE = "experiments.eval_digital_baseline"
HEADER_BYTES = 2
GREY = 0.5


# --------------------------------------------------------------------------
# Codecs
# --------------------------------------------------------------------------
def _to_pil(image: np.ndarray):
    from PIL import Image

    return Image.fromarray(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8).transpose(1, 2, 0))


def encode_pillow(image: np.ndarray, codec: str, quality: int) -> bytes:
    buffer = io.BytesIO()
    pil = _to_pil(image)
    if codec == "jpeg":
        pil.save(buffer, format="JPEG", quality=quality, optimize=True)
    elif codec == "webp":
        pil.save(buffer, format="WEBP", quality=quality, method=6)
    elif codec == "jpeg2000":
        # quality 1..100 -> compression ratio 20.8..1.0 (higher quality = lower ratio)
        ratio = 1.0 + (100.0 - quality) * 0.2
        pil.save(buffer, format="JPEG2000", quality_mode="rates", quality_layers=[ratio], irreversible=True)
    else:
        raise ValueError(codec)
    return buffer.getvalue()


def decode_pillow(stream: bytes, shape) -> Optional[np.ndarray]:
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # most favourable to the baseline
    try:
        pil = Image.open(io.BytesIO(stream)).convert("RGB")
        pil.load()
    except Exception:  # noqa: BLE001 - any codec failure means "nothing displayable"
        return None
    if pil.size != (shape[2], shape[1]):
        return None
    return np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0


def have_bpg() -> bool:
    return shutil.which("bpgenc") is not None and shutil.which("bpgdec") is not None


def encode_bpg(image: np.ndarray, quality: int) -> bytes:
    with tempfile.TemporaryDirectory() as folder:
        source, target = os.path.join(folder, "in.png"), os.path.join(folder, "out.bpg")
        _to_pil(image).save(source)
        subprocess.run(["bpgenc", "-q", str(quality), "-o", target, source], check=True, capture_output=True)
        with open(target, "rb") as handle:
            return handle.read()


def decode_bpg(stream: bytes, shape) -> Optional[np.ndarray]:
    from PIL import Image

    with tempfile.TemporaryDirectory() as folder:
        source, target = os.path.join(folder, "in.bpg"), os.path.join(folder, "out.png")
        with open(source, "wb") as handle:
            handle.write(stream)
        try:
            subprocess.run(["bpgdec", "-o", target, source], check=True, capture_output=True)
            pil = Image.open(target).convert("RGB")
        except Exception:  # noqa: BLE001
            return None
    return np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0


def encode_stream(image: np.ndarray, codec: str, quality: int) -> bytes:
    return encode_bpg(image, quality) if codec == "bpg" else encode_pillow(image, codec, quality)


def decode_stream(stream: bytes, codec: str, shape) -> Optional[np.ndarray]:
    return decode_bpg(stream, shape) if codec == "bpg" else decode_pillow(stream, shape)


def fit_to_budget(image: np.ndarray, codec: str, budget: int) -> Optional[bytes]:
    """Highest-quality stream of at most ``budget`` bytes (binary search), or ``None``."""
    low, high = (0, 51) if codec == "bpg" else (1, 100)
    best = None
    if codec == "bpg":  # bpgenc: lower q = better quality
        while low <= high:
            mid = (low + high) // 2
            stream = encode_stream(image, codec, mid)
            if len(stream) <= budget:
                best, high = stream, mid - 1
            else:
                low = mid + 1
        return best
    while low <= high:
        mid = (low + high) // 2
        stream = encode_stream(image, codec, mid)
        if len(stream) <= budget:
            best, low = stream, mid + 1
        else:
            high = mid - 1
    return best


def pack(stream: bytes, total: int) -> np.ndarray:
    payload = len(stream).to_bytes(HEADER_BYTES, "big") + stream
    return np.frombuffer(payload.ljust(total, b"\x00"), dtype=np.uint8)


def unpack(data: np.ndarray, failed: np.ndarray) -> bytes:
    data = np.where(failed, 0, data).astype(np.uint8)
    length = int.from_bytes(bytes(data[:HEADER_BYTES]), "big")
    return bytes(data[HEADER_BYTES:HEADER_BYTES + min(length, len(data) - HEADER_BYTES)])


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------
def build_context(payload):
    return payload


def run_cell(context, task):
    seed, rate = task
    images, bitstreams, data = context["images"], context["bitstreams"], context["data"]
    batch_size = context["batch_size"]
    codecs = context["codecs"]
    rows, per_image = [], {}
    stats = {(codec, coded): {"psnr": [], "ssim": [], "erased": 0, "total": 0, "failed": 0, "clean": []}
             for codec in codecs for coded in (True, False)}
    for batch_index, start in enumerate(range(0, len(images), batch_size)):
        stop = min(start + batch_size, len(images))
        x = images[start:stop]
        master = sample_master_mask(stop - start, rate, torch.Generator().manual_seed(seed * 100000 + batch_index),
                                    num_packets=UNIFORM_RS.total_packets)
        for codec in codecs:
            payload = torch.from_numpy(data[codec][start:stop].astype(np.int64))
            for coded in (True, False):
                scheme = UNIFORM_RS if coded else NO_RS
                packets = encode_blocks(payload, scheme, None)
                received = apply_erasure_mask(packets, slice_master_mask(master, scheme.total_packets)).received
                decoded = decode_blocks(received, scheme, None)
                indices = decoded.indices.numpy().astype(np.uint8)
                failed = decoded.failed.numpy()
                recon = np.full(x.shape, GREY, dtype=np.float32)
                for row in range(stop - start):
                    stream_ok = bitstreams[codec][start + row] is not None
                    image = decode_stream(unpack(indices[row], failed[row]), codec, x.shape[1:]) if stream_ok else None
                    if image is not None:
                        recon[row] = image
                recon = torch.from_numpy(recon)
                s = stats[(codec, coded)]
                s["psnr"].append(psnr_per_image(recon, x).numpy())
                s["ssim"].append(ssim(recon, x).numpy())
                s["erased"] += int(received.erased.sum())
                s["total"] += received.erased.numel()
                s["failed"] += int(decoded.block_failed.any(dim=1).sum())
    for (codec, coded), s in stats.items():
        psnr = np.concatenate(s["psnr"])
        scheme = UNIFORM_RS if coded else NO_RS
        name = f"{codec}+rs" if coded else f"{codec}_no_rs"
        rows.append({
            "method": name, "scheme": "uniform" if coded else "no_rs", "importance": None, "interleave": "sequential",
            "codec": codec, "erasure_rate": rate, "seed": seed, "dataset": context["dataset"], "split": context["split"],
            "images": len(psnr), "psnr_db_mean_per_image": float(psnr.mean()), "psnr_db_std_per_image": float(psnr.std()),
            "psnr_db_median": float(np.median(psnr)), "psnr_db_p5": float(np.percentile(psnr, 5)),
            "ssim_mean": float(np.concatenate(s["ssim"]).mean()),
            "any_block_failure_rate": s["failed"] / len(psnr), "failed_symbol_fraction": s["failed"] / len(psnr),
            "realized_erasure_rate": s["erased"] / max(s["total"], 1),
            "coding_overhead_fraction": scheme.overhead, "side_info_bytes_per_image": 0,
            "total_transmitted_bytes_per_image": scheme.transmitted_symbols,
            "total_overhead_fraction_including_side_info": scheme.overhead, "data_symbols": scheme.data_symbols,
            "scheme_tiers_packets": [list(t) for t in scheme.tiers], "packet_size": scheme.packet_size,
        })
        per_image[f"{name}__s{seed}__r{rate:.3f}"] = psnr.astype(np.float32)
    return rows, per_image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--codecs", default="jpeg2000,jpeg,webp")
    parser.add_argument("--budget_bytes", type=int, default=UNIFORM_RS.data_symbols)
    parser.add_argument("--erasure_rates", default="0.0,0.1,0.2,0.3,0.35,0.4,0.45,0.5")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/phase4/digital")
    parser.add_argument("--name", default="digital")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    parser.add_argument("--wandb_name", default=None)
    args = parser.parse_args()

    seed_everything(args.global_seed)
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    if "bpg" in codecs and not have_bpg():
        print("WARNING: bpgenc/bpgdec not on PATH; dropping 'bpg' (JPEG2000/WebP/JPEG still run)")
        codecs.remove("bpg")
    if is_official_test("cifar10", args.split):
        log_test_access("experiments.eval_digital_baseline", None, note=f"name={args.name}")
    images = load_image_tensor("cifar10", args.split, args.data_dir, args.max_images, args.split_seed)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rates = [float(r) for r in args.erasure_rates.split(",") if r.strip()]

    # Compression is independent of seed and loss rate: do it once.
    budget = args.budget_bytes - HEADER_BYTES
    bitstreams, data, clean = {}, {}, {}
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731
    numpy_images = images.numpy()
    for codec in codecs:
        streams = [fit_to_budget(img, codec, budget) for img in tqdm(numpy_images, desc=f"encoding {codec}")]
        bitstreams[codec] = streams
        data[codec] = np.stack([pack(s if s is not None else b"", args.budget_bytes) for s in streams])
        decoded = [decode_stream(s, codec, numpy_images.shape[1:]) if s is not None else None for s in streams]
        recon = np.stack([d if d is not None else np.full(numpy_images.shape[1:], GREY, np.float32) for d in decoded])
        clean[codec] = {
            "clean_psnr_db": float(psnr_per_image(torch.from_numpy(recon), images).mean()),
            "mean_stream_bytes": float(np.mean([len(s) for s in streams if s is not None])),
            "images_that_do_not_fit": int(sum(s is None for s in streams)),
        }
        print(f"{codec}: clean PSNR {clean[codec]['clean_psnr_db']:.2f} dB, "
              f"mean stream {clean[codec]['mean_stream_bytes']:.0f} B, no-fit {clean[codec]['images_that_do_not_fit']}")

    run = wandb_utils.init_run(args.wandb, args.wandb_project, args.wandb_name or "eval-digital", vars(args), tags=["digital", args.split],
                               group="phase4", job_type="eval")
    payload = {"images": images, "bitstreams": bitstreams, "data": data, "codecs": codecs,
               "batch_size": args.batch_size, "dataset": "cifar10", "split": args.split}
    results = run_tasks(module_function(MODULE, "run_cell"), list(itertools.product(seeds, rates)),
                        module_function(MODULE, "build_context"), payload, args.workers, "digital cells",
                        lambda task, result: wandb_utils.log_metrics(
                            run, {"seed": task[0], "erasure_rate": task[1],
                                  **{f"{r['method']}/psnr_db": r["psnr_db_mean_per_image"] for r in result[0]}}))
    rows = [row for cell_rows, _ in results for row in cell_rows]
    per_image = {k: v for _, cell in results for k, v in cell.items()}
    metadata = build_provenance(args=args, global_seed=args.global_seed,
                                split_info={"dataset": "cifar10", "split": args.split},
                                extra={"evaluation": {"codecs": codecs, "budget_bytes": args.budget_bytes,
                                                      "clean": clean, "images_per_seed": len(images),
                                                      "channel_seeds": seeds, "erasure_rates": rates,
                                                      "bpg_available": have_bpg()}})
    baseline = f"{codecs[0]}+rs"
    outputs = write_sweep_outputs(args.output_dir, args.name, rows, metadata, per_image, baseline=baseline,
                                  title="Digital separation (codec + RS) under packet loss",
                                  metrics=SUMMARY_METRICS)
    print("\n".join(f"- {line}" for line in outputs["insights"]))
    wandb_utils.log_table(run, "seed_summary", outputs["summary"])
    wandb_utils.log_images(run, "figures", outputs["figures"])
    wandb_utils.finish(run)


if __name__ == "__main__":
    main()
