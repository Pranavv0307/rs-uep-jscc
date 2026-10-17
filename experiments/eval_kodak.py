"""Step C.1: the UEP framework on full-resolution Kodak images (768x512).

The encoder/decoder are fully convolutional, so a model trained on 32x32 CIFAR
images runs on the native Kodak images (no cropping or resizing); the latent is
16 channels x 128 x 192 = 393,216 byte symbols (k/n is unchanged).

Packetisation (roadmap C.1): 128-byte packets, 3,072 data packets
(= 16 channels x 192 packets), 4,608 transmitted (50% overhead), tiers
channel-aligned. A Reed-Solomon codeword over GF(256) has at most 255 symbols,
so the roadmap's RS(1536, 768) cannot be a single code: every tier is split
into equal sub-blocks of <= 255 packets (``coding/scaling.py``) with the same
total data and parity. Finite-length effects of that split are visible in the
results and are discussed in the report.

Metrics: PSNR, SSIM, MS-SSIM and (if ``pip install lpips``) LPIPS. If the clean
PSNR is far below the CIFAR figure the CIFAR model does not generalise to
photographic content -- train a high-resolution model first
(``experiments/train_highres.py``) and pass its checkpoint here.

    python -m experiments.eval_kodak --checkpoint <ckpt> --oracle_ranking_file <ranking.json> \\
        --kodak_dir ./data/kodak --seeds 0,1,2,3,4 --workers 8 --save_images_rate 0.4 \\
        --output_dir results/phase3/kodak
"""
import itertools
import os

import numpy as np
import torch

from coding.erasure_channel import sample_master_mask
from coding.interleave import packet_mask_from_timeline
from coding.rs_pipeline import SCHEME_CANDIDATES
from coding.scaling import scheme_catalog
from data.datasets import KodakDataset
from experiments import wandb_utils
from experiments.eval_rs_uep import build_parser, collect_rankings, parse_csv, pick_device
from experiments.model_io import load_model_bundle, load_sensitivity_head
from experiments.pipeline_core import (
    SUMMARY_METRICS,
    CellAccumulator,
    batch_ranges,
    canonical_source,
    make_context,
    make_method_specs,
    method_order,
    module_function,
    precompute,
    run_tasks,
    transmit,
)
from experiments.metrics import ms_ssim, psnr_per_image
from experiments.reporting import write_sweep_outputs
from experiments.reproducibility import build_provenance, seed_everything
from experiments.stats_utils import paired_vs_baseline

MODULE = "experiments.eval_kodak"


def save_png(tensor: torch.Tensor, path: str) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = (tensor.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(array).save(path)


def build_context(payload):
    args = payload["args"]
    device = torch.device(payload["device"])
    bundle = load_model_bundle(args["checkpoint"], device)
    height, width = payload["X"].shape[-2:]
    spatial_hw = (height // 4, width // 4)
    symbols_per_channel = spatial_hw[0] * spatial_hw[1]
    specs = payload["specs"]
    schemes = scheme_catalog(bundle.channels, symbols_per_channel, args["packet_size"], args["overhead"],
                             names=sorted({s.scheme for s in specs}))
    context = make_context(bundle, payload["X"], payload["prep"], schemes, 1, spatial_hw)
    context.specs, context.args, context.names = specs, args, payload["names"]
    context.master_n = max(s.total_packets for s in schemes.values())
    return context


def run_cell(context, task):
    seed, rate = task
    args = context.args
    constants = {"erasure_rate": rate, "seed": seed, "dataset": "kodak", "split": "all",
                 "checkpoint": args["checkpoint"], "channels": context.channels}
    accumulators = {spec.name: CellAccumulator(spec, context.schemes[spec.scheme], context.channels, constants)
                    for spec in context.specs}
    save_here = (args["save_images_rate"] is not None and abs(rate - args["save_images_rate"]) < 1e-9
                 and seed == args["save_images_seed"])
    for image_index, start, stop in batch_ranges(len(context.X), 1):
        x = context.X[start:stop].to(context.device)
        idx = context.prep["idx"][start:stop].long().to(context.device)
        generator = torch.Generator().manual_seed(seed * 100000 + image_index)
        master = sample_master_mask(1, rate, generator, num_packets=context.master_n)
        for spec in context.specs:
            scheme = context.schemes[spec.scheme]
            order_generator = torch.Generator().manual_seed(seed * 100000 + image_index + 50000)
            order = method_order(spec, context.prep["orders"], start, stop, context.channels, order_generator,
                                 context.device)
            mask = packet_mask_from_timeline(master, scheme, spec.interleave)
            recon, decoded, erased, total = transmit(context.bundle, idx, scheme, order, mask, context.snr_db,
                                                     context.spatial_hw)
            accumulators[spec.name].add(recon, x, decoded, erased, total, extra_metrics=True)
            if save_here:
                save_png(recon[0], os.path.join(args["output_dir"], "images", spec.name,
                                                f"{context.names[image_index]}"))
    rows, per_image = [], {}
    for name, accumulator in accumulators.items():
        row, psnr = accumulator.finish()
        rows.append(row)
        per_image[f"{name}__s{seed}__r{rate:.3f}"] = psnr
    return rows, per_image


def main():
    parser = build_parser(__doc__)
    parser.add_argument("--kodak_dir", default="./data/kodak")
    parser.add_argument("--save_images_rate", type=float, default=None,
                        help="also write reconstructions (PNG) at this loss rate for --save_images_seed")
    parser.add_argument("--save_images_seed", type=int, default=0)
    parser.set_defaults(packet_size=128, schemes="three_tier_25_50_25,three_tier_25_25_50", importance="oracle",
                        batch_size=1, output_dir="results/phase3/kodak", name="kodak", dataset="cifar10",
                        split="all")
    args = parser.parse_args()
    seed_everything(args.global_seed)

    device = pick_device(args.device)
    bundle = load_model_bundle(args.checkpoint, device)
    dataset = KodakDataset(args.kodak_dir)
    count = len(dataset) if args.max_images is None else min(args.max_images, len(dataset))
    loaded = [dataset[i][0] for i in range(count)]
    # A few Kodak images are portrait (512x768). Rotate them to landscape so all 24 share one shape;
    # resolution and content are unchanged, only the orientation.
    rotated = [i for i, img in enumerate(loaded) if img.shape[-2] > img.shape[-1]]
    images = torch.stack([torch.rot90(img, 1, (1, 2)) if i in rotated else img for i, img in enumerate(loaded)])
    names = [dataset[i][1] for i in range(count)]
    print(f"rotated {len(rotated)} portrait images to landscape: {[names[i] for i in rotated]}")
    rankings = collect_rankings(args)
    sources = [canonical_source(s) for s in parse_csv(args.importance)]
    schemes_arg = list(SCHEME_CANDIDATES) if args.schemes == "all" else list(parse_csv(args.schemes))
    specs = make_method_specs(schemes_arg, sources, args.include_no_rs, args.include_random_control)
    seeds, rates = parse_csv(args.seeds, int), parse_csv(args.erasure_rates, float)

    spatial_hw = (images.shape[-2] // 4, images.shape[-1] // 4)
    head = load_sensitivity_head(args.head_checkpoint, device) if args.head_checkpoint else None
    prep = precompute(bundle, images, sources, 1, {n: e["ranking"] for n, e in rankings.items()}, head,
                      spatial_hw=spatial_hw)
    clean_psnr = float(np.mean(prep["clean_psnr"]))
    print(f"{count} Kodak images {tuple(images.shape[-2:])}: clean PSNR {clean_psnr:.2f} dB")
    if clean_psnr < 24.0:
        print("WARNING: clean PSNR < 24 dB. The CIFAR model does not generalise to this resolution; fine-tune "
              "with experiments/train_highres.py and evaluate that checkpoint (roadmap C.1 fix).")
    example = scheme_catalog(bundle.channels, spatial_hw[0] * spatial_hw[1], args.packet_size, args.overhead,
                             names=["uniform", schemes_arg[0]])
    for name, scheme in example.items():
        print(f"{name}: {len(scheme.tiers)} RS blocks, {sum(k for _, k in scheme.tiers)} data / "
              f"{scheme.total_packets} packets (max block {max(n for n, _ in scheme.tiers)})")

    run = wandb_utils.init_run(args.wandb, args.wandb_project, args.wandb_name or "eval-kodak", vars(args),
                               tags=["kodak"], group="phase3", job_type="eval")
    payload = {"args": vars(args), "device": "cpu" if args.workers > 1 else str(device), "specs": specs,
               "X": images, "prep": prep, "names": names}
    results = run_tasks(module_function(MODULE, "run_cell"), list(itertools.product(seeds, rates)),
                        module_function(MODULE, "build_context"), payload, args.workers, "kodak cells",
                        lambda task, result: wandb_utils.log_metrics(
                            run, {"seed": task[0], "erasure_rate": task[1],
                                  **{f"{r['method']}/psnr_db": r["psnr_db_mean_per_image"] for r in result[0]}}))
    rows = [row for cell_rows, _ in results for row in cell_rows]
    per_image = {k: v for _, cell in results for k, v in cell.items()}

    metadata = build_provenance(
        args=args, checkpoint_path=args.checkpoint, global_seed=args.global_seed, device=device,
        extra_files={f"ranking_{n}": e["info"]["path"] for n, e in rankings.items()},
        split_info={"dataset": "kodak", "images": names},
        extra={"evaluation": {"clean_psnr_mean": clean_psnr, "images_per_seed": count,
                              "image_size": list(images.shape[-2:]), "packet_size": args.packet_size,
                              "channel_seeds": seeds, "erasure_rates": rates, "importance_sources": sources,
                              "lpips_available": any("lpips_mean" in r for r in rows),
                              "gf256_note": "tiers split into sub-blocks of <= 255 packets (coding/scaling.py)"}})
    outputs = write_sweep_outputs(args.output_dir, args.name, rows, metadata, per_image, cdf_rate=0.4,
                                  title="Kodak (768x512) erasure sweep", metrics=SUMMARY_METRICS)
    # Perceptual gains alongside PSNR
    notes = []
    for metric in ("ms_ssim_mean", "lpips_mean"):
        if metric in rows[0]:
            paired = [p for p in paired_vs_baseline(rows, "uniform", metric, ("erasure_rate",))
                      if abs(p["erasure_rate"] - 0.4) < 1e-9]
            if paired:
                best = max(paired, key=lambda p: p["mean"] if metric == "ms_ssim_mean" else -p["mean"])
                notes.append(f"{metric} at loss 0.4: best {best['method']} {best['mean']:+.4f} vs uniform "
                             f"(roadmap target: ΔMS-SSIM ≥ +0.04).")
    with open(outputs["markdown"], "a", encoding="utf-8") as handle:
        handle.write("\n# Perceptual metrics\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    print("\n".join(f"- {line}" for line in outputs["insights"] + notes))
    wandb_utils.log_table(run, "seed_summary", outputs["summary"])
    wandb_utils.log_images(run, "figures", outputs["figures"])
    wandb_utils.finish(run)


if __name__ == "__main__":
    main()
