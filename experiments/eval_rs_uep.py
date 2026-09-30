"""Evaluate fixed-budget RS allocations on the ADJSCC-Q test path.

The evaluator compares schemes on the same images and, for every 96-packet
scheme, the same sampled packet-erasure mask. Channel-order side information
is assumed reliable and is counted as 16 bytes per image for tiered methods.

Examples::

    python -m experiments.eval_rs_uep --max_images 1024 --seeds 0,1,2
    python -m experiments.eval_rs_uep --max_images 1000 --plot_output results/eval/rs_uep.png
"""
import argparse
import json
import os
from collections import defaultdict

import torch

from coding.erasure_channel import apply_erasure_mask, sample_erasure_mask
from coding.importance import channel_order_from_attention
from coding.rs_pipeline import (
    NO_RS,
    SCHEME_CANDIDATES,
    decode_blocks,
    dequantize_with_fallback,
    encode_blocks,
    random_channel_order,
)
from data.cifar10 import get_cifar10_loaders
from experiments.eval_adjsccq_clean import DEFAULT_CHECKPOINT, load_adjsccq_checkpoint
from experiments.metrics import psnr_per_image, ssim


DEFAULT_ERASURE_RATES = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5)
DEFAULT_METHODS = tuple(SCHEME_CANDIDATES)


def _parse_csv_numbers(value, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _method_order(method, attention_order, batch_size, generator):
    if method == "uniform":
        return None, 0
    if method.endswith("_random"):
        return random_channel_order(batch_size, 16, generator), 16
    return attention_order, 16


def _method_scheme(method):
    return NO_RS if method == "no_rs" else SCHEME_CANDIDATES[method.removesuffix("_random")]


def _method_names(include_no_rs, include_random):
    names = list(DEFAULT_METHODS)
    if include_no_rs:
        names.insert(0, "no_rs")
    if include_random:
        names.extend(f"{name}_random" for name in DEFAULT_METHODS if name != "uniform")
    return tuple(names)


def _new_stats(method, scheme, erasure_rate, seed, checkpoint):
    return {
        "method": method,
        "scheme_tiers_packets": [list(tier) for tier in scheme.tiers],
        "packet_size": scheme.packet_size,
        "data_symbols": scheme.data_symbols,
        "coding_symbols": scheme.transmitted_symbols,
        "coding_overhead_fraction": scheme.overhead,
        "erasure_rate": erasure_rate,
        "seed": seed,
        "checkpoint": checkpoint,
        "images": 0,
        "psnr_sum": 0.0,
        "psnr_squared_sum": 0.0,
        "ssim_sum": 0.0,
        "failed_symbols": 0,
        "block_failures": [0 for _ in scheme.tiers],
        "erased_packets": 0,
        "total_packets": 0,
        "side_info_bytes_per_image": 0,
    }


def _finish_stats(stats):
    images = stats["images"]
    mean_psnr = stats["psnr_sum"] / images
    variance = max(stats["psnr_squared_sum"] / images - mean_psnr ** 2, 0.0)
    side_info = stats["side_info_bytes_per_image"]
    total_bytes = stats["coding_symbols"] + side_info
    data_symbols = stats["data_symbols"]
    return {
        **stats,
        "psnr_db_mean_per_image": mean_psnr,
        "psnr_db_std_per_image": variance ** 0.5,
        "ssim_mean": stats["ssim_sum"] / images,
        "failed_symbol_fraction": stats["failed_symbols"] / (images * data_symbols),
        "block_failure_rate_by_tier": [count / images for count in stats["block_failures"]],
        "realized_erasure_rate": stats["erased_packets"] / stats["total_packets"],
        "side_info_bytes_total": images * side_info,
        "total_transmitted_bytes_per_image": total_bytes,
        "total_overhead_fraction_including_side_info": total_bytes / data_symbols - 1.0,
    }


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, config = load_adjsccq_checkpoint(args.checkpoint, device)
    snr_db = config["af_conditioning"]["nominal_snr_db"]
    _, loader = get_cifar10_loaders(args.data_dir, batch_size=args.batch_size, num_workers=0)
    methods = _method_names(args.include_no_rs, args.include_random_control)
    rates = _parse_csv_numbers(args.erasure_rates, float)
    seeds = _parse_csv_numbers(args.seeds, int)
    results = []

    for seed in seeds:
        for erasure_rate in rates:
            accumulators = {
                method: _new_stats(method, _method_scheme(method), erasure_rate, seed, args.checkpoint)
                for method in methods
            }
            for batch_index, (x, _) in enumerate(loader):
                if args.max_images is not None:
                    remaining = args.max_images - batch_index * args.batch_size
                    if remaining <= 0:
                        break
                    x = x[:remaining]
                x = x.to(device)
                batch_size = x.shape[0]
                _, attention, indices = encoder(x, snr_db, return_attn=True, return_indices=True)
                attention_order = channel_order_from_attention(attention["af5_bottleneck"])
                mask_generator = torch.Generator().manual_seed(seed * 100000 + batch_index)
                mask_96 = sample_erasure_mask(batch_size, 96, erasure_rate, mask_generator)
                mask_64 = sample_erasure_mask(batch_size, 64, erasure_rate, mask_generator)

                for method in methods:
                    scheme = _method_scheme(method)
                    order_generator = torch.Generator().manual_seed(seed * 100000 + batch_index + 50000)
                    order, side_info_bytes = _method_order(
                        method, attention_order, batch_size, order_generator
                    )
                    packets = encode_blocks(indices, scheme, order)
                    mask = mask_64 if scheme is NO_RS else mask_96
                    received = apply_erasure_mask(packets, mask).received
                    decoded = decode_blocks(received, scheme, order)
                    latent = dequantize_with_fallback(
                        encoder.quantizer, decoded.indices, decoded.failed
                    )
                    reconstruction = decoder(latent, snr_db)
                    psnr_values = psnr_per_image(reconstruction, x)
                    ssim_values = ssim(reconstruction, x)
                    stats = accumulators[method]
                    stats["images"] += batch_size
                    stats["psnr_sum"] += psnr_values.sum().item()
                    stats["psnr_squared_sum"] += (psnr_values ** 2).sum().item()
                    stats["ssim_sum"] += ssim_values.sum().item()
                    stats["failed_symbols"] += decoded.failed.sum().item()
                    stats["block_failures"] = [
                        total + int(failed)
                        for total, failed in zip(stats["block_failures"], decoded.block_failed.sum(dim=0).tolist())
                    ]
                    stats["erased_packets"] += received.erased.sum().item()
                    stats["total_packets"] += received.erased.numel()
                    stats["side_info_bytes_per_image"] = side_info_bytes

            results.extend(_finish_stats(accumulators[method]) for method in methods)
    return results


def write_plot(rows, output):
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    figure, axis = plt.subplots(figsize=(11, 6))
    for method, method_rows in grouped.items():
        method_rows.sort(key=lambda row: row["erasure_rate"])
        axis.plot(
            [row["erasure_rate"] for row in method_rows],
            [row["psnr_db_mean_per_image"] for row in method_rows],
            marker="o",
            label=method,
        )
    axis.set(xlabel="Packet erasure rate", ylabel="Mean PSNR (dB)", title="RS-UEP validation sweep")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_dir", default="./cifar10_data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_images", type=int, default=1024)
    parser.add_argument("--erasure_rates", default=",".join(map(str, DEFAULT_ERASURE_RATES)))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--include_no_rs", action="store_true")
    parser.add_argument("--include_random_control", action="store_true")
    parser.add_argument("--output", default="results/eval/rs_uep_sweep.json")
    parser.add_argument("--plot_output", default="results/eval/rs_uep_sweep.png")
    parser.add_argument("--wandb", action="store_true", help="Log results and plots to wandb")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    args = parser.parse_args()

    rows = evaluate(args)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(rows, handle, indent=2)
    
    if args.plot_output:
        os.makedirs(os.path.dirname(args.plot_output), exist_ok=True)
        write_plot(rows, args.plot_output)
        
    print(f"saved {len(rows)} rows -> {args.output}")
    if args.plot_output:
        print(f"saved plot -> {args.plot_output}")

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))
        if rows:
            # Log the full dataset as a wandb Table so it can be grouped/graphed in the UI
            table = wandb.Table(columns=list(rows[0].keys()))
            for row in rows:
                table.add_data(*[row[k] for k in table.columns])
            wandb.log({"rs_uep_sweep_data": table})
            
        if args.plot_output:
            # Log the generated matplotlib figure directly
            wandb.log({"rs_uep_sweep_plot": wandb.Image(args.plot_output)})
        wandb.finish()


if __name__ == "__main__":
    main()