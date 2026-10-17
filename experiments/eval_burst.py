"""Step D.1: tiered RS under bursty (Gilbert-Elliott) packet loss.

Loss is sampled **on the transmission timeline** from a two-state Markov chain
(Good loses ~1%, Bad ~85%) whose long-run loss rate and mean burst length are
set independently, and an iid (Bernoulli) channel with the same loss rate is run
as the memoryless reference. Every method sees the same timeline losses; each
method reads the timeline slots its own packets occupy.

Two packet orderings are compared for every tiered scheme:

* ``sequential`` -- tier 0 first, then tier 1, ... (a burst can wipe a whole tier)
* ``round_robin`` (suffix ``#rr``) -- packets of every coding block are spread
  evenly over the timeline, so a burst costs every tier only its proportional share

``uniform`` is a single MDS block, so for it only the *number* of losses matters,
not where they fall; bursts therefore only change the variance of that number.

Good result (roadmap): interleaved UEP beats uniform by > 3 dB for burst length >= 4.

    python -m experiments.eval_burst --checkpoint <ckpt> --oracle_ranking_file <ranking.json> \\
        --split val --avg_losses 0.35 --burst_lengths 1,2,4,8,16 --seeds 0,1,2,3,4 \\
        --output_dir results/phase4/burst
"""
import itertools
import os
from collections import defaultdict

import numpy as np
import torch

from coding.erasure_channel import GilbertElliottChannel, mean_loss_run_length, sample_master_mask
from coding.interleave import packet_mask_from_timeline
from coding.rs_pipeline import SCHEME_CANDIDATES
from experiments import wandb_utils
from experiments.eval_rs_uep import (
    build_context,
    build_parser,
    evaluation_metadata,
    parse_csv,
    prepare_data,
)
from experiments.pipeline_core import (
    SUMMARY_METRICS,
    CellAccumulator,
    batch_ranges,
    make_method_specs,
    method_order,
    module_function,
    run_tasks,
    transmit,
)
from experiments.reporting import write_sweep_outputs
from experiments.reproducibility import seed_everything, write_json

MODULE = "experiments.eval_burst"


def run_cell(context, task):
    seed, avg_loss, burst_length = task
    args = context.args
    constants = {"erasure_rate": avg_loss, "avg_loss": avg_loss, "burst_length": burst_length, "seed": seed,
                 "dataset": args["dataset"], "split": args["split"], "checkpoint": args["checkpoint"],
                 "channels": context.channels}
    accumulators = {spec.name: CellAccumulator(spec, context.schemes[spec.scheme], context.channels, constants)
                    for spec in context.specs}
    channel = None if burst_length <= 1 else GilbertElliottChannel.from_average_loss(avg_loss, burst_length)
    lost_sum = lost_total = 0
    run_lengths = []
    for batch_index, start, stop in batch_ranges(len(context.X), context.batch_size):
        x = context.X[start:stop].to(context.device)
        idx = context.prep["idx"][start:stop].long().to(context.device)
        generator = torch.Generator().manual_seed(seed * 100000 + batch_index)
        if channel is None:
            timeline = sample_master_mask(stop - start, avg_loss, generator, num_packets=context.master_n)
        else:
            timeline = channel.sample_mask(stop - start, context.master_n, generator)
        lost_sum += int(timeline.sum())
        lost_total += timeline.numel()
        run_lengths.append(mean_loss_run_length(timeline))
        for spec in context.specs:
            scheme = context.schemes[spec.scheme]
            order_generator = torch.Generator().manual_seed(seed * 100000 + batch_index + 50000)
            order = method_order(spec, context.prep["orders"], start, stop, context.channels, order_generator,
                                 context.device)
            mask = packet_mask_from_timeline(timeline, scheme, spec.interleave)
            recon, decoded, erased, total = transmit(context.bundle, idx, scheme, order, mask, context.snr_db)
            accumulators[spec.name].add(recon, x, decoded, erased, total)
    rows, per_image = [], {}
    for name, accumulator in accumulators.items():
        row, psnr = accumulator.finish()
        row["timeline_loss_rate"] = lost_sum / lost_total
        row["timeline_mean_burst"] = float(np.mean(run_lengths))
        rows.append(row)
        per_image[f"{name}__s{seed}__r{avg_loss:.3f}__L{burst_length:g}"] = psnr
    return rows, per_image


def burst_notes(rows, baseline="uniform"):
    """Burst-specific findings beyond the generic sweep insights."""
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["method"], row["burst_length"])].setdefault(row["seed"], row["psnr_db_mean_per_image"])
    lengths = sorted({length for _, length in by_key})
    methods = sorted({method for method, _ in by_key})
    notes = []
    for length in lengths:
        base = by_key.get((baseline, length))
        if not base:
            continue
        gains = {}
        for method in methods:
            values = by_key.get((method, length))
            if values and method != baseline:
                seeds = sorted(set(values) & set(base))
                gains[method] = float(np.mean([values[s] - base[s] for s in seeds]))
        if gains:
            best = max(gains, key=gains.get)
            notes.append(f"Burst length {length:g}: best method {best} at {gains[best]:+.2f} dB over {baseline}"
                         + ("  (meets the >3 dB target)" if length >= 4 and gains[best] > 3 else ""))
    for method in methods:
        if method.endswith("#rr"):
            plain = method[:-3]
            for length in lengths:
                a, b = by_key.get((method, length)), by_key.get((plain, length))
                if a and b:
                    seeds = sorted(set(a) & set(b))
                    gain = float(np.mean([a[s] - b[s] for s in seeds]))
                    if length >= 4:
                        notes.append(f"Interleaving {plain}: {gain:+.2f} dB at burst length {length:g}")
    return notes


def main():
    parser = build_parser(__doc__)
    parser.add_argument("--avg_losses", default="0.35", help="long-run loss rates")
    parser.add_argument("--burst_lengths", default="1,2,4,8,16",
                        help="mean Bad-state burst length in packets (1 = iid Bernoulli reference)")
    parser.add_argument("--interleave", default="sequential,round_robin")
    parser.set_defaults(schemes="three_tier_25_50_25,three_tier_25_25_50", importance="oracle",
                        output_dir="results/phase4/burst", name="burst")
    args = parser.parse_args()
    seed_everything(args.global_seed)

    schemes = list(SCHEME_CANDIDATES) if args.schemes == "all" else list(parse_csv(args.schemes))
    seeds = parse_csv(args.seeds, int)
    losses, lengths = parse_csv(args.avg_losses, float), parse_csv(args.burst_lengths, float)
    interleaves = parse_csv(args.interleave)
    data = prepare_data(args, note="eval_burst")
    specs = make_method_specs(schemes, data["sources"], args.include_no_rs, args.include_random_control, interleaves)
    print(f"{len(data['images'])} images, {len(specs)} methods, losses {losses}, burst lengths {lengths}")

    run = wandb_utils.init_run(args.wandb, args.wandb_project, args.wandb_name or "eval-burst", vars(args),
                               tags=["burst", args.split], group=args.wandb_group, job_type="eval")
    payload = {"args": vars(args), "device": "cpu" if args.workers > 1 else str(data["device"]), "specs": specs,
               "X": data["images"], "prep": data["prep"]}

    def log_cell(task, result):
        seed, loss, length = task
        wandb_utils.log_metrics(run, {"seed": seed, "avg_loss": loss, "burst_length": length,
                                      **{f"{r['method']}/psnr_db": r["psnr_db_mean_per_image"] for r in result[0]}})

    tasks = list(itertools.product(seeds, losses, lengths))
    results = run_tasks(module_function(MODULE, "run_cell"), tasks, build_context, payload, args.workers,
                        "burst cells", log_cell)
    rows = [row for cell_rows, _ in results for row in cell_rows]
    per_image = {} if args.no_per_image else {k: v for _, cell in results for k, v in cell.items()}

    metadata = evaluation_metadata(args, data, {
        "channel_seeds": list(seeds), "avg_losses": list(losses), "burst_lengths": list(lengths),
        "interleave_modes": list(interleaves), "good_loss": 0.01, "bad_loss": 0.85,
        "mask_protocol": "Gilbert-Elliott timeline mask shared by all methods; burst_length 1 = Bernoulli"})
    outputs = {}
    for loss in losses:
        subset = [row for row in rows if row["avg_loss"] == loss]
        out_dir = os.path.join(args.output_dir, f"loss_{loss:g}")
        outputs[loss] = write_sweep_outputs(
            out_dir, f"burst_p{loss:g}", subset, metadata, None, x_key="burst_length",
            group_keys=("method", "burst_length"), paired_group=("burst_length",),
            title=f"Burst-loss study, average loss {loss:g}", xlabel="Mean burst length (packets)",
            extra_keys=("timeline_mean_burst",),
            metrics=SUMMARY_METRICS + ("timeline_mean_burst", "timeline_loss_rate"))
        notes = burst_notes(subset)
        with open(outputs[loss]["markdown"], "a", encoding="utf-8") as handle:
            handle.write("\n# Burst-specific findings\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
        print(f"\n== average loss {loss:g} ==\n" + "\n".join(f"- {n}" for n in notes + outputs[loss]["insights"]))
        wandb_utils.log_table(run, f"seed_summary_p{loss:g}", outputs[loss]["summary"])
        wandb_utils.log_images(run, f"figures_p{loss:g}", outputs[loss]["figures"])
    if per_image:
        np.savez_compressed(os.path.join(args.output_dir, "burst.per_image.npz"), **per_image)
    write_json(os.path.join(args.output_dir, "burst_all.json"), {"metadata": metadata, "rows": rows})
    wandb_utils.finish(run)


if __name__ == "__main__":
    main()
