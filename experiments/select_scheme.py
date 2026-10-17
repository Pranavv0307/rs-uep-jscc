"""Choose the RS tier allocation on the VALIDATION sweep, never on the test set.

Roadmap Rule 1 forbids tuning code rates after seeing test numbers. This script
reads a sweep produced with ``eval_rs_uep --split val`` and records, per
importance source, which scheme is best at each loss rate (and which is best
overall in the high-loss region). The final test run is then restricted to those
choices, so the test sweep contains no selection step.

    python -m experiments.select_scheme --sweep results/phase1/val_sweep/val_sweep.json \\
        --output results/phase1/val_sweep/selected_schemes.json
"""
import argparse
import os
from collections import defaultdict

import numpy as np

from experiments import plotting
from experiments.reproducibility import write_json
from experiments.reporting import load_summary


def select(summary, high_loss_from: float):
    table = defaultdict(lambda: defaultdict(dict))  # source -> scheme -> rate -> (mean, std)
    for row in summary:
        if "@" not in row["method"]:
            continue
        scheme, source = row["method"].split("@", 1)
        table[source][scheme][row["erasure_rate"]] = (row["psnr_db_mean_per_image__mean"],
                                                      row.get("psnr_db_mean_per_image__std_over_seeds") or 0.0)
    selection = {}
    for source, by_scheme in table.items():
        rates = sorted({r for rates_ in by_scheme.values() for r in rates_})
        per_rate = {}
        for rate in rates:
            candidates = {s: v[rate][0] for s, v in by_scheme.items() if rate in v}
            per_rate[f"{rate:g}"] = max(candidates, key=candidates.get)
        high = [r for r in rates if r >= high_loss_from]
        overall = {s: float(np.mean([v[r][0] for r in high if r in v])) for s, v in by_scheme.items()
                   if any(r in v for r in high)}
        selection[source] = {"per_rate": per_rate,
                             "best_overall_high_loss": max(overall, key=overall.get) if overall else None,
                             "high_loss_mean_psnr": overall}
    return selection, table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, help="JSON written by eval_rs_uep on the validation split")
    parser.add_argument("--output", default=None)
    parser.add_argument("--high_loss_from", type=float, default=0.35)
    args = parser.parse_args()

    summary, metadata = load_summary(args.sweep)
    split = (metadata or {}).get("split", {}) or {}
    if split.get("split") == "test":
        raise SystemExit("refusing to select schemes on the TEST split (roadmap Rule 1); pass a validation sweep")
    selection, table = select(summary, args.high_loss_from)
    union = sorted({scheme for entry in selection.values() for scheme in entry["per_rate"].values()})
    output = args.output or os.path.join(os.path.dirname(args.sweep), "selected_schemes.json")
    write_json(output, {"source_sweep": args.sweep, "selected_on": split.get("split", "unknown"),
                        "selection": selection, "schemes_union": union,
                        "schemes_arg": ",".join(union)})

    out_dir = os.path.dirname(output) or "."
    for source, by_scheme in table.items():
        schemes = sorted(by_scheme)
        rates = sorted({r for v in by_scheme.values() for r in v})
        matrix = np.array([[by_scheme[s].get(r, (np.nan, 0))[0] for r in rates] for s in schemes])
        plotting.plot_heatmap(matrix, schemes, [f"{r:g}" for r in rates],
                              os.path.join(out_dir, f"scheme_selection_{source.replace(':', '_')}"),
                              title=f"Validation PSNR by scheme ({source})", cbar_label="dB", fmt="{:.1f}")
    for source, entry in selection.items():
        print(f"{source}: best overall in the high-loss region = {entry['best_overall_high_loss']}; "
              f"per rate = {entry['per_rate']}")
    print(f"--schemes {','.join(union)}\nsaved {output}")


if __name__ == "__main__":
    main()
