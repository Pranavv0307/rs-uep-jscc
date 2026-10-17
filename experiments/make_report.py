"""Track E: paper figures, headline tables and one consolidated insights file.

Reads the JSON written by the sweep scripts (each carries a seed summary, paired
gains and provenance) and produces, in ``--out_dir``:

* ``fig3_primary``        no RS / uniform / attention-UEP / predicted-UEP / oracle-UEP curves
* ``fig3_gain``           paired gain over uniform with 95% bands
* ``fig_model_comparison``  uniform vs best UEP for every ``--compare`` model
                          (base, erasure-robust, with inpainter, ...)
* ``fig5_cliff``          Deep JSCC vs codec+RS (``--digital``)
* ``fig_kn_sweep``        gain vs bandwidth ratio (``--kn ratio=path``)
* ``headline.md``         the main table: per loss rate, each method's PSNR ± std and paired gain ± CI
* ``INSIGHTS.md``         every automatically derived finding, section by section

Every figure is a vector PDF plus a 300-dpi PNG; colours are colour-blind safe and
each method has its own marker, so the figures survive black-and-white printing.

    python -m experiments.make_report --main results/final/final_test.json \\
        --compare base=results/final/final_test.json --compare robust=results/phase2/robust_test.json \\
        --digital results/phase4/digital/digital.json --kn 0.0625=results/phase3/kn_0.0625/kn.json \\
        --out_dir results/report
"""
import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from experiments import plotting, wandb_utils
from experiments.reporting import load_result, sweep_insights
from experiments.reproducibility import write_json
from experiments.stats_utils import crossover

PRIMARY_SOURCES = ("attention", "predicted", "oracle", "oracle_perimage")


def best_methods_by_source(summary: Sequence[dict], high_loss_from: float = 0.35) -> Dict[str, str]:
    """For each importance source, the scheme with the highest mean PSNR over the high-loss region."""
    scores = defaultdict(lambda: defaultdict(list))
    for row in summary:
        if "@" not in row["method"] or row["erasure_rate"] < high_loss_from:
            continue
        scheme, source = row["method"].split("@", 1)
        scores[source][row["method"]].append(row["psnr_db_mean_per_image__mean"])
    return {source: max(methods, key=lambda m: np.mean(methods[m])) for source, methods in scores.items()}


def relabel(summary: Sequence[dict], prefix: str, keep: Optional[Dict[str, str]] = None) -> List[dict]:
    """Copy rows with ``method`` renamed to ``"<prefix>: <alias>"``; ``keep`` maps method -> alias."""
    rows = []
    for row in summary:
        if keep is not None and row["method"] not in keep:
            continue
        alias = keep[row["method"]] if keep else row["method"]
        rows.append({**row, "method": f"{prefix}: {alias}" if prefix else alias})
    return rows


def headline_markdown(summary, paired, chosen: Dict[str, str], baseline="uniform") -> str:
    rates = sorted({r["erasure_rate"] for r in summary})
    by = {(r["method"], r["erasure_rate"]): r for r in summary}
    gain = {(p["method"], p["erasure_rate"]): p for p in paired}
    names = [m for m in ("no_rs", baseline) if any(r["method"] == m for r in summary)] + list(chosen.values())
    lines = ["# Headline table (mean ± std over channel seeds; gain = paired difference to uniform, 95% CI)\n",
             "| loss | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for rate in rates:
        cells = []
        for name in names:
            row = by.get((name, rate))
            if row is None:
                cells.append("n/a")
                continue
            std = row.get("psnr_db_mean_per_image__std_over_seeds")
            text = f"{row['psnr_db_mean_per_image__mean']:.2f}" + (f" ± {std:.2f}" if std is not None else "")
            g = gain.get((name, rate))
            if g and name != baseline:
                text += f" ({g['mean']:+.2f}" + (f" ± {g['ci95_half_width']:.2f}" if g["ci95_half_width"] else "") + \
                        (" *" if g["significant"] else "") + ")"
            cells.append(text)
        lines.append(f"| {rate:g} | " + " | ".join(cells) + " |")
    lines.append("\n`*` = 95% interval excludes zero.")
    return "\n".join(lines) + "\n"


def penalty_and_peak(paired: Sequence[dict], summary: Sequence[dict]) -> Dict[str, Optional[float]]:
    """Worst gain below break-even (low-loss penalty), peak gain, and break-even rate of the best method."""
    by_rate = defaultdict(list)
    for p in paired:
        by_rate[p["erasure_rate"]].append(p["mean"])
    rates = sorted(by_rate)
    best = [max(by_rate[r]) for r in rates]
    return {"worst_best_gain_db": float(min(best)) if best else None,
            "peak_gain_db": float(max(best)) if best else None,
            "break_even_rate": crossover(rates, best) if best else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--main", required=True, help="final erasure sweep JSON")
    parser.add_argument("--compare", action="append", default=[], help="label=path (repeatable)")
    parser.add_argument("--digital", default=None)
    parser.add_argument("--kn", action="append", default=[], help="ratio=path (repeatable)")
    parser.add_argument("--extra_insights", action="append", default=[],
                        help="label=path to any JSON with an 'insights' list (multiseed, transfer, head, ...)")
    parser.add_argument("--out_dir", default="results/report")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="rs-uep-jscc")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    main_result = load_result(args.main)
    summary, paired = main_result["seed_summary"], main_result["paired_vs_baseline"]
    chosen = best_methods_by_source(summary)
    figures: List[str] = []

    # Figure 3: primary curves
    aliases = {}
    for baseline_name in ("no_rs", "uniform"):
        if any(r["method"] == baseline_name for r in summary):
            aliases[baseline_name] = baseline_name
    for source in PRIMARY_SOURCES:
        if source in chosen:
            aliases[chosen[source]] = f"{source} UEP ({chosen[source].split('@')[0]})"
    primary = relabel(summary, "", aliases)
    figures += plotting.plot_metric_curves(primary, list(aliases.values()), os.path.join(args.out_dir, "fig3_primary"),
                                           title="PSNR vs packet erasure rate (error bars: std over seeds)")
    gain_rows = [{**p, "method": aliases[p["method"]]} for p in paired if p["method"] in aliases]
    figures += plotting.plot_paired_gain(gain_rows, [a for m, a in aliases.items() if m != "uniform"],
                                         os.path.join(args.out_dir, "fig3_gain"),
                                         title="Gain over uniform RS (paired, 95% CI)")
    with open(os.path.join(args.out_dir, "headline.md"), "w") as handle:
        handle.write(headline_markdown(summary, paired, {s: chosen[s] for s in PRIMARY_SOURCES if s in chosen}))

    sections = [("Main erasure sweep", main_result.get("insights", sweep_insights(summary, paired)))]
    stats = {"main": penalty_and_peak(paired, summary)}

    # Model comparison
    if args.compare:
        combined, table_rows = [], []
        for item in args.compare:
            label, _, path = item.partition("=")
            result = load_result(path)
            model_summary, model_paired = result["seed_summary"], result["paired_vs_baseline"]
            best = best_methods_by_source(model_summary)
            source = "oracle" if "oracle" in best else next(iter(best), None)
            keep = {"uniform": "uniform"}
            if source:
                keep[best[source]] = f"{source} UEP"
            combined += relabel(model_summary, label, keep)
            stats[label] = penalty_and_peak(model_paired, model_summary)
            table_rows.append((label, stats[label]))
        names = sorted({r["method"] for r in combined})
        figures += plotting.plot_metric_curves(combined, names, os.path.join(args.out_dir, "fig_model_comparison"),
                                               title="Models compared (uniform RS vs best UEP)", legend_cols=1)
        notes = [f"{label}: worst best-method gain {s['worst_best_gain_db']:+.2f} dB, peak gain "
                 f"{s['peak_gain_db']:+.2f} dB, break-even at loss "
                 f"{'never' if s['break_even_rate'] is None else format(s['break_even_rate'], '.2f')}."
                 for label, s in table_rows]
        if len(table_rows) > 1:
            base_penalty = table_rows[0][1]["worst_best_gain_db"]
            for label, s in table_rows[1:]:
                notes.append(f"Low-loss penalty change {table_rows[0][0]} -> {label}: "
                             f"{base_penalty:+.2f} dB -> {s['worst_best_gain_db']:+.2f} dB "
                             f"({s['worst_best_gain_db'] - base_penalty:+.2f} dB).")
        sections.append(("Model comparison (low-loss penalty, peak gain)", notes))

    # Digital baseline
    if args.digital:
        digital = load_result(args.digital)
        d_summary = digital["seed_summary"]
        merged = relabel(summary, "", aliases) + relabel(d_summary, "", None)
        keep_digital = [r["method"] for r in d_summary if r["method"].endswith("+rs")]
        names = list(aliases.values()) + sorted(set(keep_digital))
        figures += plotting.plot_metric_curves(merged, names, os.path.join(args.out_dir, "fig5_cliff"),
                                               title="Deep JSCC vs codec + RS (same bytes, same losses)")
        notes = []
        by = defaultdict(dict)
        for row in merged:
            by[row["method"]][row["erasure_rate"]] = row["psnr_db_mean_per_image__mean"]
        jscc = next((a for m, a in aliases.items() if "@" in m), "uniform")
        for codec in sorted(set(keep_digital)):
            clean_c, clean_j = by[codec].get(0.0), by[jscc].get(0.0)
            if clean_c is not None and clean_j is not None:
                notes.append(f"Clean: {codec} {clean_c:.2f} dB vs {jscc} {clean_j:.2f} dB.")
            below = [r for r in sorted(by[codec]) if r in by[jscc] and by[codec][r] < by[jscc][r]]
            notes.append(f"{codec} falls below {jscc} from loss = {below[0]:g}." if below else
                         f"{codec} never falls below {jscc} in this sweep.")
            if 0.35 in by[codec] and 0.35 in by[jscc]:
                notes.append(f"At loss 0.35: {codec} {by[codec][0.35]:.2f} dB vs {jscc} {by[jscc][0.35]:.2f} dB.")
        sections.append(("Digital separation baseline", notes))

    # k/n sweep
    if args.kn:
        points = []
        for item in args.kn:
            ratio, _, path = item.partition("=")
            result = load_result(path)
            best = best_methods_by_source(result["seed_summary"])
            source = "oracle" if "oracle" in best else next(iter(best), None)
            if source is None:
                continue
            for p in result["paired_vs_baseline"]:
                if p["method"] == best[source]:
                    points.append((float(ratio), p["erasure_rate"], p["mean"], p["ci95_half_width"] or 0.0))
        if points:
            import matplotlib.pyplot as plt

            fig, axis = plt.subplots(figsize=(4.4, 3.4))
            for i, rate in enumerate(sorted({p[1] for p in points if p[1] >= 0.35})):
                series = sorted((p for p in points if p[1] == rate), key=lambda p: p[0])
                axis.errorbar([p[0] for p in series], [p[2] for p in series], yerr=[p[3] for p in series], capsize=2.5,
                              label=f"loss {rate:g}", linewidth=1.3, **plotting.style_for(f"kn{rate}"))
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set(xlabel="Bandwidth ratio k/n", ylabel="Oracle-UEP gain over uniform (dB)",
                     title="Does the benefit depend on the bandwidth budget?")
            axis.set_xscale("log")
            axis.legend()
            figures += plotting.save_figure(fig, os.path.join(args.out_dir, "fig_kn_sweep"))
            best_point = max(points, key=lambda p: p[2])
            sections.append(("Bandwidth-ratio sweep", [
                f"Largest gain {best_point[2]:+.2f} dB at k/n = {best_point[0]:.4f}, loss {best_point[1]:g}.",
                "Gain is positive at every k/n in the high-loss region." if all(p[2] > 0 for p in points if p[1] >= 0.4)
                else "Gain is NOT positive at every k/n in the high-loss region: the principle is budget dependent."]))

    for item in args.extra_insights:
        label, _, path = item.partition("=")
        data = load_result(path)
        if data.get("insights"):
            sections.append((label, data["insights"]))

    with open(os.path.join(args.out_dir, "INSIGHTS.md"), "w") as handle:
        handle.write("# Insights (generated from the result files, not hand-written)\n\n")
        for title, lines in sections:
            handle.write(f"## {title}\n\n" + "\n".join(f"- {line}" for line in lines) + "\n\n")
    write_json(os.path.join(args.out_dir, "report_stats.json"), {"stats": stats, "chosen": chosen,
                                                                  "sources": {"main": args.main}})
    print("\n".join(f"## {t}\n" + "\n".join(f"- {l}" for l in ls) for t, ls in sections))
    print(f"figures: {len(figures)} files in {args.out_dir}")

    run = wandb_utils.init_run(args.wandb, args.wandb_project, "report", vars(args), job_type="report", group="final")
    wandb_utils.log_images(run, "report", figures)
    wandb_utils.finish(run)


if __name__ == "__main__":
    main()
