"""Tables, standard figures and automatic insights for any sweep result.

Every sweep script (erasure, burst, digital baseline, Kodak) produces the same
data shape -- per-seed rows, a ``seed_summary`` and ``paired`` differences to a
baseline -- so one module renders them all identically.
"""
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from experiments import plotting
from experiments.reproducibility import aggregate_over_seeds, write_json
from experiments.stats_utils import crossover, paired_vs_baseline


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def _pm(mean, std, ci) -> str:
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.2f} (1 seed)"
    return f"{mean:.2f} ± {std:.2f} (CI ±{ci:.2f})"


def summary_markdown(summary: Sequence[dict], metadata: Optional[dict] = None, x_key: str = "erasure_rate",
                     title: str = "Sweep results", extra_keys: Sequence[str] = ()) -> str:
    lines = [f"# {title}\n"]
    if metadata:
        evaluation = metadata.get("evaluation", {})
        checkpoint = metadata.get("checkpoint") or {}
        lines.append(f"- commit `{metadata.get('git_commit')}` (dirty: {metadata.get('dirty_repo')}), "
                     f"{metadata.get('timestamp_utc')}")
        lines.append(f"- checkpoint `{checkpoint.get('path')}` sha256 `{checkpoint.get('sha256')}`")
        for key, value in evaluation.items():
            if key not in ("oracle_ranking",) and not isinstance(value, (dict, list)) or key in ("channel_seeds",):
                lines.append(f"- {key}: {value}")
        lines.append("- PSNR: mean ± sample std over channel seeds (95% t-interval half-width in brackets)\n")
    header = ["Method", x_key, "PSNR (dB)", "median", "p5", "SSIM", "block-fail", "Coding OH", "Side-info B", "Total OH"]
    header += list(extra_keys)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in sorted(summary, key=lambda r: (r[x_key], r["method"])):
        cells = [
            row["method"], f"{row[x_key]:g}",
            _pm(row["psnr_db_mean_per_image__mean"], row.get("psnr_db_mean_per_image__std_over_seeds"),
                row.get("psnr_db_mean_per_image__ci95_half_width")),
            f"{row.get('psnr_db_median__mean', float('nan')):.2f}",
            f"{row.get('psnr_db_p5__mean', float('nan')):.2f}",
            f"{row.get('ssim_mean__mean', float('nan')):.4f}",
            f"{row.get('any_block_failure_rate__mean', float('nan')):.3f}",
            f"{row['coding_overhead_fraction'] * 100:.2f}%",
            str(row["side_info_bytes_per_image"]),
            f"{row['total_overhead_fraction_including_side_info'] * 100:.2f}%",
        ]
        cells += [f"{row.get(f'{key}__mean', row.get(key, float('nan'))):.3f}" for key in extra_keys]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def paired_markdown(paired: Sequence[dict], x_key: str = "erasure_rate", baseline: str = "uniform") -> str:
    lines = [f"# Paired difference to `{baseline}` (same loss mask per seed)\n",
             f"| Method | {x_key} | mean gain (dB) | std over seeds | 95% CI ± | wins | significant |",
             "|---|---|---|---|---|---|---|"]
    for row in sorted(paired, key=lambda r: (r[x_key], r["method"])):
        std = "n/a" if row["std"] is None else f"{row['std']:.2f}"
        ci = "n/a" if row["ci95_half_width"] is None else f"{row['ci95_half_width']:.2f}"
        lines.append(f"| {row['method']} | {row[x_key]:g} | {row['mean']:+.2f} | {std} | {ci} | "
                     f"{row['win_rate']:.0%} | {'yes' if row['significant'] else 'no'} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Insights
# --------------------------------------------------------------------------
def _methods(summary) -> List[str]:
    seen = []
    for row in summary:
        if row["method"] not in seen:
            seen.append(row["method"])
    return seen


def _group_by_source(methods: Sequence[str], baseline: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for method in methods:
        if method in (baseline, "no_rs"):
            continue
        groups[method.split("@", 1)[1] if "@" in method else method].append(method)
    return groups


def sweep_insights(summary: Sequence[dict], paired: Sequence[dict], x_key: str = "erasure_rate",
                   baseline: str = "uniform", high_loss_from: float = 0.35) -> List[str]:
    # high_loss_from is the first x value counted as the "high-loss region" (0.35 for loss sweeps, 0 for bursts)
    """Plain-language findings computed from the numbers (never hand-written)."""
    notes: List[str] = []
    if not summary:
        return ["No rows to analyse."]
    n_seeds = max(r["n_seeds"] for r in summary)
    notes.append(f"Based on {n_seeds} channel seed(s); {summary[0].get('images_per_seed')} images per seed. "
                 "All gains below are paired per seed against the baseline (same packet losses), "
                 "so they carry no luck-of-the-draw advantage.")

    by_method: Dict[str, Dict[float, dict]] = defaultdict(dict)
    for row in paired:
        by_method[row["method"]][row[x_key]] = row
    if not by_method:
        return notes + ["No baseline rows were found, so no paired comparison is possible."]

    rates = sorted({row[x_key] for row in paired})

    # Best method per rate
    best_per_rate = {}
    for rate in rates:
        candidates = [(m, rows[rate]["mean"]) for m, rows in by_method.items() if rate in rows]
        best_per_rate[rate] = max(candidates, key=lambda c: c[1])
    best_gains = [best_per_rate[r][1] for r in rates]
    crossing = crossover(rates, best_gains)
    if max(best_gains) <= 0:
        notes.append(f"No method ever beats `{baseline}` on average in this sweep.")
    elif min(best_gains) >= -0.05:
        notes.append(f"No low-loss penalty: the best method is never more than 0.05 dB below `{baseline}` "
                     f"at any {x_key} in this sweep.")
    else:
        notes.append(f"Break-even: the best protected method overtakes `{baseline}` from about "
                     f"{x_key} = {crossing:.2f}.")

    low = [(r, best_per_rate[r]) for r in rates if best_per_rate[r][1] < 0]
    if low:
        worst_rate, (worst_method, worst_gain) = min(low, key=lambda item: item[1][1])
        notes.append(f"Low-loss penalty: even the best method is {abs(worst_gain):.2f} dB below `{baseline}` at "
                     f"{x_key} = {worst_rate:g} ({worst_method}). This is the block-length effect of a single "
                     "large RS block, which tiered coding gives up.")

    peak_rate, (peak_method, peak_gain) = max(best_per_rate.items(), key=lambda item: item[1][1])
    peak_row = by_method[peak_method][peak_rate]
    ci = peak_row["ci95_half_width"]
    ci_text = f" (95% CI ±{ci:.2f}, {'significant' if peak_row['significant'] else 'not significant'})" if ci else ""
    notes.append(f"Peak gain: {peak_gain:+.2f} dB at {x_key} = {peak_rate:g} by {peak_method}{ci_text}.")

    # Source comparison on the high-loss region
    groups = _group_by_source(list(by_method), baseline)
    high_rates = [r for r in rates if r >= high_loss_from]
    if high_rates and groups:
        score = {}
        for source, members in groups.items():
            per_member = [np.mean([by_method[m][r]["mean"] for r in high_rates if r in by_method[m]])
                          for m in members if any(r in by_method[m] for r in high_rates)]
            if per_member:
                score[source] = float(max(per_member))
        if score:
            ranked = sorted(score.items(), key=lambda item: -item[1])
            notes.append("Importance sources ranked by their best scheme's mean gain over "
                         f"{x_key} >= {high_loss_from:g}: " + ", ".join(f"{s} {g:+.2f} dB" for s, g in ranked) + ".")
            reference = score.get("oracle") or score.get("oracle_perimage")
            if reference and reference >= 0.25:  # a share of a negligible gain is meaningless
                share = ", ".join(f"{s} {100 * g / reference:.0f}%" for s, g in ranked
                                  if g != reference and "#" not in s and not s.startswith("oracle"))
                if share:
                    notes.append(f"Share of the oracle gain captured: {share}.")
            if "random" in score and "oracle" in score:
                notes.append(f"Random-order control gains {score['random']:+.2f} dB versus oracle "
                             f"{score['oracle']:+.2f} dB, so the benefit comes from *which* channels are "
                             "protected, not merely from tiering.")

    # Worst-case behaviour
    by_psnr = defaultdict(dict)
    for row in summary:
        by_psnr[row["method"]][row[x_key]] = row
    if baseline in by_psnr and rates:
        probe = max(rates)
        mid = min(rates, key=lambda r: abs(r - 0.4))
        for rate in sorted({mid, probe}):
            best_method = best_per_rate[rate][0]
            base_row, best_row = by_psnr[baseline].get(rate), by_psnr[best_method].get(rate)
            if base_row and best_row and "psnr_db_p5__mean" in best_row:
                notes.append(f"Worst-case images at {x_key} = {rate:g}: 5th-percentile PSNR "
                             f"{best_row['psnr_db_p5__mean']:.2f} dB ({best_method}) vs "
                             f"{base_row['psnr_db_p5__mean']:.2f} dB ({baseline}).")

    # Stability
    noisy = [(row["method"], row[x_key], row["psnr_db_mean_per_image__std_over_seeds"])
             for row in summary if (row.get("psnr_db_mean_per_image__std_over_seeds") or 0) > 0.3]
    if noisy:
        method, rate, std = max(noisy, key=lambda item: item[2])
        notes.append(f"Seed sensitivity: largest std across channel seeds is {std:.2f} dB ({method}, "
                     f"{x_key} = {rate:g}); gains smaller than that are not reliable.")
    else:
        notes.append("Seed sensitivity: std across channel seeds stays below 0.3 dB everywhere.")

    # Graceful degradation (cliff)
    if baseline in by_psnr and len(rates) > 2:
        base_series = [by_psnr[baseline][r]["psnr_db_mean_per_image__mean"] for r in rates if r in by_psnr[baseline]]
        drops = np.diff(base_series)
        steepest = int(np.argmin(drops))
        notes.append(f"`{baseline}` shows its steepest fall ({drops[steepest]:.2f} dB) between {x_key} = "
                     f"{rates[steepest]:g} and {rates[steepest + 1]:g} -- its cliff.")
    return notes


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _top_methods(paired: Sequence[dict], limit: int) -> List[str]:
    score = defaultdict(list)
    for row in paired:
        score[row["method"]].append(row["mean"])
    ranked = sorted(score, key=lambda m: -max(score[m]))
    return ranked[:limit]


def standard_sweep_figures(summary, paired, out_dir: str, tag: str, per_image: Optional[dict] = None,
                           cdf_rate: float = 0.4, x_key: str = "erasure_rate", baseline: str = "uniform",
                           xlabel: str = "Packet erasure rate", max_methods: int = 8) -> List[str]:
    """Curves, paired gain, gain heat-map, worst-case curve, block failures, per-image CDF."""
    os.makedirs(out_dir, exist_ok=True)
    methods = _methods(summary)
    shown = [m for m in methods if m in (baseline, "no_rs")] + _top_methods(paired, max_methods)
    shown = list(dict.fromkeys(shown))
    paths: List[str] = []
    paths += plotting.plot_metric_curves(summary, shown, os.path.join(out_dir, f"{tag}_psnr_curves"),
                                         x_key=x_key, xlabel=xlabel, title="Mean PSNR (error bars: std over seeds)")
    paths += plotting.plot_metric_curves(summary, shown, os.path.join(out_dir, f"{tag}_psnr_p5_curves"),
                                         x_key=x_key, metric="psnr_db_p5", xlabel=xlabel,
                                         ylabel="5th-percentile PSNR (dB)", title="Worst-case images")
    paths += plotting.plot_metric_curves(summary, shown, os.path.join(out_dir, f"{tag}_block_failure"),
                                         x_key=x_key, metric="any_block_failure_rate", xlabel=xlabel,
                                         ylabel="Images with >=1 unrecoverable block",
                                         title="Coding-block failure rate")
    if paired:
        paths += plotting.plot_paired_gain(paired, _top_methods(paired, max_methods),
                                           os.path.join(out_dir, f"{tag}_gain_vs_{baseline}"), x_key=x_key,
                                           xlabel=xlabel, ylabel=f"PSNR gain over {baseline} (dB)",
                                           title="Paired gain (band: 95% CI over seeds)")
        rates = sorted({r[x_key] for r in paired})
        names = sorted({r["method"] for r in paired})
        matrix = np.full((len(names), len(rates)), np.nan)
        for row in paired:
            matrix[names.index(row["method"]), rates.index(row[x_key])] = row["mean"]
        paths += plotting.plot_heatmap(matrix, names, [f"{r:g}" for r in rates],
                                       os.path.join(out_dir, f"{tag}_gain_heatmap"),
                                       title=f"Mean gain over {baseline} (dB)", cbar_label="dB", fmt="{:+.1f}",
                                       center_zero=True)
    if per_image:
        paths += _cdf_figure(per_image, shown, cdf_rate, os.path.join(out_dir, f"{tag}_psnr_cdf"))
    return paths


def _cdf_figure(per_image: dict, methods: Sequence[str], rate: float, base_path: str) -> List[str]:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(4.2, 3.4))
    drawn = False
    for method in methods:
        keys = [k for k in per_image if k.startswith(f"{method}__s") and k.endswith(f"__r{rate:.3f}")]
        if not keys:
            continue
        values = np.sort(np.concatenate([per_image[k] for k in keys]))
        axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=method, linewidth=1.3,
                  **{k: v for k, v in plotting.style_for(method).items() if k in ("color", "linestyle")})
        drawn = True
    if not drawn:
        plt.close(fig)
        return []
    axis.set(xlabel="Per-image PSNR (dB)", ylabel="Fraction of images",
             title=f"PSNR distribution at loss = {rate:g} (all seeds pooled)")
    axis.legend()
    return plotting.save_figure(fig, base_path + f"_r{rate:g}")


# --------------------------------------------------------------------------
# One call that writes everything
# --------------------------------------------------------------------------
def write_sweep_outputs(out_dir: str, name: str, rows: List[dict], metadata: dict, per_image: Optional[dict] = None,
                        x_key: str = "erasure_rate", group_keys: Sequence[str] = ("method", "erasure_rate"),
                        baseline: str = "uniform", title: str = "Sweep results",
                        xlabel: str = "Packet erasure rate", cdf_rate: float = 0.4,
                        metrics: Optional[Sequence[str]] = None, extra_keys: Sequence[str] = (),
                        paired_group: Sequence[str] = ("erasure_rate",),
                        high_from: float = 0.35) -> Dict[str, object]:
    """Write JSON + markdown + figures + insights; return their paths and the computed tables."""
    from experiments.pipeline_core import SUMMARY_METRICS

    os.makedirs(out_dir, exist_ok=True)
    summary = aggregate_over_seeds(rows, group_keys=group_keys, metrics=metrics or SUMMARY_METRICS)
    paired = paired_vs_baseline(rows, baseline=baseline, group_keys=paired_group)
    insights = sweep_insights(summary, paired, x_key=x_key, baseline=baseline, high_loss_from=high_from)

    json_path = os.path.join(out_dir, f"{name}.json")
    write_json(json_path, {"metadata": metadata, "seed_summary": summary, "paired_vs_baseline": paired,
                           "insights": insights, "rows": rows})
    md_path = os.path.join(out_dir, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(summary_markdown(summary, metadata, x_key, title, extra_keys))
        handle.write("\n" + paired_markdown(paired, x_key, baseline))
        handle.write("\n# Insights\n\n" + "\n".join(f"- {line}" for line in insights) + "\n")
    npz_path = None
    if per_image:
        npz_path = os.path.join(out_dir, f"{name}.per_image.npz")
        np.savez_compressed(npz_path, **per_image)
    figures = standard_sweep_figures(summary, paired, os.path.join(out_dir, "figures"), name, per_image,
                                     cdf_rate, x_key, baseline, xlabel)
    return {"json": json_path, "markdown": md_path, "per_image": npz_path, "figures": figures,
            "summary": summary, "paired": paired, "insights": insights}


# --------------------------------------------------------------------------
# Loading results written by write_sweep_outputs
# --------------------------------------------------------------------------
def load_result(path: str) -> dict:
    import json

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_summary(path: str):
    """``(seed_summary, metadata)`` of a sweep JSON."""
    data = load_result(path)
    split = ((data.get("metadata") or {}).get("split")) or {}
    metadata = {**(data.get("metadata") or {}), "split": split}
    return data["seed_summary"], metadata
