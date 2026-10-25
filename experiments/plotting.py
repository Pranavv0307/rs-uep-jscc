"""Publication-style figures shared by every experiment script.

* colour-blind-safe palette (Okabe-Ito) and distinct markers / line styles, so
  curves stay distinguishable in black-and-white print (roadmap section 11);
* every figure is saved as **both** a vector PDF and a PNG;
* one stable style per method name, so the same method looks the same in every
  figure of the paper.
"""
import os
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]
LINESTYLES = ["-", "--", "-.", ":"]

IEEE_COLUMN_IN = 3.5
IEEE_DOUBLE_IN = 7.16

_STYLE_CACHE: Dict[str, dict] = {}


def apply_style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.25,
        "figure.dpi": 120, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


apply_style()


def style_for(name: str) -> dict:
    """Stable colour / marker / linestyle for a method name within one process."""
    if name not in _STYLE_CACHE:
        index = len(_STYLE_CACHE)
        _STYLE_CACHE[name] = {
            "color": OKABE_ITO[index % len(OKABE_ITO)],
            "marker": MARKERS[index % len(MARKERS)],
            "linestyle": LINESTYLES[(index // len(OKABE_ITO)) % len(LINESTYLES)],
        }
    return _STYLE_CACHE[name]


def save_figure(fig, base_path: str, dpi: int = 300) -> List[str]:
    """Write ``<base>.png`` and ``<base>.pdf``; ``base_path`` may carry either extension."""
    # Only strip a real image extension: names like "burst_p0.35_curves" contain dots.
    base = base_path[:-4] if base_path.lower().endswith((".png", ".pdf")) else base_path
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    paths = []
    for extension in ("png", "pdf"):
        path = f"{base}.{extension}"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    plt.close(fig)
    return paths


def _summary_series(summary: Sequence[dict], method: str, x_key: str, metric: str):
    rows = sorted((r for r in summary if r["method"] == method), key=lambda r: r[x_key])
    x = [r[x_key] for r in rows]
    y = [r[f"{metric}__mean"] for r in rows]
    err = [r.get(f"{metric}__std_over_seeds") or 0.0 for r in rows]
    return x, y, err


def plot_metric_curves(
    summary: Sequence[dict],
    methods: Iterable[str],
    out_path: str,
    x_key: str = "erasure_rate",
    metric: str = "psnr_db_mean_per_image",
    xlabel: str = "Packet erasure rate",
    ylabel: str = "PSNR (dB)",
    title: str = "",
    labels: Optional[Dict[str, str]] = None,
    figsize=(IEEE_DOUBLE_IN * 0.8, 3.6),
    legend_cols: int = 2,
) -> List[str]:
    """Mean over channel seeds with +/- std error bars, one curve per method."""
    fig, axis = plt.subplots(figsize=figsize)
    for method in methods:
        x, y, err = _summary_series(summary, method, x_key, metric)
        if not x:
            continue
        axis.errorbar(x, y, yerr=err, capsize=2.5, linewidth=1.3, markersize=4.5,
                      label=(labels or {}).get(method, method), **style_for(method))
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.legend(ncol=legend_cols, framealpha=0.9)
    return save_figure(fig, out_path)


def plot_paired_gain(
    paired: Sequence[dict],
    methods: Iterable[str],
    out_path: str,
    x_key: str = "erasure_rate",
    xlabel: str = "Packet erasure rate",
    ylabel: str = "PSNR gain over uniform RS (dB)",
    title: str = "",
    labels: Optional[Dict[str, str]] = None,
    figsize=(IEEE_DOUBLE_IN * 0.8, 3.6),
) -> List[str]:
    """Paired per-seed difference to the baseline with a 95% interval band."""
    fig, axis = plt.subplots(figsize=figsize)
    axis.axhline(0.0, color="black", linewidth=0.8)
    for method in methods:
        rows = sorted((r for r in paired if r["method"] == method), key=lambda r: r[x_key])
        if not rows:
            continue
        x = np.array([r[x_key] for r in rows], dtype=float)
        mean = np.array([r["mean"] for r in rows], dtype=float)
        half = np.array([r["ci95_half_width"] or 0.0 for r in rows], dtype=float)
        style = style_for(method)
        axis.plot(x, mean, linewidth=1.3, markersize=4.5, label=(labels or {}).get(method, method), **style)
        axis.fill_between(x, mean - half, mean + half, color=style["color"], alpha=0.15, linewidth=0)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.legend(ncol=2, framealpha=0.9)
    return save_figure(fig, out_path)


def plot_heatmap(matrix: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str], out_path: str,
                 title: str = "", cbar_label: str = "", fmt: str = "{:.1f}", cmap: str = "viridis",
                 figsize=None, center_zero: bool = False) -> List[str]:
    matrix = np.asarray(matrix, dtype=float)
    figsize = figsize or (max(4.0, 0.7 * len(col_labels) + 2), max(2.5, 0.35 * len(row_labels) + 1.5))
    fig, axis = plt.subplots(figsize=figsize)
    kwargs = {}
    if center_zero:
        limit = np.nanmax(np.abs(matrix)) or 1.0
        kwargs.update(vmin=-limit, vmax=limit)
        cmap = "RdBu"
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, **kwargs)
    axis.set_xticks(range(len(col_labels)), [str(c) for c in col_labels])
    axis.set_yticks(range(len(row_labels)), [str(r) for r in row_labels])
    axis.grid(False)
    for (i, j), value in np.ndenumerate(matrix):
        if np.isfinite(value):
            axis.text(j, i, fmt.format(value), ha="center", va="center", fontsize=6.5, color="white"
                      if not center_zero else "black")
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label(cbar_label)
    axis.set_title(title)
    return save_figure(fig, out_path)


def plot_bars(values: Sequence[float], labels: Sequence[str], out_path: str, ylabel: str = "", title: str = "",
              errors: Optional[Sequence[float]] = None, highlight: Sequence[int] = (), figsize=None) -> List[str]:
    figsize = figsize or (max(4.0, 0.35 * len(values) + 2), 3.2)
    fig, axis = plt.subplots(figsize=figsize)
    colors = [OKABE_ITO[1] if i in set(highlight) else OKABE_ITO[0] for i in range(len(values))]
    axis.bar(range(len(values)), values, yerr=errors, color=colors, capsize=2)
    axis.set_xticks(range(len(values)), [str(l) for l in labels], rotation=0)
    axis.set(ylabel=ylabel, title=title)
    return save_figure(fig, out_path)


def plot_scatter(x: Sequence[float], y: Sequence[float], out_path: str, xlabel: str = "", ylabel: str = "",
                 title: str = "", annotate: Optional[Sequence[str]] = None, fit_line: bool = True,
                 alpha: float = 0.3, size: float = 8, diagonal: bool = False, figsize=(3.6, 3.3)) -> List[str]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fig, axis = plt.subplots(figsize=figsize)
    axis.scatter(x, y, s=size, alpha=alpha, color=OKABE_ITO[0], edgecolors="none")
    if annotate is not None:
        for xi, yi, text in zip(x, y, annotate):
            axis.annotate(str(text), (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    if fit_line and x.size > 1 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 50)
        axis.plot(grid, slope * grid + intercept, color=OKABE_ITO[1], linestyle="--", linewidth=1.3,
                  label="linear fit")
        axis.legend()
    if diagonal:
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        axis.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, linestyle=":")
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return save_figure(fig, out_path)


def plot_lines(series: Dict[str, Sequence[float]], x: Sequence[float], out_path: str, xlabel: str = "",
               ylabel: str = "", title: str = "", log_y: bool = False, figsize=(3.8, 3.0)) -> List[str]:
    fig, axis = plt.subplots(figsize=figsize)
    for name, values in series.items():
        axis.plot(x[:len(values)], values, label=name, linewidth=1.3, **style_for(name))
    if log_y:
        axis.set_yscale("log")
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.legend()
    return save_figure(fig, out_path)
