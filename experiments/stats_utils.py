"""Small, dependency-light statistics used by the analysis scripts.

Everything here works on numpy arrays and avoids scipy so the same code runs on
Kaggle / Colab / the local machine without version surprises.
"""
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from experiments.reproducibility import seed_statistics, t_critical_95


# --------------------------------------------------------------------------
# Ranks and correlations
# --------------------------------------------------------------------------
def rankdata(values: Sequence[float]) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return float((x * y).sum() / denominator) if denominator > 0 else float("nan")


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall tau-a over all pairs (fine for the 16-32 channels used here)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    pairs = x.size * (x.size - 1)
    return float((dx * dy).sum() / pairs) if pairs else float("nan")


def row_spearman(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Spearman correlation of every row of ``predicted`` with the same row of ``target``.

    Ranks are computed with a double argsort, so exact ties are broken by
    position; for continuous scores ties essentially never occur.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rp = np.argsort(np.argsort(predicted, axis=1), axis=1).astype(np.float64)
    rt = np.argsort(np.argsort(target, axis=1), axis=1).astype(np.float64)
    rp -= rp.mean(axis=1, keepdims=True)
    rt -= rt.mean(axis=1, keepdims=True)
    denominator = np.sqrt((rp ** 2).sum(axis=1) * (rt ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rp * rt).sum(axis=1) / denominator


def topk_overlap(predicted: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Per row: fraction of the true top-``k`` channels that the prediction also puts in its top ``k``."""
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    top_p = np.argsort(-predicted, axis=1, kind="stable")[:, :k]
    top_t = np.argsort(-target, axis=1, kind="stable")[:, :k]
    hits = [len(set(p.tolist()) & set(t.tolist())) for p, t in zip(top_p, top_t)]
    return np.asarray(hits, dtype=np.float64) / k


def top1_accuracy(predicted: np.ndarray, target: np.ndarray) -> float:
    return float((np.argmax(predicted, axis=1) == np.argmax(target, axis=1)).mean())


def r_squared(predicted: np.ndarray, target: np.ndarray) -> float:
    residual = ((target - predicted) ** 2).sum()
    total = ((target - target.mean()) ** 2).sum()
    return float(1.0 - residual / total) if total > 0 else float("nan")


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------
def bootstrap_ci(
    data: np.ndarray,
    statistic,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Percentile bootstrap over the first axis of ``data``: ``(estimate, low, high)``."""
    data = np.asarray(data)
    rng = np.random.default_rng(seed)
    estimate = float(statistic(data))
    resampled = np.empty(n_boot)
    for b in range(n_boot):
        resampled[b] = statistic(data[rng.integers(0, len(data), len(data))])
    low, high = np.nanpercentile(resampled, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return estimate, float(low), float(high)


def paired_difference(a: Sequence[float], b: Sequence[float]) -> Dict[str, Optional[float]]:
    """Paired comparison ``a - b`` (same seeds): mean, std, 95% t-interval, win-rate."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    diff = a - b
    stats = seed_statistics(diff.tolist())
    return {
        "mean": stats["mean"],
        "std": stats["std"],
        "ci95_half_width": stats["ci95_half_width"],
        "n": int(diff.size),
        "win_rate": float((diff > 0).mean()),
        "significant": bool(
            stats["ci95_half_width"] is not None and abs(stats["mean"]) > stats["ci95_half_width"]
        ),
    }


def paired_vs_baseline(
    rows: Iterable[Dict],
    baseline: str = "uniform",
    metric: str = "psnr_db_mean_per_image",
    group_keys: Sequence[str] = ("erasure_rate",),
) -> List[Dict]:
    """For every method and group, the per-seed paired difference to ``baseline``.

    Rows must carry ``method`` and ``seed``. Because every method sees the same
    master loss mask per seed, the per-seed difference removes the shared
    channel randomness, which is exactly why paired statistics are used.
    """
    table: Dict[tuple, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = tuple(row[g] for g in group_keys)
        table[key][row["method"]][row["seed"]] = row[metric]

    out = []
    for key, by_method in sorted(table.items()):
        if baseline not in by_method:
            continue
        base = by_method[baseline]
        for method, values in by_method.items():
            if method == baseline:
                continue
            seeds = sorted(set(values) & set(base))
            if not seeds:
                continue
            entry = dict(zip(group_keys, key))
            entry["method"] = method
            entry["baseline"] = baseline
            entry.update(paired_difference([values[s] for s in seeds], [base[s] for s in seeds]))
            out.append(entry)
    return out


def crossover(rates: Sequence[float], differences: Sequence[float]) -> Optional[float]:
    """Smallest erasure rate at which ``difference`` turns positive (linear interpolation).

    Returns ``None`` if it never becomes positive; the first rate if it is
    positive everywhere.
    """
    rates = list(rates)
    differences = list(differences)
    if not rates:
        return None
    if differences[0] > 0:
        return rates[0]
    for i in range(1, len(rates)):
        if differences[i] > 0:
            r0, r1, d0, d1 = rates[i - 1], rates[i], differences[i - 1], differences[i]
            return r0 + (0.0 - d0) * (r1 - r0) / (d1 - d0)
    return None
