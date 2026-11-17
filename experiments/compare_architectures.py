"""G1: is sensitivity-aware UEP an ADJSCC effect or a Deep JSCC effect? Compare model families.

Each model is given by name ``<family>/<model>`` (e.g. ``adjsccq/seed42``,
``deepjsccq/seed42``) with

* ``--model NAME=FOLDER``  its training folder (``oracle_ranking.json``; ``metrics.json``
  optional, for clean PSNR and convergence), and
* ``--sweep NAME=JSON``    its packet-erasure sweep from ``experiments/eval_rs_uep.py``
  (optional; without it only the damage profile is compared).

Per model it reports the clean test PSNR, convergence (best validation epoch >= 50,
the rule of ``results/analysis/convergence_ledger.json``), the sorted leave-one-out
damage profile, the top-1 / top-4 share of total damage and, from the *paired* sweep,
the gain over uniform RS of the best tiered scheme of every importance source at every
loss rate and averaged over loss >= 0.35 -- the same "best scheme among the
validation-selected ones" rule that produced INSIGHTS.md. Per family: mean, sample std
and 95% t-interval over the converged models. Then the pre-registered decision rule
(``docs/02_research_plan_and_design_decisions/ARCHITECTURE_DECISION_RECORD_AD01_TO_AD21.md``, AD-09) is applied to each family:

* C1 concentration:   family mean top-4 share >= ``--min_top4_share``         (0.40; even split = 4/C)
* C2 UEP gain:        every model's oracle gain at ``--decision_rate`` >= ``--min_gain_db``
                      with the 95% interval above zero                           (0.40 loss, 1.5 dB)
* C3 identity:        every model's random-order gain < ``--max_random_fraction`` of its
                      oracle gain (mean over loss >= 0.35)                       (0.5)
* FLAT (no structure): family mean top-4 share < ``--flat_top4_share`` and oracle gain within
                      ``--equivalence_margin_db`` of random                      (0.30, 0.5 dB)

Verdict: C1-C3 all hold -> ``criteria met`` (UEP works on this family as on ADJSCC-Q);
FLAT holds -> ``no exploitable structure``; otherwise ``intermediate``. The thresholds
are command-line arguments only so that they are visible in the provenance; they
must not be changed after the test sweeps have been seen.

    python -m experiments.compare_architectures \\
        --model adjsccq/seed42=published_results/gpu/models/seed42 \\
        --sweep adjsccq/seed42=published_results/gpu/final/final_test.json \\
        --model deepjsccq/seed42=results/g1/deepjsccq/seed42 \\
        --sweep deepjsccq/seed42=results/g1/deepjsccq/seed42/test_sweep/test_sweep.json \\
        --output_dir results/g1/compare
"""
import argparse
import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from experiments import plotting
from experiments.reproducibility import build_provenance, seed_statistics, write_json

CONVERGED_MIN_BEST_EPOCH = 50   # results/analysis/convergence_ledger.json, "FINAL RULE"
KEY_SOURCES = ("oracle", "energy", "l1", "predicted", "random", "attention")


def rate_key(rate: float) -> str:
    return f"{float(rate):g}"


def parse_named(specs: Optional[List[str]], flag: str) -> "OrderedDict[str, str]":
    out: "OrderedDict[str, str]" = OrderedDict()
    for spec in specs or []:
        name, sep, path = spec.partition("=")
        if not sep or "/" not in name or not path:
            raise SystemExit(f"{flag} expects FAMILY/NAME=PATH, got {spec!r}")
        if name in out:
            raise SystemExit(f"{flag} {name} given twice")
        out[name] = path
    return out


# --------------------------------------------------------------------------
# Per-model inputs
# --------------------------------------------------------------------------
def load_model_folder(folder: str) -> dict:
    """Damage profile (+ clean PSNR and convergence when ``metrics.json`` exists)."""
    with open(os.path.join(folder, "oracle_ranking.json"), encoding="utf-8") as handle:
        oracle = json.load(handle)
    damage = np.asarray(oracle["psnr_drop_db_by_channel"], dtype=np.float64)
    profile = -np.sort(-damage)
    total = float(profile.sum())
    entry = {
        "folder": folder,
        "channels": int(damage.size),
        "sorted_damage_db": profile.tolist(),
        "worst_channel_damage_db": float(profile[0]),
        "least_channel_damage_db": float(profile[-1]),
        "top1_share": float(profile[0] / total) if total > 0 else None,
        "top4_share": float(profile[:4].sum() / total) if total > 0 else None,
        "even_split_top4_share": 4.0 / damage.size,
        "ranking": oracle.get("ranking_channels_most_to_least_important"),
        "clean_test_psnr_db": None, "clean_val_psnr_db": None, "best_epoch": None, "converged": None,
        "arch": None, "train_device": None,
    }
    metrics_path = os.path.join(folder, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        entry["clean_test_psnr_db"] = (metrics.get("clean_test") or {}).get("psnr_db_mean_per_image")
        entry["clean_val_psnr_db"] = (metrics.get("clean_validation") or {}).get("psnr_db_mean_per_image")
        entry["best_epoch"] = metrics.get("best_epoch")
        if entry["best_epoch"] is not None:
            entry["converged"] = int(entry["best_epoch"]) >= CONVERGED_MIN_BEST_EPOCH
        entry["arch"] = metrics.get("arch", "adjsccq")      # pre-G0 metrics have no arch key
        entry["train_device"] = metrics.get("train_device")
    return entry


def sweep_gains(path: str, high_loss_from: float = 0.35) -> dict:
    """Best tiered gain over uniform per importance source and loss rate (sequential layout only)."""
    with open(path, encoding="utf-8") as handle:
        sweep = json.load(handle)
    best: Dict[str, Dict[str, dict]] = {}
    for row in sweep.get("paired_vs_baseline", []):
        method = row.get("method", "")
        if row.get("baseline") != "uniform" or "@" not in method or "#" in method:
            continue
        scheme, source = method.split("@", 1)
        key = rate_key(row["erasure_rate"])
        current = best.setdefault(source, {}).get(key)
        if current is None or row["mean"] > current["gain_db"]:
            half = row.get("ci95_half_width") or 0.0
            best[source][key] = {"gain_db": float(row["mean"]), "ci95_half_width": float(half),
                                 "ci95_low": float(row["mean"]) - float(half), "scheme": scheme}
    uniform = {rate_key(r["erasure_rate"]): r["psnr_db_mean_per_image__mean"]
               for r in sweep.get("seed_summary", []) if r.get("method") == "uniform"}
    high_loss = {}
    for source, per_rate in best.items():
        values = [v["gain_db"] for k, v in per_rate.items() if float(k) >= high_loss_from - 1e-9]
        high_loss[source] = float(np.mean(values)) if values else None
    return {"path": path, "best_gain": best, "high_loss_mean_gain_db": high_loss, "uniform_psnr_db": uniform,
            "sources": sorted(best)}


# --------------------------------------------------------------------------
# Families and the decision rule
# --------------------------------------------------------------------------
def _stats(values: List[float]) -> Optional[dict]:
    values = [float(v) for v in values if v is not None]
    return seed_statistics(values) if values else None


def family_summary(models: Dict[str, dict], decision_rate: float) -> dict:
    """Aggregate the converged models of one family (unknown convergence counts as converged, flagged)."""
    key = rate_key(decision_rate)
    used = {name: m for name, m in models.items() if m["converged"] is not False}
    excluded = sorted(set(models) - set(used))

    def gain_at(m, source):
        return ((m.get("sweep") or {}).get("best_gain", {}).get(source, {}).get(key) or {}).get("gain_db")

    def high_loss(m, source):
        return (m.get("sweep") or {}).get("high_loss_mean_gain_db", {}).get(source)

    summary = {
        "models_used": sorted(used), "models_excluded_not_converged": excluded,
        "convergence_unknown": sorted(n for n, m in used.items() if m["converged"] is None),
        "top4_share": _stats([m["top4_share"] for m in used.values()]),
        "top1_share": _stats([m["top1_share"] for m in used.values()]),
        "worst_channel_damage_db": _stats([m["worst_channel_damage_db"] for m in used.values()]),
        "clean_test_psnr_db": _stats([m["clean_test_psnr_db"] for m in used.values()]),
        "gain_at_decision_rate_db": {}, "high_loss_mean_gain_db": {}, "share_of_oracle_gain": {},
    }
    for source in KEY_SOURCES:
        summary["gain_at_decision_rate_db"][source] = _stats([gain_at(m, source) for m in used.values()])
        summary["high_loss_mean_gain_db"][source] = _stats([high_loss(m, source) for m in used.values()])
        shares = []
        for m in used.values():
            own, ref = high_loss(m, source), high_loss(m, "oracle")
            if own is not None and ref:
                shares.append(own / ref)
        summary["share_of_oracle_gain"][source] = _stats(shares)
    return summary


def decide(models: Dict[str, dict], summary: dict, criteria: dict) -> dict:
    key = rate_key(criteria["decision_rate"])
    used = {n: models[n] for n in summary["models_used"]}
    checks, notes = {}, []

    top4 = (summary["top4_share"] or {}).get("mean")
    checks["C1_concentration"] = None if top4 is None else bool(top4 >= criteria["min_top4_share"])

    c2 = []
    for name, m in used.items():
        cell = ((m.get("sweep") or {}).get("best_gain", {}).get("oracle", {}) or {}).get(key)
        if cell is None:
            notes.append(f"{name}: no oracle gain at loss {key} (no sweep or rate missing)")
            continue
        c2.append(cell["gain_db"] >= criteria["min_gain_db"] and cell["ci95_low"] > 0)
    checks["C2_uep_gain"] = bool(all(c2)) if c2 and len(c2) == len(used) else None

    c3, flat_margin = [], []
    for name, m in used.items():
        hl = (m.get("sweep") or {}).get("high_loss_mean_gain_db", {})
        oracle, random = hl.get("oracle"), hl.get("random")
        if oracle is None or random is None:
            notes.append(f"{name}: random-order control missing; C3 / FLAT cannot use this model")
            continue
        c3.append(oracle > 0 and random / oracle < criteria["max_random_fraction"])
        flat_margin.append(oracle - random <= criteria["equivalence_margin_db"])
    checks["C3_identity_matters"] = bool(all(c3)) if c3 else None

    flat = None
    if top4 is not None and flat_margin:
        flat = bool(top4 < criteria["flat_top4_share"] and all(flat_margin))
    checks["FLAT_no_structure"] = flat

    core = [checks["C1_concentration"], checks["C2_uep_gain"], checks["C3_identity_matters"]]
    if all(value is True for value in core):
        verdict = "criteria met"
    elif flat is True:
        verdict = "no exploitable structure"
    elif any(value is None for value in core):
        verdict = "incomplete"
    else:
        verdict = "intermediate"
    return {"checks": checks, "verdict": verdict, "notes": notes}


def deltas(families: Dict[str, dict], reference: str) -> Dict[str, dict]:
    """Family mean minus reference-family mean for the headline quantities."""
    if reference not in families:
        return {}
    ref = families[reference]["summary"]

    def mean(block):
        return None if block is None else block.get("mean")

    out = {}
    for name, family in families.items():
        if name == reference:
            continue
        s = family["summary"]
        row = {
            "top4_share": (mean(s["top4_share"]), mean(ref["top4_share"])),
            "worst_channel_damage_db": (mean(s["worst_channel_damage_db"]), mean(ref["worst_channel_damage_db"])),
            "clean_test_psnr_db": (mean(s["clean_test_psnr_db"]), mean(ref["clean_test_psnr_db"])),
        }
        for source in ("oracle", "energy", "random"):
            row[f"gain_at_decision_rate_db@{source}"] = (mean(s["gain_at_decision_rate_db"][source]),
                                                         mean(ref["gain_at_decision_rate_db"][source]))
            row[f"high_loss_mean_gain_db@{source}"] = (mean(s["high_loss_mean_gain_db"][source]),
                                                       mean(ref["high_loss_mean_gain_db"][source]))
        out[f"{name} - {reference}"] = {
            k: {"value": a, "reference": b, "difference": None if a is None or b is None else a - b}
            for k, (a, b) in row.items()}
    return out


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def _fmt(block, digits=2, pct=False):
    if block is None:
        return "-"
    if isinstance(block, dict):
        mean, std = block.get("mean"), block.get("std")
        scale = 100.0 if pct else 1.0
        text = f"{scale * mean:.{digits}f}"
        if std is not None:
            text += f" ± {scale * std:.{digits}f}"
        return text + ("%" if pct else "")
    return f"{100 * block:.0f}%" if pct else f"{block:+.{digits}f}"


def write_markdown(path: str, result: dict) -> None:
    criteria = result["criteria"]
    key = rate_key(criteria["decision_rate"])
    lines = ["# Architecture comparison (G1): is sensitivity-aware UEP specific to ADJSCC-Q?", "",
             f"Decision rate: loss {key}. Gains are paired differences to uniform RS of the best "
             "(validation-selected) tiered scheme per source; families use converged models only.", "",
             "## Per model", "",
             "| model | arch | clean test PSNR | best epoch | converged | worst ch. (dB) | top-4 share | "
             f"oracle gain @{key} (CI) | random gain @{key} | energy gain @{key} | oracle mean ≥0.35 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in result["models"].items():
        sweep = m.get("sweep") or {}
        gains = sweep.get("best_gain", {})

        def cell(source):
            c = gains.get(source, {}).get(key)
            return "-" if c is None else f"{c['gain_db']:+.2f}"

        oracle = gains.get("oracle", {}).get(key)
        oracle_text = "-" if oracle is None else f"{oracle['gain_db']:+.2f} (±{oracle['ci95_half_width']:.2f})"
        high_loss = sweep.get("high_loss_mean_gain_db", {}).get("oracle")
        clean = "-" if m["clean_test_psnr_db"] is None else f"{m['clean_test_psnr_db']:.2f}"
        epoch = "-" if m["best_epoch"] is None else str(m["best_epoch"])
        converged = {True: "yes", False: "**no**", None: "unknown"}[m["converged"]]
        top4 = "-" if m["top4_share"] is None else f"{100 * m['top4_share']:.0f}%"
        high_loss_text = "-" if high_loss is None else f"{high_loss:+.2f}"
        cells = [name, m.get("arch") or "-", clean, epoch, converged, f"{m['worst_channel_damage_db']:.2f}", top4,
                 oracle_text, cell("random"), cell("energy"), high_loss_text]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Per family (mean ± sample std over converged models)", "",
              "| family | models | top-4 share | worst ch. (dB) | clean test PSNR | "
              f"oracle @{key} | energy @{key} | random @{key} | energy share of oracle | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, family in result["families"].items():
        s = family["summary"]
        lines.append(
            f"| {name} | {len(s['models_used'])} | {_fmt(s['top4_share'], 0, pct=True)} | "
            f"{_fmt(s['worst_channel_damage_db'])} | {_fmt(s['clean_test_psnr_db'])} | "
            f"{_fmt(s['gain_at_decision_rate_db']['oracle'])} | {_fmt(s['gain_at_decision_rate_db']['energy'])} | "
            f"{_fmt(s['gain_at_decision_rate_db']['random'])} | {_fmt(s['share_of_oracle_gain']['energy'], 0, pct=True)} | "
            f"**{family['decision']['verdict']}** |")
    lines += ["", "## Decision rule (pre-registered, AD-09)", "",
              f"- C1 concentration: family mean top-4 share ≥ {criteria['min_top4_share']:.2f}",
              f"- C2 UEP gain: every model's oracle gain at loss {key} ≥ {criteria['min_gain_db']:+.2f} dB, CI above 0",
              f"- C3 identity matters: random gain < {criteria['max_random_fraction']:.2f} × oracle gain (loss ≥ 0.35)",
              f"- FLAT: top-4 share < {criteria['flat_top4_share']:.2f} and oracle − random ≤ "
              f"{criteria['equivalence_margin_db']:.2f} dB", ""]
    for name, family in result["families"].items():
        checks = ", ".join(f"{k}={v}" for k, v in family["decision"]["checks"].items())
        lines.append(f"- **{name}**: {family['decision']['verdict']} ({checks})")
        lines += [f"  - note: {n}" for n in family["decision"]["notes"]]
    if result["deltas"]:
        lines += ["", "## Differences between family means", "", "| comparison | quantity | value | reference | difference |",
                  "|---|---|---|---|---|"]
        for comparison, rows in result["deltas"].items():
            for quantity, v in rows.items():
                if v["difference"] is None:
                    continue
                lines.append(f"| {comparison} | {quantity} | {v['value']:.3f} | {v['reference']:.3f} | "
                             f"{v['difference']:+.3f} |")
    lines += ["", "## Insights", ""] + [f"- {n}" for n in result["insights"]] + [""]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def insights(result: dict) -> List[str]:
    key = rate_key(result["criteria"]["decision_rate"])
    notes = []
    for name, family in result["families"].items():
        s = family["summary"]
        top4, oracle = s["top4_share"], s["gain_at_decision_rate_db"]["oracle"]
        text = f"{name}: {len(s['models_used'])} converged model(s)"
        if top4:
            text += f", top-4 channels carry {100 * top4['mean']:.0f}% of the damage (even split {100 * 4 / 16:.0f}% at 16 channels)"
        if oracle:
            text += f", oracle UEP gain at loss {key} {oracle['mean']:+.2f} dB"
        notes.append(text + f" -> {family['decision']['verdict']}.")
        if s["models_excluded_not_converged"]:
            notes.append(f"{name}: excluded as not converged (best epoch < {CONVERGED_MIN_BEST_EPOCH}): "
                         + ", ".join(s["models_excluded_not_converged"]) + ".")
    for comparison, rows in result["deltas"].items():
        share = rows.get("top4_share", {})
        gain = rows.get("gain_at_decision_rate_db@oracle", {})
        if share.get("difference") is not None:
            notes.append(f"{comparison}: top-4 share {100 * share['difference']:+.1f} points"
                         + ("" if gain.get("difference") is None else
                            f", oracle gain at loss {key} {gain['difference']:+.2f} dB") + ".")
    devices = {m.get("train_device") for m in result["models"].values()}
    if len(devices - {None}) > 1 or (None in devices and len(devices) > 1):
        notes.append("Models were trained on different hardware (" + ", ".join(sorted(str(d) for d in devices))
                     + "); see AD-07 for the same-hardware anchor.")
    return notes


def make_figures(result: dict, out_dir: str) -> List[str]:
    key = rate_key(result["criteria"]["decision_rate"])
    figures = []
    xs, ys, labels = [], [], []
    for name, m in result["models"].items():
        cell = ((m.get("sweep") or {}).get("best_gain", {}).get("oracle", {}) or {}).get(key)
        if cell is not None and m["top4_share"] is not None:
            xs.append(m["top4_share"])
            ys.append(cell["gain_db"])
            labels.append(name.replace("adjsccq/", "A:").replace("deepjsccq/", "D:"))
    if len(xs) >= 2:
        figures += plotting.plot_scatter(xs, ys, os.path.join(out_dir, "gain_vs_concentration"),
                                         xlabel="Top-4 share of LOO damage", ylabel=f"Oracle UEP gain at loss {key} (dB)",
                                         annotate=labels, fit_line=True, alpha=0.9, size=24)
    profiles = {name: m["sorted_damage_db"] for name, m in result["models"].items()}
    if profiles:
        width = max(len(v) for v in profiles.values())
        figures += plotting.plot_lines(profiles, list(range(1, width + 1)), os.path.join(out_dir, "sorted_profiles"),
                                       xlabel="Channel rank", ylabel="LOO damage (dB)", figsize=(4.2, 3.0))
    curves = {}
    for name, family in result["families"].items():
        rates = sorted({float(k) for n in family["summary"]["models_used"]
                        for k in ((result["models"][n].get("sweep") or {}).get("best_gain", {}).get("oracle", {}))})
        if not rates:
            continue
        means = []
        for rate in rates:
            values = [((result["models"][n].get("sweep") or {}).get("best_gain", {}).get("oracle", {})
                       .get(rate_key(rate)) or {}).get("gain_db") for n in family["summary"]["models_used"]]
            values = [v for v in values if v is not None]
            means.append(float(np.mean(values)) if values else np.nan)
        curves[name] = (rates, means)
    if curves:
        shared = sorted({r for rates, _ in curves.values() for r in rates})
        series = {name: [dict(zip(rates, means)).get(r, np.nan) for r in shared] for name, (rates, means) in curves.items()}
        figures += plotting.plot_lines(series, shared, os.path.join(out_dir, "oracle_gain_by_family"),
                                       xlabel="Packet loss rate", ylabel="Oracle UEP gain over uniform (dB)")
    return figures


def build_result(model_paths, sweep_paths, criteria, reference: str, high_loss_from: float = 0.35) -> dict:
    unknown = set(sweep_paths) - set(model_paths)
    if unknown:
        raise SystemExit(f"--sweep given for unknown model(s) {sorted(unknown)}; add --model for each")
    models = OrderedDict()
    for name, folder in model_paths.items():
        entry = load_model_folder(folder)
        entry["family"] = name.split("/", 1)[0]
        entry["sweep"] = sweep_gains(sweep_paths[name], high_loss_from) if name in sweep_paths else None
        models[name] = entry
    families = OrderedDict()
    for family in dict.fromkeys(m["family"] for m in models.values()):
        members = {n: m for n, m in models.items() if m["family"] == family}
        summary = family_summary(members, criteria["decision_rate"])
        families[family] = {"summary": summary, "decision": decide(members, summary, criteria)}
    result = {"criteria": criteria, "reference_family": reference, "models": models, "families": families,
              "deltas": deltas(families, reference)}
    result["insights"] = insights(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="FAMILY/NAME=training folder")
    parser.add_argument("--sweep", action="append", default=None, help="FAMILY/NAME=eval_rs_uep JSON")
    parser.add_argument("--reference_family", default="adjsccq")
    parser.add_argument("--decision_rate", type=float, default=0.4)
    parser.add_argument("--high_loss_from", type=float, default=0.35)
    parser.add_argument("--min_top4_share", type=float, default=0.40)
    parser.add_argument("--min_gain_db", type=float, default=1.5)
    parser.add_argument("--max_random_fraction", type=float, default=0.5)
    parser.add_argument("--flat_top4_share", type=float, default=0.30)
    parser.add_argument("--equivalence_margin_db", type=float, default=0.5)
    parser.add_argument("--output_dir", default="results/g1/compare")
    args = parser.parse_args()

    model_paths = parse_named(args.model, "--model")
    sweep_paths = parse_named(args.sweep, "--sweep")
    criteria = {k: getattr(args, k) for k in ("decision_rate", "min_top4_share", "min_gain_db", "max_random_fraction",
                                               "flat_top4_share", "equivalence_margin_db")}
    result = build_result(model_paths, sweep_paths, criteria, args.reference_family, args.high_loss_from)

    os.makedirs(args.output_dir, exist_ok=True)
    files = {f"model:{n}": os.path.join(p, "oracle_ranking.json") for n, p in model_paths.items()}
    files.update({f"sweep:{n}": p for n, p in sweep_paths.items()})
    result["figures"] = make_figures(result, args.output_dir)
    write_json(os.path.join(args.output_dir, "compare.json"),
               {"metadata": build_provenance(args=args, extra_files=files), **result})
    write_markdown(os.path.join(args.output_dir, "compare.md"), result)
    print("\n".join(f"- {n}" for n in result["insights"]))


if __name__ == "__main__":
    main()
