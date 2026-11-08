"""Perceptual check of the speech sweep (S1): PESQ / STOI on a subset, plus listening examples.

The full erasure sweep (``eval_rs_uep --dataset librispeech``) reports per-segment SNR
and SI-SDR for every method, 5 channel seeds and all test segments. PESQ and STOI are far
too slow for that grid, so this script recomputes a small, fixed subset with them
(AD-17): a few methods, a few loss rates, one channel seed, the first ``--max_items``
segments. The packet losses are generated exactly as in ``eval_rs_uep.run_cell``
(same master-mask and random-order seeding per batch), so for the same seed, rate,
batch size and segments it reproduces the sweep's realisations.

* PESQ: wide-band ITU-T P.862.2 via the ``pesq`` package (MOS-LQO, 1.04-4.64);
  segments PESQ rejects (no speech detected) are counted and skipped.
* STOI: Taal et al. 2011 via ``pystoi`` (0-1).
* Either package missing -> that metric is reported as unavailable, not faked.

    python -m experiments.eval_speech_perceptual --checkpoint results/speech/deepjsccq1d/seed42/best_val.pth \\
        --oracle_ranking_file results/speech/deepjsccq1d/seed42/oracle_ranking.json --split test \\
        --selection results/speech/deepjsccq1d/seed42/val_sweep/selected_schemes.json \\
        --sources oracle,energy --output_dir results/speech/deepjsccq1d/seed42/perceptual
"""
import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from coding.erasure_channel import sample_master_mask
from coding.interleave import packet_mask_from_timeline
from coding.scaling import scheme_catalog
from data.datasets import is_official_test, load_eval_tensor
from data.speech import SAMPLE_RATE
from experiments.metrics import quality_db_per_item, si_sdr_db_per_item
from experiments.model_io import encode_full, load_model_bundle
from experiments.pipeline_core import MethodSpec, canonical_source, display_source, method_order, transmit
from experiments.importance_sources import compute_orders
from experiments.reproducibility import build_provenance, log_test_access, seed_everything, write_json

DEFAULT_RATES = "0.0,0.3,0.35,0.4,0.5"


def perceptual_scores(reference: np.ndarray, degraded: np.ndarray) -> Dict[str, Optional[float]]:
    """PESQ-WB and STOI of one segment; ``None`` where a package is missing or PESQ rejects the segment."""
    scores: Dict[str, Optional[float]] = {"pesq_wb": None, "stoi": None}
    try:
        from pesq import pesq
        try:
            scores["pesq_wb"] = float(pesq(SAMPLE_RATE, reference, degraded, "wb"))
        except Exception:  # noqa: BLE001 - pesq raises NoUtterancesError etc. on unusable segments
            scores["pesq_wb"] = float("nan")
    except ImportError:
        pass
    try:
        from pystoi import stoi
        scores["stoi"] = float(stoi(reference, degraded, SAMPLE_RATE, extended=False))
    except ImportError:
        pass
    return scores


def method_specs(selection_path: Optional[str], sources: List[str], schemes: List[str]) -> List[MethodSpec]:
    """``no_rs``, ``uniform`` and, per source, its validation-selected high-loss scheme (or ``--schemes``)."""
    specs = [MethodSpec("no_rs", "no_rs"), MethodSpec("uniform", "uniform")]
    chosen = {}
    if selection_path:
        with open(selection_path, encoding="utf-8") as handle:
            selection = json.load(handle)
        if selection.get("selected_on") == "test":
            raise SystemExit("the selection file was made on the test split; refusing (roadmap Rule 1)")
        chosen = {src: entry["best_overall_high_loss"] for src, entry in selection["selection"].items()}
    for token in sources:
        source = canonical_source(token)
        names = schemes or ([chosen[display_source(source)]] if display_source(source) in chosen else [])
        if not names:
            raise SystemExit(f"no scheme for source {token!r}: pass --selection or --schemes")
        for scheme in names:
            specs.append(MethodSpec(f"{scheme}@{display_source(source)}", scheme, source))
    return specs


def write_wav(path: str, waveform: torch.Tensor) -> None:
    import soundfile

    os.makedirs(os.path.dirname(path), exist_ok=True)
    soundfile.write(path, waveform.detach().cpu().numpy().reshape(-1).clip(-1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--oracle_ranking_file", required=True, help="train-split ranking for source 'oracle'")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--max_items", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=250, help="keep equal to the sweep's for identical masks")
    parser.add_argument("--selection", default=None, help="selected_schemes.json from the VALIDATION sweep")
    parser.add_argument("--schemes", default="", help="explicit schemes instead of --selection")
    parser.add_argument("--sources", default="oracle,energy")
    parser.add_argument("--erasure_rates", default=DEFAULT_RATES)
    parser.add_argument("--seed", type=int, default=0, help="channel seed")
    parser.add_argument("--packet_size", type=int, default=16)
    parser.add_argument("--overhead", type=float, default=0.5)
    parser.add_argument("--save_examples", type=int, default=3, help="segments written as WAV per method and rate")
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    seed_everything(args.global_seed)
    bundle = load_model_bundle(args.checkpoint, torch.device("cpu"))
    if bundle.modality != "speech":
        raise SystemExit(f"{args.checkpoint} is an image model ({bundle.arch})")
    if is_official_test("librispeech", args.split):
        log_test_access("experiments.eval_speech_perceptual", args.checkpoint, note=f"{args.max_items} segments")
    items = load_eval_tensor("librispeech", args.split, args.data_dir, args.max_items)
    with open(args.oracle_ranking_file, encoding="utf-8") as handle:
        ranking = json.load(handle)["ranking_channels_most_to_least_important"]
    specs = method_specs(args.selection, [s.strip() for s in args.sources.split(",") if s.strip()],
                         [s.strip() for s in args.schemes.split(",") if s.strip()])
    catalog = scheme_catalog(bundle.channels, bundle.symbols_per_channel, args.packet_size, args.overhead,
                             names=sorted({s.scheme for s in specs}))
    master_n = max(scheme.total_packets for scheme in catalog.values())
    rates = [float(r) for r in args.erasure_rates.split(",") if r.strip()]
    sources = sorted({s.source for s in specs if s.source not in (None, "random")})

    # Indices and orders do not depend on the loss: compute once.
    idx_parts, order_parts = [], {s: [] for s in sources}
    with torch.no_grad():
        for start in range(0, len(items), args.batch_size):
            x = items[start:start + args.batch_size]
            info = encode_full(bundle.encoder, x, bundle.snr_db)
            idx_parts.append(info["idx"].long())
            for source, order in compute_orders(bundle, x, sources, bundle.snr_db, {"oracle": ranking},
                                                info=info).items():
                order_parts[source].append(order)
    idx = torch.cat(idx_parts)
    orders = {s: torch.cat(parts) for s, parts in order_parts.items()}

    rows, wav_dir = [], os.path.join(args.output_dir, "examples")
    for i in range(min(args.save_examples, len(items))):
        write_wav(os.path.join(wav_dir, f"item{i:02d}_original.wav"), items[i])
    for rate in rates:
        per_method = {spec.name: {"snr": [], "si_sdr": [], "pesq_wb": [], "stoi": []} for spec in specs}
        for batch_index, start in enumerate(range(0, len(items), args.batch_size)):
            stop = min(start + args.batch_size, len(items))
            x = items[start:stop]
            master = sample_master_mask(stop - start, rate, torch.Generator().manual_seed(args.seed * 100000 + batch_index),
                                        num_packets=master_n)
            for spec in specs:
                scheme = catalog[spec.scheme]
                order_generator = torch.Generator().manual_seed(args.seed * 100000 + batch_index + 50000)
                order = method_order(spec, orders, start, stop, bundle.channels, order_generator, bundle.device)
                mask = packet_mask_from_timeline(master, scheme, spec.interleave)
                with torch.no_grad():
                    recon, *_ = transmit(bundle, idx[start:stop], scheme, order, mask, bundle.snr_db)
                bucket = per_method[spec.name]
                bucket["snr"] += quality_db_per_item(recon, x).tolist()
                bucket["si_sdr"] += si_sdr_db_per_item(recon, x).tolist()
                for j in range(stop - start):
                    scores = perceptual_scores(x[j, 0].numpy().astype(np.float64), recon[j, 0].numpy().astype(np.float64))
                    bucket["pesq_wb"].append(scores["pesq_wb"])
                    bucket["stoi"].append(scores["stoi"])
                    if start + j < args.save_examples:
                        write_wav(os.path.join(wav_dir, f"item{start + j:02d}_{spec.name.replace('@', '_at_')}"
                                                        f"_loss{rate:g}.wav"), recon[j])
        for name, bucket in per_method.items():
            def mean(values):
                values = [v for v in values if v is not None and np.isfinite(v)]
                return float(np.mean(values)) if values else None
            rows.append({"method": name, "erasure_rate": rate, "items": len(bucket["snr"]),
                         "snr_db_mean": mean(bucket["snr"]), "si_sdr_db_mean": mean(bucket["si_sdr"]),
                         "pesq_wb_mean": mean(bucket["pesq_wb"]), "stoi_mean": mean(bucket["stoi"]),
                         "pesq_rejected": int(sum(1 for v in bucket["pesq_wb"] if v is not None and not np.isfinite(v)))})
        print(f"loss {rate:g}: " + ", ".join(f"{r['method']} PESQ {r['pesq_wb_mean']}" for r in rows[-len(specs):]))

    os.makedirs(args.output_dir, exist_ok=True)
    write_json(os.path.join(args.output_dir, "perceptual.json"), {
        "metadata": build_provenance(args=args, checkpoint_path=args.checkpoint, dataset="LibriSpeech",
                                     extra_files={"oracle_ranking": args.oracle_ranking_file,
                                                  **({"selection": args.selection} if args.selection else {})},
                                     split_info={"dataset": "librispeech", "split": args.split},
                                     global_seed=args.global_seed),
        "rows": rows})
    lines = ["# Speech perceptual check (PESQ-WB / STOI)", "",
             f"{len(items)} {args.split} segments, channel seed {args.seed}. Unavailable metrics are shown as '-'.", "",
             "| method | loss | SNR (dB) | SI-SDR (dB) | PESQ-WB | STOI | PESQ rejected |", "|---|---|---|---|---|---|---|"]
    fmt = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    for r in rows:
        lines.append(f"| {r['method']} | {r['erasure_rate']:g} | {fmt(r['snr_db_mean'])} | {fmt(r['si_sdr_db_mean'])} | "
                     f"{fmt(r['pesq_wb_mean'])} | {fmt(r['stoi_mean'], 3)} | {r['pesq_rejected']} |")
    with open(os.path.join(args.output_dir, "perceptual.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"saved {args.output_dir}/perceptual.json, perceptual.md and {wav_dir}/")


if __name__ == "__main__":
    main()
