"""Readable check of the text sweep (T1): word accuracy, bits/char and side-by-side example passages.

The text counterpart of ``eval_speech_perceptual`` (AD-27). The full erasure sweep
(``eval_rs_uep --dataset text8``) already reports character and word accuracy for every method,
5 channel seeds and all test windows. This script replays a small, fixed subset: a few methods,
a few loss rates, one channel seed, the first ``--max_items`` windows. It then writes what a
reader can judge directly: the received text. The packet losses are generated exactly as in
``eval_rs_uep.run_cell`` (same master-mask and random-order seeding per batch), so for the same
seed, rate, batch size and windows it reproduces the sweep's realisations.

Outputs in ``--output_dir``:

* ``text_examples.json`` / ``.md``: char accuracy, word accuracy and decoder bits/char per method
  and loss rate (bits/char = cross-entropy of the true text under the decoder's softmax; lower
  means the decoder is more confident in the right characters);
* ``examples.md``: for ``--save_examples`` windows, the first ``--excerpt_chars`` characters of
  the original and of every method at every loss rate. Wrong characters are shown in UPPER CASE
  (text8 is all lower case) and a wrong space as ``_``, so errors stand out.

    python -m experiments.eval_text_examples --checkpoint results/text/deepjsccq_text/seed42/best_val.pth \\
        --oracle_ranking_file results/text/deepjsccq_text/seed42/oracle_ranking.json --split test \\
        --selection results/text/deepjsccq_text/seed42/val_sweep/selected_schemes.json \\
        --sources oracle,energy --output_dir results/text/deepjsccq_text/seed42/examples
"""
import argparse
import json
import os

import numpy as np
import torch

from coding.erasure_channel import sample_master_mask
from coding.interleave import packet_mask_from_timeline
from coding.scaling import scheme_catalog
from data.datasets import is_official_test, load_eval_tensor
from data.text import VOCAB
from experiments.eval_speech_perceptual import method_specs   # generic: no_rs, uniform, selected scheme per source
from experiments.importance_sources import compute_orders
from experiments.metrics import bits_per_char_per_item, quality_db_per_item, word_accuracy_per_item
from experiments.model_io import encode_full, load_model_bundle
from experiments.pipeline_core import method_order, transmit
from experiments.reproducibility import build_provenance, log_test_access, seed_everything, write_json

DEFAULT_RATES = "0.0,0.3,0.35,0.4,0.5"


def highlight(reference, predicted) -> str:
    """Predicted text with every wrong character upper-cased (a wrong space shown as ``_``)."""
    out = []
    for ref, pred in zip(np.asarray(reference).tolist(), np.asarray(predicted).tolist()):
        char = VOCAB[int(pred)]
        if int(ref) != int(pred):
            char = "_" if char == " " else char.upper()
        out.append(char)
    return "".join(out)


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
    parser.add_argument("--save_examples", type=int, default=3, help="windows shown per method and rate")
    parser.add_argument("--excerpt_chars", type=int, default=240, help="characters shown per example")
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    seed_everything(args.global_seed)
    bundle = load_model_bundle(args.checkpoint, torch.device("cpu"))
    if bundle.modality != "text":
        raise SystemExit(f"{args.checkpoint} is not a text model ({bundle.arch}, {bundle.modality})")
    if is_official_test("text8", args.split):
        log_test_access("experiments.eval_text_examples", args.checkpoint, note=f"{args.max_items} windows")
    items = load_eval_tensor("text8", args.split, args.data_dir, args.max_items)
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

    rows = []
    shown = min(args.save_examples, len(items))
    excerpts = {i: {"original": "".join(VOCAB[int(c)] for c in items[i, :args.excerpt_chars])} for i in range(shown)}
    for rate in rates:
        per_method = {spec.name: {"char": [], "word": [], "bpc": []} for spec in specs}
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
                    logits, *_ = transmit(bundle, idx[start:stop], scheme, order, mask, bundle.snr_db)
                bucket = per_method[spec.name]
                bucket["char"] += quality_db_per_item(logits, x).tolist()
                bucket["word"] += word_accuracy_per_item(logits, x).tolist()
                bucket["bpc"] += bits_per_char_per_item(logits, x).tolist()
                predicted = logits.argmax(dim=1)
                for j in range(stop - start):
                    if start + j < shown:
                        excerpts[start + j][f"{spec.name} @ loss {rate:g}"] = highlight(
                            x[j, :args.excerpt_chars], predicted[j, :args.excerpt_chars])
        for name, bucket in per_method.items():
            rows.append({"method": name, "erasure_rate": rate, "items": len(bucket["char"]),
                         "char_acc_mean": float(np.mean(bucket["char"])), "word_acc_mean": float(np.mean(bucket["word"])),
                         "bits_per_char_mean": float(np.mean(bucket["bpc"]))})
        print(f"loss {rate:g}: " + ", ".join(f"{r['method']} char {r['char_acc_mean']:.1f}% word {r['word_acc_mean']:.1f}%"
                                             for r in rows[-len(specs):]))

    os.makedirs(args.output_dir, exist_ok=True)
    write_json(os.path.join(args.output_dir, "text_examples.json"), {
        "metadata": build_provenance(args=args, checkpoint_path=args.checkpoint, dataset="text8",
                                     extra_files={"oracle_ranking": args.oracle_ranking_file,
                                                  **({"selection": args.selection} if args.selection else {})},
                                     split_info={"dataset": "text8", "split": args.split},
                                     global_seed=args.global_seed),
        "rows": rows, "examples": excerpts})
    lines = ["# Text check (character / word accuracy, bits per character)", "",
             f"{len(items)} {args.split} windows of {items.shape[1]} characters, channel seed {args.seed}.", "",
             "| method | loss | char acc (%) | word acc (%) | bits/char |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['erasure_rate']:g} | {r['char_acc_mean']:.2f} | {r['word_acc_mean']:.2f} | "
                     f"{r['bits_per_char_mean']:.3f} |")
    with open(os.path.join(args.output_dir, "text_examples.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    example_lines = ["# Example passages", "",
                     f"First {args.excerpt_chars} characters of {shown} {args.split} windows. Wrong characters are in "
                     "UPPER CASE; a wrong space is shown as `_`.", ""]
    for i, versions in excerpts.items():
        example_lines += [f"## Window {i}", "", "```text"]
        width = max(len(label) for label in versions)
        example_lines += [f"{label:<{width}} | {text}" for label, text in versions.items()]
        example_lines += ["```", ""]
    with open(os.path.join(args.output_dir, "examples.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(example_lines) + "\n")
    print(f"saved {args.output_dir}/text_examples.json, text_examples.md and examples.md")


if __name__ == "__main__":
    main()
