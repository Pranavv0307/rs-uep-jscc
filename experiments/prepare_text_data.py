"""Download text8 and build the text window caches (T1). Run once per machine.

    python -m experiments.prepare_text_data --download --build --data_dir ./data

``--download`` fetches ``text8.zip`` (31 MB) from mattmahoney.net into ``<data_dir>/text8/`` and
extracts the single 100 MB file ``text8``. An existing, complete file is kept.
``--build`` writes ``<data_dir>/text8_windows/{train,val,test}.{npy,json}`` (see ``data/text.py``
for the windowing rules); an existing split is kept unless ``--force``. The JSON manifest records
every window offset, the array's SHA-256 and the source file's SHA-256, so two machines can check
that they evaluate the same data.
"""
import argparse
import os

from data.text import (
    DEFAULT_CAPS,
    SPLIT_RANGES,
    build_split,
    cache_paths,
    download_text8,
    load_manifest,
    read_text8,
    source_path,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--train_cap", type=int, default=DEFAULT_CAPS["train"])
    parser.add_argument("--eval_cap", type=int, default=None, help="windows in val and in test (default: all 2,441)")
    parser.add_argument("--force", action="store_true", help="rebuild caches that already exist")
    args = parser.parse_args()
    if not (args.download or args.build):
        parser.error("choose --download and/or --build")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = set(splits) - set(SPLIT_RANGES)
    if unknown:
        parser.error(f"unknown splits {sorted(unknown)}; choose from {sorted(SPLIT_RANGES)}")

    if args.download:
        print(f"text8: {download_text8(args.data_dir)}")

    if args.build:
        corpus = None
        for split in splits:
            npy_path, _ = cache_paths(args.data_dir, split)
            if os.path.exists(npy_path) and not args.force:
                manifest = load_manifest(args.data_dir, split)
                print(f"{split}: kept existing cache ({manifest['num_segments']} windows, "
                      f"sha256 {manifest['array_sha256'][:12]}...)")
                continue
            if corpus is None:
                corpus = read_text8(source_path(args.data_dir))   # validates length and alphabet
            cap = args.train_cap if split == "train" else args.eval_cap
            manifest = build_split(args.data_dir, split, cap=cap, corpus=corpus)
            print(f"{split}: {manifest['num_segments']} windows of {manifest['segment_length']} characters "
                  f"(of {manifest['windows_available']} available), sha256 {manifest['array_sha256']}")
            if cap is not None and manifest["num_segments"] < cap:
                print(f"WARNING: {split} has fewer windows than the cap {cap}")


if __name__ == "__main__":
    main()
