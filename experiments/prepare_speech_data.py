"""Download LibriSpeech and build the speech segment caches (S1). Run once per machine.

    python -m experiments.prepare_speech_data --download --build --data_dir ./data

``--download`` fetches the three official OpenSLR archives (train-clean-100 6.3 GB,
dev-clean 0.3 GB, test-clean 0.3 GB) into ``<data_dir>/LibriSpeech_archives`` and
extracts them to ``<data_dir>/LibriSpeech``. Already-extracted subsets are skipped.
``--build`` writes ``<data_dir>/librispeech_segments/{train,val,test}.{npy,json}``
(see ``data/speech.py`` for the segmenting rules); an existing split is kept unless
``--force``. The JSON manifest records every segment and the array's SHA-256, so two
machines can check that they evaluate the same data.
"""
import argparse
import os

from data.speech import DEFAULT_CAPS, SPLIT_SUBSETS, build_split, cache_paths, download_subset, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--train_cap", type=int, default=DEFAULT_CAPS["train"])
    parser.add_argument("--eval_cap", type=int, default=DEFAULT_CAPS["val"], help="segments in val and in test")
    parser.add_argument("--force", action="store_true", help="rebuild caches that already exist")
    args = parser.parse_args()
    if not (args.download or args.build):
        parser.error("choose --download and/or --build")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = set(splits) - set(SPLIT_SUBSETS)
    if unknown:
        parser.error(f"unknown splits {sorted(unknown)}; choose from {sorted(SPLIT_SUBSETS)}")

    if args.download:
        for split in splits:
            print(f"{split}: {download_subset(args.data_dir, SPLIT_SUBSETS[split])}")

    if args.build:
        try:
            from tqdm import tqdm as progress
        except ImportError:  # pragma: no cover
            progress = None
        for split in splits:
            npy_path, _ = cache_paths(args.data_dir, split)
            if os.path.exists(npy_path) and not args.force:
                manifest = load_manifest(args.data_dir, split)
                print(f"{split}: kept existing cache ({manifest['num_segments']} segments, "
                      f"sha256 {manifest['array_sha256'][:12]}...)")
                continue
            cap = args.train_cap if split == "train" else args.eval_cap
            manifest = build_split(args.data_dir, split, cap=cap, progress=progress)
            print(f"{split}: {manifest['num_segments']} segments from {manifest['speakers']} speakers "
                  f"({manifest['utterances_scanned']} utterances scanned, pool {manifest['pool_size']}), "
                  f"sha256 {manifest['array_sha256'][:12]}...")
            if manifest["num_segments"] < cap:
                print(f"WARNING: {split} has fewer segments than the cap {cap}")


if __name__ == "__main__":
    main()
