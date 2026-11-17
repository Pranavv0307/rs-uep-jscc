"""text8 windows for the text study (T1): download, window cache, loaders.

Source
------
text8 (Mahoney, http://mattmahoney.net/dc/textdata.html): the first 10^8 bytes of an English
Wikipedia dump (March 2006), cleaned to lowercase ``a-z`` and single spaces, so the
alphabet has exactly 27 symbols. It is the standard character-level benchmark and ships
as one 31 MB zip.

Splits (the standard contiguous text8 split; AD-23)
--------------------------------------------------
=========  ======================  ==========================================
split      characters              role
=========  ======================  ==========================================
``train``  [0, 90,000,000)         optimisation + train-split oracle ranking
``val``    [90M, 95,000,000)       checkpoint choice, scheme selection
``test``   [95M, 100,000,000)      final tables only (every access is logged)
=========  ======================  ==========================================

The three regions are disjoint stretches of text (different articles, apart from the two
boundary articles), as in every character-level language-modelling paper that uses text8.

Windows
-------
``SEGMENT_LENGTH = 2048`` characters, non-overlapping, aligned to the start of each split
region; the trailing remainder is dropped. Symbols are stored as ids (space = 0, a-z = 1-26,
``VOCAB``). The window list of each split is shuffled (seeded) and capped: train 40,000 of
43,945 (to match CIFAR-10's and LibriSpeech's 40k), val / test all 2,441 each. Everything is
deterministic given ``DATA_SEED``.

The cache is ``<data_dir>/text8_windows/<split>.npy`` (uint8, shape (N, L)) plus
``<split>.json`` (parameters, per-window character offset, SHA-256 of the array and of the
source file). Training streams from a memory map; evaluation loads an int64 tensor ``(N, L)``.

    python -m experiments.prepare_text_data --download --build --data_dir ./data
"""
import hashlib
import json
import os
import urllib.request
import zipfile
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.reproducibility import seed_worker

VOCAB = " abcdefghijklmnopqrstuvwxyz"   # id = position; space = 0
SPACE_ID = 0
SEGMENT_LENGTH = 2048
DATA_SEED = 42
TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
TEXT8_CHARS = 100_000_000
SPLIT_RANGES = {"train": (0, 90_000_000), "val": (90_000_000, 95_000_000), "test": (95_000_000, 100_000_000)}
DEFAULT_CAPS = {"train": 40_000, "val": None, "test": None}   # None = every window of the split
CACHE_DIR = "text8_windows"
SOURCE_DIR = "text8"

_LOOKUP = np.full(256, 255, dtype=np.uint8)
for _index, _char in enumerate(VOCAB):
    _LOOKUP[ord(_char)] = _index


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(data_dir: str) -> str:
    return os.path.join(data_dir, SOURCE_DIR, "text8")


def download_text8(data_dir: str, progress: Optional[Callable[[str], None]] = print) -> str:
    """Fetch ``text8.zip`` and extract its single member to ``<data_dir>/text8/text8`` (idempotent).

    Only the member named exactly ``text8`` is read, and it is written to a fixed path, so a
    tampered archive cannot place files elsewhere.
    """
    target = source_path(data_dir)
    if os.path.exists(target) and os.path.getsize(target) == TEXT8_CHARS:
        return target
    archive = os.path.join(data_dir, SOURCE_DIR, "text8.zip")
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    if not os.path.exists(archive):
        if progress:
            progress(f"downloading {TEXT8_URL}")
        partial = archive + ".part"
        urllib.request.urlretrieve(TEXT8_URL, partial)
        os.replace(partial, archive)
    if progress:
        progress(f"extracting {archive}")
    with zipfile.ZipFile(archive) as handle:
        if "text8" not in handle.namelist():
            raise RuntimeError(f"{archive} has no member 'text8' (members: {handle.namelist()[:5]})")
        partial = target + ".part"
        with handle.open("text8") as source, open(partial, "wb") as sink:
            for block in iter(lambda: source.read(1 << 20), b""):
                sink.write(block)
        os.replace(partial, target)
    return target


def read_text8(path: str, expected_length: Optional[int] = TEXT8_CHARS) -> np.ndarray:
    """The whole corpus as uint8 symbol ids; refuses characters outside ``VOCAB``."""
    with open(path, "rb") as handle:
        raw = np.frombuffer(handle.read(), dtype=np.uint8)
    if expected_length is not None and raw.size != expected_length:
        raise ValueError(f"{path}: expected {expected_length} characters, got {raw.size}")
    ids = _LOOKUP[raw]
    if (ids == 255).any():
        bad = sorted({chr(c) for c in raw[ids == 255][:100]})
        raise ValueError(f"{path}: characters outside the text8 alphabet: {bad}")
    return ids


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------
def encode_text(text: str) -> np.ndarray:
    """``str`` -> uint8 ids (helper for tests and examples)."""
    ids = _LOOKUP[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]
    if (ids == 255).any():
        raise ValueError("text contains characters outside the text8 alphabet")
    return ids


def decode_ids(ids) -> str:
    """ids (any int array / tensor, 1-D) -> ``str``."""
    return "".join(VOCAB[int(i)] for i in np.asarray(ids).reshape(-1))


def window_offsets(split: str, segment_length: int = SEGMENT_LENGTH,
                   ranges: Dict[str, Tuple[int, int]] = SPLIT_RANGES) -> np.ndarray:
    """Start offsets (in the corpus) of every full, non-overlapping window of a split."""
    start, stop = ranges[split]
    return np.arange(start, stop - segment_length + 1, segment_length, dtype=np.int64)


def cache_paths(data_dir: str, split: str) -> Tuple[str, str]:
    folder = os.path.join(data_dir, CACHE_DIR)
    return os.path.join(folder, f"{split}.npy"), os.path.join(folder, f"{split}.json")


def build_split(data_dir: str, split: str, cap: Optional[int] = None, segment_length: int = SEGMENT_LENGTH,
                seed: int = DATA_SEED, corpus: Optional[np.ndarray] = None, source: Optional[str] = None,
                ranges: Dict[str, Tuple[int, int]] = SPLIT_RANGES) -> Dict:
    """Write the window cache of one split; returns its manifest.

    ``corpus`` (uint8 ids) may be passed to avoid re-reading the 100 MB file per split;
    ``ranges`` exists for tests on a tiny corpus.
    """
    cap = DEFAULT_CAPS[split] if cap is None and split in DEFAULT_CAPS else cap
    source = source or source_path(data_dir)
    if corpus is None:
        if not os.path.exists(source):
            raise FileNotFoundError(f"{source} missing; run: python -m experiments.prepare_text_data --download")
        corpus = read_text8(source)
    offsets = window_offsets(split, segment_length, ranges)
    rng = np.random.default_rng([seed, list(SPLIT_RANGES).index(split)])
    order = rng.permutation(len(offsets))
    if cap is not None:
        order = order[:cap]
    chosen = offsets[order]

    npy_path, json_path = cache_paths(data_dir, split)
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    array = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint8, shape=(len(chosen), segment_length))
    for slot, offset in enumerate(chosen):
        array[slot] = corpus[offset:offset + segment_length]
    array.flush()
    del array

    manifest = {
        "dataset": "text8", "split": split, "char_range": list(ranges[split]), "num_segments": int(len(chosen)),
        "segment_length": segment_length, "vocab": VOCAB, "cap": cap, "seed": seed,
        "windows_available": int(len(offsets)),
        "source_sha256": file_sha256(source) if os.path.exists(source) else None,
        "array_sha256": file_sha256(npy_path),
        "offsets": [int(o) for o in chosen],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    return manifest


def load_manifest(data_dir: str, split: str) -> Dict:
    _, json_path = cache_paths(data_dir, split)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} missing; run: python -m experiments.prepare_text_data --build")
    with open(json_path, encoding="utf-8") as handle:
        return json.load(handle)


def split_info(data_dir: str) -> Dict:
    """Short provenance block for every split (no per-window list)."""
    info = {}
    for split in SPLIT_RANGES:
        try:
            manifest = load_manifest(data_dir, split)
        except FileNotFoundError:
            continue
        info[split] = {k: manifest[k] for k in ("char_range", "num_segments", "array_sha256", "source_sha256", "seed",
                                                "segment_length")}
    return info


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def to_ids(windows: np.ndarray) -> torch.Tensor:
    """uint8 ``(N, L)`` -> int64 ``(N, L)`` (the embedding needs long indices)."""
    return torch.from_numpy(np.asarray(windows, dtype=np.int64))


class TextSegments(Dataset):
    """Memory-mapped windows; items are ``(ids (L,), 0)`` like ``(image, label)``.

    The array is opened lazily in each worker, so the dataset pickles cheaply
    (DataLoader workers use the ``spawn`` start method on macOS and Windows).
    """

    def __init__(self, data_dir: str, split: str, max_items: Optional[int] = None):
        self.path, _ = cache_paths(data_dir, split)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"{self.path} missing; run: python -m experiments.prepare_text_data --build")
        rows = int(np.load(self.path, mmap_mode="r").shape[0])
        self.length = rows if max_items is None else min(rows, max_items)
        self._array = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        return to_ids(self._array[index][None])[0], 0

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_array"] = None
        return state


def load_text_tensor(split: str, data_dir: str, max_items: Optional[int] = None) -> torch.Tensor:
    """First ``max_items`` windows of a split as one int64 tensor ``(N, L)`` (deterministic order)."""
    path, _ = cache_paths(data_dir, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; run: python -m experiments.prepare_text_data --build")
    array = np.load(path, mmap_mode="r")
    count = array.shape[0] if max_items is None else min(max_items, array.shape[0])
    return to_ids(array[:count])


def make_text_loaders(data_dir: str, batch_size: int, num_workers: int, loader_seed: int,
                      max_train: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Train (shuffled, seeded, drop_last) / validation / test loaders, mirroring ``data.speech.make_speech_loaders``."""
    generator = torch.Generator().manual_seed(loader_seed)
    common = dict(num_workers=num_workers, worker_init_fn=seed_worker, persistent_workers=num_workers > 0)
    train = DataLoader(TextSegments(data_dir, "train", max_train), batch_size=batch_size, shuffle=True,
                       generator=generator, drop_last=True, **common)
    val = DataLoader(TextSegments(data_dir, "val"), batch_size=batch_size, shuffle=False, **common)
    test = DataLoader(TextSegments(data_dir, "test"), batch_size=batch_size, shuffle=False, **common)
    return train, val, test
