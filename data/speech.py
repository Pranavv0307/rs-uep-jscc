"""LibriSpeech segments for the speech study (S1): download, segment cache, loaders.

Splits (speaker-disjoint by LibriSpeech's construction; AD-15)
-------------------------------------------------------------
=========  ===================  ==========================================
split      LibriSpeech subset   role
=========  ===================  ==========================================
``train``  train-clean-100      optimisation + train-split oracle ranking
``val``    dev-clean            checkpoint choice, scheme selection
``test``   test-clean           final tables only (every access is logged)
=========  ===================  ==========================================

Segments
--------
16 kHz mono, ``SEGMENT_LENGTH = 16384`` samples (1.024 s), non-overlapping, taken
from the start of each utterance. A segment is kept only if its RMS is at least
``SILENCE_DBFS`` (-40 dBFS): near-silent segments make per-segment SNR meaningless.
At most ``SEGMENTS_PER_UTTERANCE`` (2) segments are drawn per utterance (seeded),
the pool is shuffled (seeded) and capped (train 40,000 to match CIFAR-10's 40k;
val / test 4,000 each). Everything is deterministic given ``DATA_SEED``.

The cache is ``<data_dir>/librispeech_segments/<split>.npy`` (int16, shape (N, L))
plus ``<split>.json`` (parameters, per-segment utterance / speaker / offset / level,
SHA-256 of the array). Building it reads each FLAC with ``soundfile``. Training
streams from a memory map; evaluation loads a float32 tensor ``(N, 1, L)`` in [-1, 1].

    python -m experiments.prepare_speech_data --download --build --data_dir ./data
"""
import glob
import hashlib
import json
import os
import tarfile
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.reproducibility import seed_worker

SAMPLE_RATE = 16000
SEGMENT_LENGTH = 16384
SILENCE_DBFS = -40.0
SEGMENTS_PER_UTTERANCE = 2
DATA_SEED = 42
SPLIT_SUBSETS = {"train": "train-clean-100", "val": "dev-clean", "test": "test-clean"}
DEFAULT_CAPS = {"train": 40_000, "val": 4_000, "test": 4_000}
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{subset}.tar.gz"
CACHE_DIR = "librispeech_segments"
INT16_SCALE = 32768.0


# --------------------------------------------------------------------------
# Download (official OpenSLR archives)
# --------------------------------------------------------------------------
def _safe_members(archive: tarfile.TarFile, root: str):
    """Regular files and directories that stay inside ``root`` (no links, devices or ``..``)."""
    root = os.path.realpath(root)
    for member in archive.getmembers():
        target = os.path.realpath(os.path.join(root, member.name))
        if not (target == root or target.startswith(root + os.sep)):
            raise RuntimeError(f"unsafe path in archive: {member.name}")
        if member.isfile() or member.isdir():
            yield member


def download_subset(data_dir: str, subset: str, progress: Optional[Callable[[str], None]] = print) -> str:
    """Fetch and extract one LibriSpeech subset into ``<data_dir>/LibriSpeech/<subset>`` (idempotent)."""
    target = os.path.join(data_dir, "LibriSpeech", subset)
    if os.path.isdir(target) and glob.glob(os.path.join(target, "*", "*", "*.flac")):
        return target
    archive_dir = os.path.join(data_dir, "LibriSpeech_archives")
    os.makedirs(archive_dir, exist_ok=True)
    archive = os.path.join(archive_dir, f"{subset}.tar.gz")
    if not os.path.exists(archive):
        url = LIBRISPEECH_URL.format(subset=subset)
        if progress:
            progress(f"downloading {url}")
        partial = archive + ".part"
        urllib.request.urlretrieve(url, partial)
        os.replace(partial, archive)
    if progress:
        progress(f"extracting {archive}")
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(data_dir, members=list(_safe_members(handle, data_dir)))
    return target


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Segmenting
# --------------------------------------------------------------------------
def list_utterances(subset_root: str) -> List[str]:
    """``<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac``, sorted (deterministic)."""
    return sorted(glob.glob(os.path.join(subset_root, "*", "*", "*.flac")))


def utterance_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def speaker_id(path: str) -> str:
    return utterance_id(path).split("-")[0]


def level_dbfs(segment: np.ndarray) -> float:
    """RMS level of an int16 segment in dB relative to full scale."""
    rms = float(np.sqrt(np.mean((segment.astype(np.float64) / INT16_SCALE) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-12))


def candidate_offsets(audio: np.ndarray, segment_length: int = SEGMENT_LENGTH,
                      silence_dbfs: float = SILENCE_DBFS) -> List[Tuple[int, float]]:
    """``(offset, level_dbfs)`` of every non-overlapping, non-silent segment of one utterance."""
    out = []
    for offset in range(0, len(audio) - segment_length + 1, segment_length):
        level = level_dbfs(audio[offset:offset + segment_length])
        if level >= silence_dbfs:
            out.append((offset, level))
    return out


def read_flac(path: str) -> np.ndarray:
    import soundfile  # imported lazily: only needed when building the cache

    audio, rate = soundfile.read(path, dtype="int16", always_2d=False)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {rate}")
    if audio.ndim != 1:
        raise ValueError(f"{path}: expected mono audio")
    return audio


def cache_paths(data_dir: str, split: str) -> Tuple[str, str]:
    folder = os.path.join(data_dir, CACHE_DIR)
    return os.path.join(folder, f"{split}.npy"), os.path.join(folder, f"{split}.json")


def build_split(data_dir: str, split: str, cap: Optional[int] = None, segment_length: int = SEGMENT_LENGTH,
                per_utterance: int = SEGMENTS_PER_UTTERANCE, silence_dbfs: float = SILENCE_DBFS,
                seed: int = DATA_SEED, subset_root: Optional[str] = None,
                progress: Optional[Callable[[Sequence], Sequence]] = None) -> Dict:
    """Write the segment cache of one split; returns its manifest.

    Two passes: (1) read every utterance, list its non-silent segments and draw at most
    ``per_utterance`` of them; (2) shuffle the pool, cap it, and write the chosen
    segments into an int16 memory map (each file is read once, in sorted order).
    """
    cap = DEFAULT_CAPS[split] if cap is None else cap
    subset = SPLIT_SUBSETS[split]
    root = subset_root or os.path.join(data_dir, "LibriSpeech", subset)
    files = list_utterances(root)
    if not files:
        raise FileNotFoundError(f"no .flac files under {root}; run with --download first")
    rng = np.random.default_rng([seed, list(SPLIT_SUBSETS).index(split)])

    pool: List[Tuple[str, int, float]] = []
    for path in (progress(files) if progress else files):
        candidates = candidate_offsets(read_flac(path), segment_length, silence_dbfs)
        if not candidates:
            continue
        take = min(per_utterance, len(candidates))
        for index in sorted(rng.choice(len(candidates), size=take, replace=False).tolist()):
            pool.append((path, *candidates[index]))
    order = rng.permutation(len(pool))[:cap]
    chosen = [pool[i] for i in order]

    npy_path, json_path = cache_paths(data_dir, split)
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    array = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.int16, shape=(len(chosen), segment_length))
    by_file: Dict[str, List[Tuple[int, int]]] = {}
    for slot, (path, offset, _) in enumerate(chosen):
        by_file.setdefault(path, []).append((slot, offset))
    for path in sorted(by_file):
        audio = read_flac(path)
        for slot, offset in by_file[path]:
            array[slot] = audio[offset:offset + segment_length]
    array.flush()
    del array

    manifest = {
        "dataset": "LibriSpeech", "split": split, "subset": subset, "num_segments": len(chosen),
        "segment_length": segment_length, "sample_rate": SAMPLE_RATE, "silence_dbfs": silence_dbfs,
        "segments_per_utterance": per_utterance, "cap": cap, "seed": seed,
        "utterances_scanned": len(files), "pool_size": len(pool),
        "speakers": len({speaker_id(p) for p, _, _ in chosen}),
        "array_sha256": file_sha256(npy_path),
        "segments": [{"utterance": utterance_id(p), "speaker": speaker_id(p), "offset": int(o),
                      "level_dbfs": round(float(level), 2)} for p, o, level in chosen],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    return manifest


def load_manifest(data_dir: str, split: str) -> Dict:
    _, json_path = cache_paths(data_dir, split)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} missing; run: python -m experiments.prepare_speech_data --build")
    with open(json_path, encoding="utf-8") as handle:
        return json.load(handle)


def split_info(data_dir: str) -> Dict:
    """Short provenance block for every split (no per-segment list)."""
    info = {}
    for split in SPLIT_SUBSETS:
        try:
            manifest = load_manifest(data_dir, split)
        except FileNotFoundError:
            continue
        info[split] = {k: manifest[k] for k in ("subset", "num_segments", "speakers", "array_sha256", "seed",
                                                "silence_dbfs", "segment_length")}
    return info


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def to_float(segments: np.ndarray) -> torch.Tensor:
    """int16 ``(N, L)`` -> float32 ``(N, 1, L)`` in [-1, 1)."""
    return torch.from_numpy(np.asarray(segments, dtype=np.float32) / INT16_SCALE).unsqueeze(1)


class SpeechSegments(Dataset):
    """Memory-mapped segments; items are ``(waveform (1, L), 0)`` like ``(image, label)``.

    The array is opened lazily in each worker, so the dataset pickles cheaply
    (DataLoader workers use the ``spawn`` start method on macOS).
    """

    def __init__(self, data_dir: str, split: str, max_items: Optional[int] = None):
        self.path, _ = cache_paths(data_dir, split)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"{self.path} missing; run: python -m experiments.prepare_speech_data --build")
        rows = int(np.load(self.path, mmap_mode="r").shape[0])
        self.length = rows if max_items is None else min(rows, max_items)
        self._array = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        return to_float(self._array[index][None])[0], 0

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_array"] = None
        return state


def load_speech_tensor(split: str, data_dir: str, max_items: Optional[int] = None) -> torch.Tensor:
    """First ``max_items`` segments of a split as one float32 tensor ``(N, 1, L)`` (deterministic order)."""
    path, _ = cache_paths(data_dir, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; run: python -m experiments.prepare_speech_data --build")
    array = np.load(path, mmap_mode="r")
    count = array.shape[0] if max_items is None else min(max_items, array.shape[0])
    return to_float(array[:count])


def make_speech_loaders(data_dir: str, batch_size: int, num_workers: int, loader_seed: int,
                        max_train: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Train (shuffled, seeded, drop_last) / validation / test loaders, mirroring ``data.splits.make_loaders``."""
    generator = torch.Generator().manual_seed(loader_seed)
    common = dict(num_workers=num_workers, worker_init_fn=seed_worker, persistent_workers=num_workers > 0)
    train = DataLoader(SpeechSegments(data_dir, "train", max_train), batch_size=batch_size, shuffle=True,
                       generator=generator, drop_last=True, **common)
    val = DataLoader(SpeechSegments(data_dir, "val"), batch_size=batch_size, shuffle=False, **common)
    test = DataLoader(SpeechSegments(data_dir, "test"), batch_size=batch_size, shuffle=False, **common)
    return train, val, test
