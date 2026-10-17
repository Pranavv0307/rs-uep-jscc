"""Tests for the Phase 0 hygiene helpers: master mask, side-info rule,
seed statistics, deterministic seeding, provenance, oracle-file loading and
the Rule 2 (no ground-truth leakage into erased packets) audit."""
import json
import math

import numpy as np
import pytest
import torch

from coding.erasure_channel import (
    MASTER_PACKET_COUNT,
    apply_erasure_mask,
    sample_master_mask,
    slice_master_mask,
)
from coding.packetizer import PacketizedLatent
from coding.rs_pipeline import (
    NO_RS,
    SCHEME_CANDIDATES,
    UNIFORM_RS,
    decode_blocks,
    encode_blocks,
    side_info_bytes,
)
from experiments.eval_rs_uep import load_oracle_ranking
from experiments.reproducibility import (
    aggregate_over_seeds,
    build_provenance,
    file_sha256,
    seed_everything,
    seed_statistics,
    t_critical_95,
)


# ---- master mask (Rule 3) ------------------------------------------------
def test_master_mask_is_deterministic_for_a_seed():
    first = sample_master_mask(8, 0.3, torch.Generator().manual_seed(5))
    second = sample_master_mask(8, 0.3, torch.Generator().manual_seed(5))
    assert first.shape == (8, MASTER_PACKET_COUNT)
    assert torch.equal(first, second)


def test_no_rs_mask_is_a_prefix_of_the_master_mask():
    master = sample_master_mask(8, 0.4, torch.Generator().manual_seed(1))
    short = slice_master_mask(master, NO_RS.total_packets)
    assert short.shape == (8, 64)
    assert torch.equal(short, master[:, :64])


def test_slice_larger_than_master_is_rejected():
    master = sample_master_mask(2, 0.1, torch.Generator().manual_seed(0))
    with pytest.raises(ValueError):
        slice_master_mask(master, MASTER_PACKET_COUNT + 1)


# ---- side-info rule ------------------------------------------------------
def test_side_info_rule():
    assert side_info_bytes("uniform") == 0
    assert side_info_bytes("no_rs") == 0
    assert side_info_bytes("three_tier_25_50_25") == 16
    assert side_info_bytes("three_tier_25_50_25_random") == 16
    assert side_info_bytes("two_tier_50_50") == 16


# ---- statistics over seeds -----------------------------------------------
def test_seed_statistics_matches_hand_calculation():
    stats = seed_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["std"] == pytest.approx(math.sqrt(2.5))  # sample std, n-1
    assert stats["ci95_half_width"] == pytest.approx(2.776 * math.sqrt(2.5) / math.sqrt(5))
    assert stats["n_seeds"] == 5


def test_single_seed_has_no_spread():
    stats = seed_statistics([7.0])
    assert stats["std"] is None and stats["ci95_half_width"] is None


def test_t_critical_never_undercovers():
    assert t_critical_95(4) == pytest.approx(2.776)
    assert t_critical_95(100) == pytest.approx(1.96)
    assert t_critical_95(22) >= t_critical_95(25)  # falls back to a smaller df


def _rows(psnrs, method="uniform", rate=0.4):
    return [
        {"method": method, "erasure_rate": rate, "seed": seed, "images": 100,
         "psnr_db_mean_per_image": value, "ssim_mean": 0.9, "realized_erasure_rate": rate,
         "side_info_bytes_per_image": 0, "coding_overhead_fraction": 0.5,
         "total_transmitted_bytes_per_image": 1536,
         "total_overhead_fraction_including_side_info": 0.5}
        for seed, value in enumerate(psnrs)
    ]


def test_aggregate_over_seeds_groups_by_method_and_rate():
    rows = _rows([20.0, 22.0, 24.0]) + _rows([30.0, 30.0, 30.0], method="oracle", rate=0.4)
    summary = {row["method"]: row for row in aggregate_over_seeds(rows)}
    assert summary["uniform"]["psnr_db_mean_per_image__mean"] == pytest.approx(22.0)
    assert summary["uniform"]["psnr_db_mean_per_image__std_over_seeds"] == pytest.approx(2.0)
    assert summary["oracle"]["psnr_db_mean_per_image__std_over_seeds"] == pytest.approx(0.0)
    assert summary["uniform"]["n_seeds"] == 3
    assert summary["uniform"]["side_info_bytes_per_image"] == 0


def test_aggregate_rejects_duplicate_seeds():
    rows = _rows([20.0, 22.0])
    rows[1]["seed"] = rows[0]["seed"]
    with pytest.raises(ValueError):
        aggregate_over_seeds(rows)


# ---- deterministic seeding (Rule 5) ---------------------------------------
def test_seed_everything_reproduces_all_rngs():
    import random

    seed_everything(123)
    first = (torch.rand(5), np.random.rand(5), random.random())
    seed_everything(123)
    second = (torch.rand(5), np.random.rand(5), random.random())
    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_seed_everything_sets_deterministic_flags():
    seed_everything(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled()


# ---- provenance (section 10) ---------------------------------------------
def test_provenance_has_required_fields_and_is_json_serializable(tmp_path):
    checkpoint = tmp_path / "ckpt.pth"
    checkpoint.write_bytes(b"abc")
    metadata = build_provenance(checkpoint_path=str(checkpoint), global_seed=7)
    for key in ("git_commit", "dirty_repo", "timestamp_utc", "python_version", "torch_version",
                "device_name", "dataset", "determinism", "command_line", "checkpoint"):
        assert key in metadata
    assert metadata["checkpoint"]["sha256"] == file_sha256(str(checkpoint))
    json.dumps(metadata)


# ---- oracle ranking file ---------------------------------------------------
@pytest.mark.parametrize("key", ["ranking_channels_most_to_least_important", "oracle_ranking_channels"])
def test_load_oracle_ranking_accepts_both_file_formats(tmp_path, key):
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps({key: list(range(15, -1, -1)), "source_split": "train"}))
    ranking, info = load_oracle_ranking(str(path))
    assert ranking == list(range(15, -1, -1))
    assert info["source_split"] == "train"
    assert len(info["sha256"]) == 64


def test_load_oracle_ranking_rejects_non_permutation(tmp_path):
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps({"oracle_ranking_channels": [0] * 16}))
    with pytest.raises(ValueError):
        load_oracle_ranking(str(path))


# ---- Rule 2 audit: erased payload must never influence the result -----------
@pytest.mark.parametrize("scheme", [UNIFORM_RS, SCHEME_CANDIDATES["three_tier_25_50_25"]])
def test_garbage_in_erased_packets_changes_nothing(scheme):
    generator = torch.Generator().manual_seed(11)
    indices = torch.randint(0, 256, (6, 1024), generator=generator)
    order = None if scheme is UNIFORM_RS else torch.stack(
        [torch.randperm(16, generator=generator) for _ in range(6)]
    )
    packets = encode_blocks(indices, scheme, order)
    mask = sample_master_mask(6, 0.45, torch.Generator().manual_seed(3))[:, : scheme.total_packets]
    received = apply_erasure_mask(packets, mask).received

    garbage = received.packets.clone()
    garbage[received.erased] = torch.randint(
        0, 256, (int(received.erased.sum()), scheme.packet_size), generator=generator, dtype=torch.long
    ).to(torch.uint8)
    garbled = PacketizedLatent(
        packets=garbage, erased=received.erased, n_symbols=received.n_symbols,
        packet_size=received.packet_size, orig_shape=received.orig_shape,
    )

    clean = decode_blocks(received, scheme, order)
    dirty = decode_blocks(garbled, scheme, order)
    assert torch.equal(clean.failed, dirty.failed)
    assert torch.equal(clean.block_failed, dirty.block_failed)
    kept = ~clean.failed  # failed positions are replaced by 0.0 in latent space, so their index is irrelevant
    assert torch.equal(clean.indices[kept], dirty.indices[kept])
