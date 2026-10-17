"""Scheme scaling, packet interleaving, Gilbert-Elliott channel and statistics helpers."""
import numpy as np
import pytest
import torch

from coding.erasure_channel import GilbertElliottChannel, mean_loss_run_length, sample_master_mask
from coding.interleave import MODES, packet_mask_from_timeline, transmission_slots
from coding.rs_pipeline import NO_RS, SCHEME_CANDIDATES, CodingScheme
from coding.scaling import largest_remainder, scaled_scheme, scheme_catalog, split_block
from experiments.stats_utils import (
    crossover,
    kendall_tau,
    paired_difference,
    paired_vs_baseline,
    pearson,
    rankdata,
    row_spearman,
    spearman,
    topk_overlap,
)


# ---- scheme scaling -------------------------------------------------------
@pytest.mark.parametrize("name", list(SCHEME_CANDIDATES))
def test_scaled_scheme_reproduces_the_reference_schemes(name):
    assert scaled_scheme(name, 16, 64, 16, 0.5).tiers == SCHEME_CANDIDATES[name].tiers


def test_scaled_no_rs_matches_reference():
    assert scaled_scheme("no_rs", 16, 64, 16).tiers == NO_RS.tiers


@pytest.mark.parametrize("channels", [6, 8, 16, 32])
def test_scaling_to_other_channel_counts_keeps_overhead_and_data(channels):
    data = channels * 4
    for name in SCHEME_CANDIDATES:
        scheme = scaled_scheme(name, channels, 64, 16, 0.5)
        assert sum(k for _, k in scheme.tiers) == data
        assert scheme.total_packets == data + round(0.5 * data)
        assert scheme.overhead == pytest.approx(0.5)
        assert all(n <= 255 for n, _ in scheme.tiers)


def test_kodak_schemes_respect_gf256_block_limit_and_budget():
    catalog = scheme_catalog(16, 24576, 128, 0.5)
    for name, scheme in catalog.items():
        assert sum(k for _, k in scheme.tiers) == 3072
        assert max(n for n, _ in scheme.tiers) <= 255
        if name == "no_rs":
            assert scheme.total_packets == 3072
        else:
            assert scheme.total_packets == 4608


def test_symbols_per_channel_must_align_with_packets():
    with pytest.raises(ValueError):
        scaled_scheme("uniform", 16, 70, 16)


def test_largest_remainder_sums_exactly():
    parts = largest_remainder(10, [1, 1, 1])
    assert sum(parts) == 10 and max(parts) - min(parts) <= 1


def test_split_block_preserves_totals():
    blocks = split_block(1536, 768)
    assert sum(k for _, k in blocks) == 768
    assert sum(n - k for n, k in blocks) == 768
    assert all(n <= 255 for n, _ in blocks)
    assert split_block(96, 64) == [(96, 64)]
    tight = split_block(510, 255)  # naive split overflows 255; must add a block
    assert all(n <= 255 for n, _ in tight)


# ---- interleaving ---------------------------------------------------------
def test_slots_are_permutations_and_sequential_is_identity():
    scheme = SCHEME_CANDIDATES["three_tier_25_50_25"]
    assert torch.equal(transmission_slots(scheme, "sequential"), torch.arange(96))
    slots = transmission_slots(scheme, "round_robin")
    assert sorted(slots.tolist()) == list(range(96))
    assert not torch.equal(slots, torch.arange(96))


def test_round_robin_spreads_a_burst_over_all_blocks():
    scheme = SCHEME_CANDIDATES["three_tier_25_50_25"]  # blocks of 32 / 48 / 16 packets
    timeline = torch.zeros(1, 96, dtype=torch.bool)
    timeline[:, :12] = True  # a 12-packet burst at the start
    block_of = torch.repeat_interleave(torch.arange(3), torch.tensor([32, 48, 16]))
    sequential = packet_mask_from_timeline(timeline, scheme, "sequential")[0]
    spread = packet_mask_from_timeline(timeline, scheme, "round_robin")[0]
    assert set(block_of[sequential].tolist()) == {0}            # whole burst inside the top tier
    assert set(block_of[spread].tolist()) == {0, 1, 2}          # shared by every tier
    assert int(sequential.sum()) == int(spread.sum()) == 12


def test_timeline_must_be_long_enough():
    with pytest.raises(ValueError):
        packet_mask_from_timeline(torch.zeros(1, 10, dtype=torch.bool), SCHEME_CANDIDATES["uniform"])


def test_modes_are_exposed():
    assert set(MODES) == {"sequential", "round_robin"}


# ---- Gilbert-Elliott --------------------------------------------------------
def test_from_average_loss_inverts_exactly():
    channel = GilbertElliottChannel.from_average_loss(0.35, 8)
    assert channel.average_loss == pytest.approx(0.35)
    assert channel.burst_length == pytest.approx(8)


def test_empirical_loss_and_burstiness():
    channel = GilbertElliottChannel.from_average_loss(0.35, 8)
    mask = channel.sample_mask(3000, 96, torch.Generator().manual_seed(0))
    iid = sample_master_mask(3000, 0.35, torch.Generator().manual_seed(0))
    assert float(mask.float().mean()) == pytest.approx(0.35, abs=0.02)
    assert mean_loss_run_length(mask) > 1.3 * mean_loss_run_length(iid)


def test_ge_mask_is_deterministic_for_a_seed():
    channel = GilbertElliottChannel.from_average_loss(0.3, 4)
    a = channel.sample_mask(8, 96, torch.Generator().manual_seed(3))
    b = channel.sample_mask(8, 96, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_infeasible_channel_is_rejected():
    with pytest.raises(ValueError):
        GilbertElliottChannel.from_average_loss(0.9, 4)
    with pytest.raises(ValueError):
        GilbertElliottChannel.from_average_loss(0.3, 0.5)


def test_mean_run_length_counts_runs():
    mask = torch.tensor([[1, 1, 0, 1, 0, 0, 1, 1, 1]], dtype=torch.bool)
    assert mean_loss_run_length(mask) == pytest.approx(6 / 3)
    assert mean_loss_run_length(torch.zeros(1, 5, dtype=torch.bool)) == 0.0


# ---- statistics -------------------------------------------------------------
def test_rank_and_correlations():
    assert rankdata([10, 20, 20, 30]).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_row_spearman_and_topk():
    target = np.array([[3.0, 1.0, 2.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
    assert row_spearman(target, target).tolist() == pytest.approx([1.0, 1.0])
    assert row_spearman(-target, target).tolist() == pytest.approx([-1.0, -1.0])
    assert topk_overlap(target, target, 2).tolist() == [1.0, 1.0]
    assert topk_overlap(-target, target, 2).tolist() == [0.0, 0.0]


def test_crossover_interpolates():
    assert crossover([0.0, 0.5], [-1.0, 1.0]) == pytest.approx(0.25)
    assert crossover([0.0, 0.5], [1.0, 2.0]) == 0.0
    assert crossover([0.0, 0.5], [-1.0, -2.0]) is None


def test_paired_difference_uses_seed_pairs():
    result = paired_difference([21.0, 21.1, 20.9, 21.0, 21.05], [20.0] * 5)
    assert result["mean"] == pytest.approx(1.01)
    assert result["win_rate"] == 1.0
    assert result["significant"] is True
    noisy = paired_difference([21, 22, 23], [20, 20, 20])  # 3 seeds: the t-interval is too wide to call it
    assert noisy["significant"] is False


def test_paired_vs_baseline_pairs_by_seed():
    rows = []
    for seed, (a, b) in enumerate([(20.0, 21.0), (22.0, 23.5), (24.0, 25.0)]):
        rows.append({"method": "uniform", "erasure_rate": 0.4, "seed": seed, "psnr_db_mean_per_image": a})
        rows.append({"method": "tiered", "erasure_rate": 0.4, "seed": seed, "psnr_db_mean_per_image": b})
    (entry,) = paired_vs_baseline(rows)
    assert entry["method"] == "tiered" and entry["mean"] == pytest.approx((1 + 1.5 + 1) / 3)
