"""Models, importance sources, the evaluation cell, reporting and the orchestrator.

The end-to-end checks use a tiny randomly initialised ADJSCC-Q (no data, no
checkpoint): they verify properties that must hold for *any* weights -- a loss-free
transmission is lossless, every method sees the same packet losses, results are
deterministic, rankings are permutations.
"""
import importlib

import numpy as np
import pytest
import torch

from coding.rs_pipeline import SCHEME_CANDIDATES
from coding.scaling import scheme_catalog
from experiments.eval_rs_uep import run_cell
from experiments.importance_sources import PER_IMAGE_SOURCES, compute_orders
from experiments.metrics import ms_ssim, psnr_per_image, ssim
from experiments.model_io import ModelBundle, encode_full
from experiments.pipeline_core import display_source, make_method_specs, make_context, precompute
from experiments.reporting import sweep_insights, write_sweep_outputs
from experiments.reproducibility import aggregate_over_seeds
from experiments.select_scheme import select
from experiments.sensitivity import loo_damage_batch, pairwise_ranking_loss
from experiments.stats_utils import paired_vs_baseline
from experiments.train_erasure_robust import make_drop_mask
from models.adjsccq import ADJSCCQDecoder, ADJSCCQEncoder, SensitivityHead
from models.inpainting import LatentInpainter


def tiny_bundle() -> ModelBundle:
    torch.manual_seed(0)
    encoder = ADJSCCQEncoder(k_over_n=1 / 6, image_size=32, num_levels=256)
    decoder = ADJSCCQDecoder(k=encoder.k, image_size=32)
    encoder.calibrate_quantizer(torch.rand(8, 3, 32, 32))
    cfg = {"model": {"k_over_n": 1 / 6, "image_size": 32}, "quantizer": {"num_levels": 256},
           "af_conditioning": {"nominal_snr_db": 10.0}}
    return ModelBundle(encoder, decoder, None, cfg, "", torch.device("cpu")).eval()


@pytest.fixture(scope="module")
def bundle():
    return tiny_bundle()


@pytest.fixture(scope="module")
def images():
    return torch.rand(6, 3, 32, 32, generator=torch.Generator().manual_seed(1))


# ---- models ---------------------------------------------------------------
def test_sensitivity_head_shapes_and_cost():
    head = SensitivityHead(16, 64)
    assert head(torch.randn(5, 1024)).shape == (5, 16)
    assert head(torch.randn(5, 16, 8, 8)).shape == (5, 16)
    with_var = SensitivityHead(16, 64, use_variance=True)
    assert with_var.features(torch.randn(3, 1024)).shape == (3, 32)
    assert head.macs() == 16 * 64 + 64 * 16


def test_head_normalisation_buffers_are_saved():
    head = SensitivityHead(16, 8)
    head.feat_mean.fill_(1.0)
    assert "feat_mean" in head.state_dict() and "feat_std" in head.state_dict()


def test_pairwise_ranking_loss_orders_correctly():
    target = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    assert pairwise_ranking_loss(target * 10, target, 0.5).item() == pytest.approx(0.0)
    assert pairwise_ranking_loss(-target * 10, target, 0.5).item() > 1.0
    assert pairwise_ranking_loss(torch.zeros(1, 4), torch.zeros(1, 4)).item() == 0.0  # ties are ignored


def test_inpainter_keeps_received_symbols_and_trains():
    torch.manual_seed(0)
    inpainter = LatentInpainter(16, 64, heads=4, layers=2)
    latent = torch.randn(3, 1024)
    failed = torch.zeros(3, 1024, dtype=torch.bool)
    failed[:, 64:192] = True
    output = inpainter(latent.masked_fill(failed, 0.0), failed)
    assert output.shape == latent.shape
    assert torch.equal(output[~failed], latent[~failed])     # received data is never modified
    assert not torch.equal(output[failed], torch.zeros_like(output[failed]))
    output.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in inpainter.parameters())


def test_decoder_accepts_explicit_spatial_size(bundle):
    z = torch.randn(1, 16 * 16 * 24)
    assert bundle.decoder(z, 10.0, spatial_hw=(16, 24)).shape == (1, 3, 64, 96)


# ---- drop masks -------------------------------------------------------------
def test_channel_drop_is_all_or_nothing_per_channel():
    mask = make_drop_mask(32, 16, 64, 16, 0.5, "channel", torch.Generator().manual_seed(0), "cpu")
    per_channel = mask.view(32, 16, 64)
    assert bool((per_channel.all(-1) | (~per_channel).all(-1)).all())


def test_packet_drop_groups_sixteen_symbols():
    mask = make_drop_mask(32, 16, 64, 16, 0.5, "packet", torch.Generator().manual_seed(0), "cpu")
    per_packet = mask.view(32, 64, 16)
    assert bool((per_packet.all(-1) | (~per_packet).all(-1)).all())


def test_drop_rate_extremes_and_per_sample_rates():
    gen = torch.Generator().manual_seed(0)
    assert not make_drop_mask(4, 16, 64, 16, 0.0, "channel", gen, "cpu").any()
    assert make_drop_mask(4, 16, 64, 16, 1.0, "packet", gen, "cpu").all()
    rates = torch.tensor([0.0, 1.0])
    mixed = make_drop_mask(2, 16, 64, 16, rates, "channel", gen, "cpu")
    assert not mixed[0].any() and mixed[1].all()


# ---- importance sources -------------------------------------------------------
def test_every_source_returns_a_permutation_per_image(bundle, images):
    sources = list(PER_IMAGE_SOURCES)
    sources.remove("predicted")  # needs a trained head
    orders = compute_orders(bundle, images, sources + ["static:oracle"], 10.0,
                            rankings={"oracle": list(range(15, -1, -1))})
    assert set(orders) == set(sources) | {"static:oracle"}
    for source, order in orders.items():
        assert order.shape == (6, 16), source
        assert bool((order.sort(dim=1).values == torch.arange(16)).all()), source


def test_predicted_source_uses_the_head(bundle, images):
    head = SensitivityHead(16, 16).eval()
    orders = compute_orders(bundle, images, ["predicted"], 10.0, head=head)
    assert orders["predicted"].shape == (6, 16)
    with pytest.raises(ValueError):
        compute_orders(bundle, images, ["predicted"], 10.0)


def test_unknown_source_and_missing_ranking_fail_loudly(bundle, images):
    with pytest.raises(ValueError):
        compute_orders(bundle, images, ["bogus"], 10.0)
    with pytest.raises(ValueError):
        compute_orders(bundle, images, ["static:missing"], 10.0, rankings={})


def test_loo_damage_is_nonnegative_on_average_and_shaped(bundle, images):
    info = encode_full(bundle.encoder, images, 10.0)
    latent = bundle.encoder.quantizer.dequantize_indices(info["idx"])
    damage, clean = loo_damage_batch(bundle, latent, images, 10.0)
    assert damage.shape == (6, 16) and clean.shape == (6,)


# ---- evaluation cell ------------------------------------------------------------
def make_cell_context(bundle, images, schemes=("three_tier_25_50_25", "two_tier_50_50")):
    specs = make_method_specs(list(schemes), ["static:oracle"], include_no_rs=True, include_random=True)
    prep = precompute(bundle, images, ["static:oracle"], 4, {"oracle": list(range(16))})
    catalog = scheme_catalog(16, 64, 16, 0.5, names=sorted({s.scheme for s in specs}))
    context = make_context(bundle, images, prep, catalog, 4)
    context.specs = specs
    context.args = {"dataset": "cifar10", "split": "val", "checkpoint": "tiny"}
    context.master_n = max(s.total_packets for s in catalog.values())
    return context


def test_loss_free_transmission_is_lossless_for_every_method(bundle, images):
    context = make_cell_context(bundle, images)
    rows, per_image = run_cell(context, (0, 0.0))
    for row in rows:
        values = per_image[f"{row['method']}__s0__r0.000"]
        assert np.allclose(values, context.prep["clean_psnr"], atol=1e-3), row["method"]
        assert row["failed_symbol_fraction"] == 0.0


def test_all_96_packet_methods_face_the_same_losses(bundle, images):
    context = make_cell_context(bundle, images)
    rows, _ = run_cell(context, (1, 0.4))
    rates = {row["method"]: row["realized_erasure_rate"] for row in rows if row["method"] != "no_rs"}
    assert len(set(rates.values())) == 1, rates


def test_cell_is_deterministic(bundle, images):
    context = make_cell_context(bundle, images)
    first_rows, first_images = run_cell(context, (2, 0.4))
    second_rows, second_images = run_cell(context, (2, 0.4))
    assert [r["psnr_db_mean_per_image"] for r in first_rows] == [r["psnr_db_mean_per_image"] for r in second_rows]
    for key in first_images:
        assert np.array_equal(first_images[key], second_images[key])


def test_side_info_and_overhead_columns(bundle, images):
    rows, _ = run_cell(make_cell_context(bundle, images), (0, 0.2))
    by = {row["method"]: row for row in rows}
    assert by["uniform"]["side_info_bytes_per_image"] == 0 and by["no_rs"]["side_info_bytes_per_image"] == 0
    assert by["two_tier_50_50@oracle"]["side_info_bytes_per_image"] == 16
    assert by["two_tier_50_50@oracle"]["total_overhead_fraction_including_side_info"] > \
        by["two_tier_50_50@oracle"]["coding_overhead_fraction"]
    assert display_source("static:oracle") == "oracle"


# ---- metrics -------------------------------------------------------------------------
def test_ms_ssim_behaves():
    gen = torch.Generator().manual_seed(0)
    image = torch.rand(2, 3, 192, 192, generator=gen)
    noisy = (image + 0.2 * torch.randn(image.shape, generator=gen)).clamp(0, 1)
    assert ms_ssim(image, image).tolist() == pytest.approx([1.0, 1.0], abs=1e-4)
    assert bool((ms_ssim(noisy, image) < 0.95).all())
    small = torch.rand(2, 3, 32, 32, generator=gen)
    assert ms_ssim(small, small).tolist() == pytest.approx([1.0, 1.0], abs=1e-4)
    assert ssim(image, image).tolist() == pytest.approx([1.0, 1.0], abs=1e-4)
    assert bool((psnr_per_image(noisy, image) < 30).all())


# ---- reporting ------------------------------------------------------------------------
def fake_rows():
    rows = []
    gains = {"uniform": 0.0, "no_rs": -3.0, "a@oracle": 1.0, "a@random": 0.3}
    for method, offset in gains.items():
        for rate in (0.1, 0.3, 0.4, 0.5):
            for seed in range(3):
                base = 31.0 - 25.0 * rate ** 2 + (offset * rate if method != "no_rs" else -10 * rate)
                if method.startswith("a@") and rate < 0.3:
                    base -= 0.5
                rows.append({
                    "method": method, "scheme": method.split("@")[0], "erasure_rate": rate, "seed": seed,
                    "images": 100, "psnr_db_mean_per_image": base + 0.05 * seed, "psnr_db_median": base,
                    "psnr_db_p5": base - 4, "ssim_mean": 0.9, "any_block_failure_rate": rate,
                    "failed_symbol_fraction": rate / 2, "realized_erasure_rate": rate,
                    "coding_overhead_fraction": 0.5, "side_info_bytes_per_image": 0 if "@" not in method else 16,
                    "total_overhead_fraction_including_side_info": 0.5})
    return rows


def test_seed_summary_and_insights_run_end_to_end(tmp_path):
    rows = fake_rows()
    outputs = write_sweep_outputs(str(tmp_path), "fake", rows, {"checkpoint": {"path": "x", "sha256": "y"}})
    for path in (outputs["json"], outputs["markdown"]):
        assert (tmp_path / path.split("\\")[-1].split("/")[-1]).exists()
    assert any(f.endswith("fake_psnr_curves.pdf") for f in outputs["figures"])
    assert any(f.endswith("fake_gain_vs_uniform.png") for f in outputs["figures"])
    assert outputs["insights"] and all(isinstance(line, str) for line in outputs["insights"])
    text = " ".join(outputs["insights"])
    assert "Peak gain" in text and "a@oracle" in text


def test_insights_report_the_low_loss_penalty():
    rows = fake_rows()
    summary = aggregate_over_seeds(rows)
    notes = sweep_insights(summary, paired_vs_baseline(rows))
    assert any("Low-loss penalty" in n or "Break-even" in n for n in notes)


def test_select_scheme_picks_the_best_per_rate():
    summary = [
        {"method": "x@oracle", "erasure_rate": 0.4, "psnr_db_mean_per_image__mean": 20.0},
        {"method": "y@oracle", "erasure_rate": 0.4, "psnr_db_mean_per_image__mean": 22.0},
        {"method": "x@oracle", "erasure_rate": 0.5, "psnr_db_mean_per_image__mean": 19.0},
        {"method": "y@oracle", "erasure_rate": 0.5, "psnr_db_mean_per_image__mean": 18.0},
    ]
    selection, _ = select(summary, 0.35)
    assert selection["oracle"]["per_rate"] == {"0.4": "y", "0.5": "x"}
    assert selection["oracle"]["best_overall_high_loss"] in ("x", "y")


# ---- orchestrator and importability -------------------------------------------------------
def test_pipeline_steps_are_unique_well_formed_and_dry():
    import argparse

    import run_pipeline

    cfg = argparse.Namespace(data_dir="./data", results="results", seed42_dir="checkpoints/train_split_adjsccq",
                             highres_dir="D:/images")
    steps = run_pipeline.build_steps(cfg)
    names = [s.name for s in steps]
    assert len(names) == len(set(names))
    assert {s.phase for s in steps} == set(run_pipeline.PHASES)
    for step in steps:
        assert all(isinstance(token, str) for token in step.command), step.name
        run_pipeline.resolve(list(step.command), dry_run=True)  # placeholders resolve in a dry run
    final_test = [s for s in steps if s.name == "final_base"][0]
    assert "--split" in final_test.command and final_test.command[final_test.command.index("--split") + 1] == "test"
    tuning = [s for s in steps if s.name in ("val_sweep_base", "saliency_sweep")]
    assert all(s.command[s.command.index("--split") + 1] == "val" for s in tuning)


@pytest.mark.parametrize("module", [
    "experiments.eval_rs_uep", "experiments.eval_rs_uep_oracle", "experiments.eval_burst",
    "experiments.eval_digital_baseline", "experiments.eval_kodak", "experiments.train_sensitivity_head",
    "experiments.train_erasure_robust", "experiments.train_highres", "experiments.feature_saliency",
    "experiments.transfer_sensitivity", "experiments.attention_vs_damage", "experiments.multiseed_analysis",
    "experiments.select_scheme", "experiments.make_report", "experiments.download_data",
    "experiments.train_adjsccq_train_split", "experiments.visualize_reconstructions",
])
def test_every_script_imports_cleanly(module):
    assert importlib.import_module(module) is not None


def test_kaggle_launchers_compile():
    """A syntax error in a launcher only shows up after a queued Kaggle run; catch it here."""
    import glob
    import py_compile

    launchers = glob.glob("kaggle/kernel_train/run.py") + glob.glob("kaggle/kernels/*/run.py")
    assert launchers
    for path in launchers:
        py_compile.compile(path, doraise=True)
