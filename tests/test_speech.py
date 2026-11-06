"""Speech study S1: 1-D models, metrics, data pipeline, evaluation pipeline and orchestration.

What must hold:

* both speech architectures produce the CIFAR latent layout (16 x 64 = 1024 bytes) and
  differ only by the AF modules; the image architectures are untouched;
* the shared metrics keep the exact image code path, and the speech metric (SNR) gives
  the same leave-one-out damage and paired gains as PSNR would;
* the evaluation pipeline (importance sources, the RS cell, reporting) runs on speech,
  a loss-free transmission is lossless, and result rows carry SI-SDR instead of SSIM;
* the LibriSpeech cache is deterministic, drops silence, and loads as (N, 1, L) in [-1, 1);
* one tiny training run writes the same files and keys as the image trainer;
* the ``speech`` pipeline phase is well formed.

No download is needed: the data tests write a tiny fake LibriSpeech tree (needs ``soundfile``).
"""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from coding.scaling import scheme_catalog
from data.datasets import is_official_test, load_eval_tensor
from experiments.eval_rs_uep import run_cell
from experiments.importance_sources import PER_IMAGE_SOURCES, compute_orders
from experiments.metrics import (
    per_item_mse,
    psnr_per_image,
    quality_db_per_item,
    si_sdr_db_per_item,
    snr_db_per_item,
)
from experiments.model_io import ModelBundle, encode_full, load_model_bundle
from experiments.pipeline_core import make_context, make_method_specs, precompute
from experiments.reporting import summary_markdown
from experiments.reproducibility import aggregate_over_seeds
from experiments.sensitivity import count_macs, loo_damage_batch
from experiments.train_sensitivity_head import compute_labels
from models.adjsccq import ADJSCCQEncoder
from models.Attention import AFModule
from models.registry import ARCHITECTURES, build_jscc, model_modality

L = 16384


def speech_cfg(arch: str) -> dict:
    with open(f"configs/speech_{arch}.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def speech_bundle(arch: str) -> ModelBundle:
    torch.manual_seed(0)
    cfg = speech_cfg(arch)
    encoder, decoder = build_jscc(cfg)
    encoder.calibrate_quantizer(0.3 * torch.randn(4, 1, L))
    return ModelBundle(encoder, decoder, None, cfg, "", torch.device("cpu"), 0).eval()


@pytest.fixture(scope="module")
def bundles():
    return {arch: speech_bundle(arch) for arch in ("adjsccq1d", "deepjsccq1d")}


@pytest.fixture(scope="module")
def clips():
    return (0.3 * torch.randn(4, 1, L, generator=torch.Generator().manual_seed(3))).clamp(-1, 1)


# ---- models ---------------------------------------------------------------------------------
@pytest.mark.parametrize("arch", ["adjsccq1d", "deepjsccq1d"])
def test_speech_model_geometry(bundles, clips, arch):
    bundle = bundles[arch]
    z_q, idx = bundle.encoder(clips, 10.0, return_indices=True)
    assert z_q.shape == (4, 1024) and idx.shape == (4, 1024)
    assert bundle.channels == 16 and bundle.symbols_per_channel == 64 and bundle.modality == "speech"
    out = bundle.decoder(z_q, 10.0)
    assert out.shape == clips.shape and float(out.abs().max()) <= 1.0
    z = bundle.encoder.encoder(clips, 10.0)
    assert torch.allclose(z.pow(2).sum(1), torch.full((4,), float(bundle.encoder.k)), rtol=1e-4)
    assert bundle.example_input(2).shape == (2, 1, L)


def test_speech_variants_differ_only_by_attention(bundles, clips):
    att, plain = bundles["adjsccq1d"], bundles["deepjsccq1d"]
    _, attn = att.encoder.encoder(clips, 10.0, return_attn=True)
    assert attn["af5_bottleneck"].shape == (4, 16) and att.has_attention
    _, attn = plain.encoder.encoder(clips, 10.0, return_attn=True)
    assert attn == {} and not plain.has_attention
    assert not any(isinstance(m, AFModule) for m in list(plain.encoder.modules()) + list(plain.decoder.modules()))
    shapes = lambda m, kind: [tuple(x.weight.shape) for x in m.modules() if type(x) is kind]  # noqa: E731
    for kind in (nn.Conv1d, nn.ConvTranspose1d):
        assert shapes(att.encoder, kind) == shapes(plain.encoder, kind)
        assert shapes(att.decoder, kind) == shapes(plain.decoder, kind)
    count = lambda *ms: sum(p.numel() for m in ms for p in m.parameters())  # noqa: E731
    af = sum(p.numel() for m in list(att.encoder.modules()) + list(att.decoder.modules()) if isinstance(m, AFModule)
             for p in m.parameters())
    assert count(att.encoder, att.decoder) - count(plain.encoder, plain.decoder) == af


def test_registry_modalities_and_image_archs_untouched():
    assert model_modality({"model": {"arch": "adjsccq1d"}}) == "speech"
    assert model_modality({"model": {}}) == "image"
    assert ARCHITECTURES["adjsccq"][0] is ADJSCCQEncoder
    assert {a for a, s in ARCHITECTURES.items() if s.modality == "speech"} == {"adjsccq1d", "deepjsccq1d"}


@pytest.mark.parametrize("arch", ["adjsccq1d", "deepjsccq1d"])
def test_speech_checkpoint_round_trip(tmp_path, bundles, clips, arch):
    source = bundles[arch]
    path = os.path.join(tmp_path, "best_val.pth")
    torch.save({"encoder": source.encoder.state_dict(), "decoder": source.decoder.state_dict(),
                "config": source.config, "split": {"seed": 0}}, path)
    loaded = load_model_bundle(path, "cpu")
    assert loaded.arch == arch and loaded.modality == "speech"
    with torch.no_grad():
        assert torch.equal(source.decoder(source.encoder(clips, 10.0), 10.0),
                           loaded.decoder(loaded.encoder(clips, 10.0), 10.0))


# ---- metrics ----------------------------------------------------------------------------------
def test_image_metric_path_is_unchanged():
    gen = torch.Generator().manual_seed(0)
    x, y = torch.rand(3, 3, 32, 32, generator=gen), torch.rand(3, 3, 32, 32, generator=gen)
    assert torch.equal(quality_db_per_item(y, x), psnr_per_image(y, x))
    assert torch.equal(per_item_mse(y, x), ((y - x) ** 2).mean(dim=[1, 2, 3]))


def test_snr_damage_equals_psnr_damage_and_si_sdr_is_scale_invariant(clips):
    gen = torch.Generator().manual_seed(1)
    a = clips + 0.05 * torch.randn(clips.shape, generator=gen)
    b = clips + 0.20 * torch.randn(clips.shape, generator=gen)
    snr_gap = snr_db_per_item(a, clips) - snr_db_per_item(b, clips)
    mse_gap = 10 * torch.log10(per_item_mse(b, clips) / per_item_mse(a, clips))
    assert torch.allclose(snr_gap, mse_gap, atol=1e-4)   # the reference power cancels (AD-16)
    assert torch.equal(quality_db_per_item(a, clips), snr_db_per_item(a, clips))
    assert torch.allclose(si_sdr_db_per_item(3.0 * a, clips), si_sdr_db_per_item(a, clips), atol=1e-4)


# ---- evaluation pipeline ------------------------------------------------------------------------
def test_importance_sources_on_speech(bundles, clips):
    sources = [s for s in PER_IMAGE_SOURCES if s != "predicted"]
    orders = compute_orders(bundles["adjsccq1d"], clips, sources + ["static:oracle"], 10.0,
                            rankings={"oracle": list(range(16))})
    for source, order in orders.items():
        assert order.shape == (4, 16), source
        assert bool((order.sort(dim=1).values == torch.arange(16)).all()), source
    with pytest.raises(ValueError, match="attention"):
        compute_orders(bundles["deepjsccq1d"], clips, ["attention"], 10.0)


def test_loo_damage_and_mac_count_on_speech(bundles, clips):
    bundle = bundles["deepjsccq1d"]
    info = encode_full(bundle.encoder, clips, 10.0)
    damage, clean = loo_damage_batch(bundle, bundle.encoder.quantizer.dequantize_indices(info["idx"]), clips, 10.0)
    assert damage.shape == (4, 16) and clean.shape == (4,)
    macs = count_macs(bundle.encoder.encoder, lambda: bundle.encoder.encoder(bundle.example_input(), 10.0))
    assert macs > 1_000_000   # Conv1d layers are counted
    labels = compute_labels(bundle, clips, batch_size=2)
    assert labels["damage"].shape == (4, 16) and "attention" not in labels


@pytest.mark.parametrize("arch", ["adjsccq1d", "deepjsccq1d"])
def test_speech_cell_is_lossless_at_zero_loss_and_reports_si_sdr(bundles, clips, arch):
    bundle = bundles[arch]
    specs = make_method_specs(["three_tier_25_50_25"], ["static:oracle", "energy"], include_no_rs=True,
                              include_random=True)
    prep = precompute(bundle, clips, ["static:oracle", "energy"], 2, {"oracle": list(range(16))})
    catalog = scheme_catalog(16, 64, 16, 0.5, names=sorted({s.scheme for s in specs}))
    context = make_context(bundle, clips, prep, catalog, 2)
    context.specs, context.master_n = specs, max(s.total_packets for s in catalog.values())
    context.args = {"dataset": "librispeech", "split": "val", "checkpoint": "tiny"}
    rows, per_image = run_cell(context, (0, 0.0))
    for row in rows:
        assert np.allclose(per_image[f"{row['method']}__s0__r0.000"], prep["clean_psnr"], atol=1e-3), row["method"]
        assert "si_sdr_db_mean" in row and "ssim_mean" not in row
    lossy, _ = run_cell(context, (1, 0.4))
    summary = aggregate_over_seeds(rows + lossy, metrics=("psnr_db_mean_per_image", "si_sdr_db_mean"))
    table = summary_markdown(summary)
    assert "SNR (dB)" in table and "SI-SDR (dB)" in table


def test_librispeech_test_split_is_logged():
    assert is_official_test("librispeech", "test") and not is_official_test("librispeech", "val")
    assert is_official_test("cifar10", "test") and not is_official_test("cifar100", "test")


# ---- data pipeline --------------------------------------------------------------------------------
def test_candidate_offsets_drop_silence():
    from data.speech import candidate_offsets

    loud = (np.sin(np.arange(L) / 5.0) * 8000).astype(np.int16)
    audio = np.concatenate([np.zeros(L, np.int16), loud, np.full(L, 3, np.int16), loud[: L // 2]])
    offsets = [offset for offset, _ in candidate_offsets(audio)]
    assert offsets == [L]   # only the loud full segment; silence, near-silence and the tail are dropped


def _fake_librispeech(data_dir: str):
    soundfile = pytest.importorskip("soundfile")
    rng = np.random.default_rng(0)
    for subset in ("train-clean-100", "dev-clean", "test-clean"):
        for speaker in (11, 22):
            folder = os.path.join(data_dir, "LibriSpeech", subset, str(speaker), "100")
            os.makedirs(folder, exist_ok=True)
            for utt in range(2):
                audio = (rng.standard_normal(int(2.5 * L)) * 3000).astype(np.int16)
                soundfile.write(os.path.join(folder, f"{speaker}-100-{utt:04d}.flac"), audio, 16000, subtype="PCM_16")
            soundfile.write(os.path.join(folder, f"{speaker}-100-0009.flac"), np.zeros(2 * L, np.int16), 16000,
                            subtype="PCM_16")  # silent utterance: contributes nothing


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    from data.speech import build_split

    data_dir = str(tmp_path_factory.mktemp("speech_data"))
    _fake_librispeech(data_dir)
    manifests = {split: build_split(data_dir, split, cap=6) for split in ("train", "val", "test")}
    return data_dir, manifests


def test_segment_cache_is_deterministic_and_drops_silence(fake_data, tmp_path):
    from data.speech import build_split, load_speech_tensor, SpeechSegments

    data_dir, manifests = fake_data
    manifest = manifests["val"]
    assert manifest["num_segments"] == 6 and manifest["pool_size"] == 8 and manifest["speakers"] == 2
    assert all(not seg["utterance"].endswith("0009") for seg in manifest["segments"])
    again = os.path.join(tmp_path, "again")
    _fake_librispeech(again)
    assert build_split(again, "val", cap=6)["array_sha256"] == manifest["array_sha256"]
    items = load_speech_tensor("val", data_dir)
    assert items.shape == (6, 1, L) and items.dtype == torch.float32 and float(items.abs().max()) < 1.0
    assert torch.equal(load_eval_tensor("librispeech", "val", data_dir, 4), items[:4])
    dataset = SpeechSegments(data_dir, "val")
    assert len(dataset) == 6 and torch.equal(dataset[2][0], items[2])


def test_speech_loaders(fake_data):
    from data.speech import make_speech_loaders

    train, val, test = make_speech_loaders(fake_data[0], batch_size=2, num_workers=0, loader_seed=1)
    x, label = next(iter(train))
    assert x.shape == (2, 1, L) and int(label[0]) == 0
    assert len(val.dataset) == len(test.dataset) == 6


def test_one_epoch_training_writes_the_image_trainer_outputs(fake_data, tmp_path, monkeypatch):
    import experiments.train_speech_jscc as trainer

    monkeypatch.setattr(trainer, "log_test_access", lambda *a, **k: None)   # keep the real audit log clean
    cfg = speech_cfg("deepjsccq1d")
    cfg["data"].update(batch_size=2, num_workers=0)
    cfg["train"]["epochs"] = 1
    config = os.path.join(tmp_path, "cfg.yaml")
    with open(config, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
    out = os.path.join(tmp_path, "run")
    trainer.main(trainer.build_parser().parse_args(
        ["--config", config, "--data_dir", fake_data[0], "--output_dir", out, "--device", "cpu", "--seed", "3"]))
    with open(os.path.join(out, "metrics.json"), encoding="utf-8") as handle:
        metrics = json.load(handle)
    for key in ("best_epoch", "clean_validation", "clean_test", "oracle", "arch", "train_device", "quality_metric"):
        assert key in metrics, key
    with open(os.path.join(out, "oracle_ranking.json"), encoding="utf-8") as handle:
        oracle = json.load(handle)
    assert sorted(oracle["ranking_channels_most_to_least_important"]) == list(range(16))
    assert oracle["source_split"] == "train" and len(oracle["psnr_drop_db_by_channel"]) == 16
    assert load_model_bundle(os.path.join(out, "best_val.pth"), "cpu").arch == "deepjsccq1d"


# ---- perceptual script and orchestrator ------------------------------------------------------------------
def test_perceptual_method_specs(tmp_path):
    from experiments.eval_speech_perceptual import method_specs

    path = os.path.join(tmp_path, "selected.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"selected_on": "val", "selection": {"oracle": {"best_overall_high_loss": "three_tier_25_50_25"},
                                                       "energy": {"best_overall_high_loss": "two_tier_50_50"}}}, handle)
    names = [s.name for s in method_specs(path, ["oracle", "energy"], [])]
    assert names == ["no_rs", "uniform", "three_tier_25_50_25@oracle", "two_tier_50_50@energy"]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"selected_on": "test", "selection": {}}, handle)
    with pytest.raises(SystemExit):
        method_specs(path, ["oracle"], [])


def test_speech_pipeline_phase_is_well_formed():
    import argparse

    import run_pipeline

    cfg = argparse.Namespace(data_dir="./data", results="results", seed42_dir="checkpoints/train_split_adjsccq",
                             highres_dir=None, train_device="mps")
    steps = [s for s in run_pipeline.build_steps(cfg) if s.phase == "speech"]
    names = {s.name for s in steps}
    for arch in run_pipeline.SPEECH_ARCHS:
        assert {f"speech_train_{arch}_seed{seed}" for seed in run_pipeline.SPEECH_SEEDS} <= names
        assert {f"speech_select_{arch}", f"speech_final_{arch}_seed42", f"speech_perceptual_{arch}"} <= names
    assert {"speech_data", "speech_compare", "speech_attention_vs_damage"} <= names
    for step in steps:
        command = run_pipeline.resolve(list(step.command), dry_run=True)
        if "experiments.train_speech_jscc" in command:
            arch = command[command.index("--arch") + 1]
            assert command[command.index("--config") + 1] == f"configs/speech_{arch}.yaml"
            assert command[command.index("--device") + 1] == "mps"
        if "experiments.eval_rs_uep" in command:
            assert command[command.index("--dataset") + 1] == "librispeech"
            split = command[command.index("--split") + 1]
            assert split == ("test" if "final" in step.name else "val"), step.name
            if "deepjsccq1d" in step.name:
                assert "attention" not in command[command.index("--importance") + 1]
    assert "speech" in run_pipeline.EXTENSION_PHASES and "speech" not in run_pipeline.PHASES
