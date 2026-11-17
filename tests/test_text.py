"""Text study T1: character-level models, metrics, data pipeline, evaluation pipeline and orchestration.

What must hold:

* both text architectures produce the CIFAR / speech latent layout (16 x 64 = 1024 bytes), return
  per-character logits, and differ only by the AF modules; the image and speech archs are untouched;
* the shared metrics keep the exact image and speech code paths; the text metric (character
  accuracy) and word accuracy are correct on hand-made cases; the Taylor loss is MSE for
  images / speech and cross-entropy for text;
* the evaluation pipeline (importance sources, LOO, the RS cell, reporting) runs on text, a
  loss-free transmission is lossless, and result rows carry word accuracy instead of SSIM;
* the text8 window cache is deterministic, respects the split ranges and loads as int64 (N, L);
* one tiny training run writes the same files and keys as the image / speech trainers;
* the ``text`` pipeline phase is well formed.

No download is needed: the data tests build a tiny synthetic corpus in the text8 alphabet.
"""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from coding.scaling import scheme_catalog
from data.datasets import is_official_test, load_eval_tensor, tensor_modality
from experiments.eval_rs_uep import run_cell
from experiments.importance_sources import PER_IMAGE_SOURCES, compute_orders
from experiments.metrics import (
    bits_per_char_per_item,
    char_accuracy_per_item,
    per_item_loss,
    per_item_mse,
    psnr_per_image,
    quality_db_per_item,
    snr_db_per_item,
    word_accuracy_per_item,
)
from experiments.model_io import ModelBundle, encode_full, load_model_bundle
from experiments.pipeline_core import make_context, make_method_specs, precompute
from experiments.reporting import summary_markdown
from experiments.reproducibility import aggregate_over_seeds
from experiments.sensitivity import count_macs, loo_damage_batch
from experiments.train_sensitivity_head import compute_labels
from models.Attention import AFModule
from models.registry import ARCHITECTURES, build_jscc, model_modality
from models.speech_jscc import SpeechJSCCQEncoder

L = 2048
V = 27
ARCHS = ("adjsccq_text", "deepjsccq_text")
CONFIGS = {"adjsccq_text": "configs/text_adjsccq.yaml", "deepjsccq_text": "configs/text_deepjsccq.yaml"}


def text_cfg(arch: str) -> dict:
    with open(CONFIGS[arch], encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def text_bundle(arch: str) -> ModelBundle:
    torch.manual_seed(0)
    cfg = text_cfg(arch)
    encoder, decoder = build_jscc(cfg)
    encoder.calibrate_quantizer(torch.randint(0, V, (4, L), generator=torch.Generator().manual_seed(1)))
    return ModelBundle(encoder, decoder, None, cfg, "", torch.device("cpu"), 0).eval()


def one_hot_logits(ids: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    """Logits whose argmax is ``ids``: (B, L) -> (B, V, L)."""
    return scale * nn.functional.one_hot(ids.long(), V).permute(0, 2, 1).float()


@pytest.fixture(scope="module")
def bundles():
    return {arch: text_bundle(arch) for arch in ARCHS}


@pytest.fixture(scope="module")
def windows():
    return torch.randint(0, V, (4, L), generator=torch.Generator().manual_seed(3))


# ---- models ---------------------------------------------------------------------------------
@pytest.mark.parametrize("arch", ARCHS)
def test_text_model_geometry(bundles, windows, arch):
    bundle = bundles[arch]
    z_q, idx = bundle.encoder(windows, 10.0, return_indices=True)
    assert z_q.shape == (4, 1024) and idx.shape == (4, 1024)
    assert bundle.channels == 16 and bundle.symbols_per_channel == 64 and bundle.modality == "text"
    logits = bundle.decoder(z_q, 10.0)
    assert logits.shape == (4, V, L)
    z = bundle.encoder.encoder(windows, 10.0)
    assert torch.allclose(z.pow(2).sum(1), torch.full((4,), float(bundle.encoder.k)), rtol=1e-4)
    example = bundle.example_input(2)
    assert example.shape == (2, L) and example.dtype == torch.long
    assert bundle.encoder(windows.to(torch.uint8), 10.0).shape == (4, 1024)   # any integer dtype


def test_text_geometry_is_validated():
    from models.text_jscc import TextJSCCEncoder

    with pytest.raises(ValueError):
        TextJSCCEncoder(segment_length=2000)          # not a multiple of 32
    with pytest.raises(ValueError):
        TextJSCCEncoder(strides=(4, 2, 2, 3))         # odd stride
    encoder = TextJSCCEncoder(segment_length=4096, strides=(4, 4, 2, 2))   # a rate-sweep geometry: 4096 / 64
    assert encoder.latent_steps == 64


def test_text_variants_differ_only_by_attention(bundles, windows):
    att, plain = bundles["adjsccq_text"], bundles["deepjsccq_text"]
    _, attn = att.encoder.encoder(windows, 10.0, return_attn=True)
    assert attn["af5_bottleneck"].shape == (4, 16) and att.has_attention
    _, attn = plain.encoder.encoder(windows, 10.0, return_attn=True)
    assert attn == {} and not plain.has_attention
    assert not any(isinstance(m, AFModule) for m in list(plain.encoder.modules()) + list(plain.decoder.modules()))
    shapes = lambda m, kind: [tuple(x.weight.shape) for x in m.modules() if type(x) is kind]  # noqa: E731
    for kind in (nn.Conv1d, nn.ConvTranspose1d, nn.Embedding):
        assert shapes(att.encoder, kind) == shapes(plain.encoder, kind)
        assert shapes(att.decoder, kind) == shapes(plain.decoder, kind)
    count = lambda *ms: sum(p.numel() for m in ms for p in m.parameters())  # noqa: E731
    af = sum(p.numel() for m in list(att.encoder.modules()) + list(att.decoder.modules()) if isinstance(m, AFModule)
             for p in m.parameters())
    assert count(att.encoder, att.decoder) - count(plain.encoder, plain.decoder) == af


def test_registry_modalities_and_other_archs_untouched():
    assert model_modality({"model": {"arch": "adjsccq_text"}}) == "text"
    assert model_modality({"model": {"arch": "deepjsccq1d"}}) == "speech"
    assert model_modality({"model": {}}) == "image"
    assert ARCHITECTURES["adjsccq1d"][0] is SpeechJSCCQEncoder
    assert {a for a, s in ARCHITECTURES.items() if s.modality == "text"} == set(ARCHS)


@pytest.mark.parametrize("arch", ARCHS)
def test_text_checkpoint_round_trip(tmp_path, bundles, windows, arch):
    source = bundles[arch]
    path = os.path.join(tmp_path, "best_val.pth")
    torch.save({"encoder": source.encoder.state_dict(), "decoder": source.decoder.state_dict(),
                "config": source.config, "split": {"seed": 0}}, path)
    loaded = load_model_bundle(path, "cpu")
    assert loaded.arch == arch and loaded.modality == "text"
    with torch.no_grad():
        assert torch.equal(source.decoder(source.encoder(windows, 10.0), 10.0),
                           loaded.decoder(loaded.encoder(windows, 10.0), 10.0))


# ---- metrics ----------------------------------------------------------------------------------
def test_image_and_speech_metric_paths_are_unchanged():
    gen = torch.Generator().manual_seed(0)
    x, y = torch.rand(3, 3, 32, 32, generator=gen), torch.rand(3, 3, 32, 32, generator=gen)
    assert torch.equal(quality_db_per_item(y, x), psnr_per_image(y, x))
    assert torch.equal(per_item_loss(y, x), per_item_mse(y, x))
    a, b = torch.rand(2, 1, 256, generator=gen), torch.rand(2, 1, 256, generator=gen)
    assert torch.equal(quality_db_per_item(b, a), snr_db_per_item(b, a))
    assert torch.equal(per_item_loss(b, a), per_item_mse(b, a))


def test_character_and_word_accuracy_on_hand_made_cases():
    from data.text import encode_text

    reference = torch.from_numpy(encode_text("the cat sat on mat").astype(np.int64))[None]
    perfect = one_hot_logits(reference)
    assert float(char_accuracy_per_item(perfect, reference)) == 100.0
    assert float(word_accuracy_per_item(perfect, reference)) == 100.0
    assert float(quality_db_per_item(perfect, reference)) == 100.0   # text goes to the accuracy branch
    wrong = reference.clone()
    wrong[0, 5] = int(encode_text("x")[0])     # "the cxt sat on mat": 1 of 18 chars, 1 of 5 words
    logits = one_hot_logits(wrong)
    assert torch.allclose(char_accuracy_per_item(logits, reference), torch.tensor([100.0 * 17 / 18]))
    assert torch.allclose(word_accuracy_per_item(logits, reference), torch.tensor([80.0]))
    spaced = reference.clone()
    spaced[0, 3] = int(encode_text("a")[0])    # a wrong space: costs a character, no word is mis-spelled
    assert torch.allclose(word_accuracy_per_item(one_hot_logits(spaced), reference), torch.tensor([100.0]))
    assert float(bits_per_char_per_item(one_hot_logits(reference, 50.0), reference)) < 1e-3


def test_taylor_loss_for_text_is_cross_entropy(windows):
    logits = torch.randn(4, V, L, generator=torch.Generator().manual_seed(2))
    expected = nn.functional.cross_entropy(logits, windows, reduction="none").mean(dim=1)
    assert torch.allclose(per_item_loss(logits, windows), expected)


def test_tensor_modality_and_test_split_logging(windows):
    assert tensor_modality(windows) == "text"
    assert tensor_modality(torch.zeros(2, 1, 64)) == "speech"
    assert tensor_modality(torch.zeros(2, 3, 32, 32)) == "image"
    assert is_official_test("text8", "test") and not is_official_test("text8", "val")


# ---- evaluation pipeline ------------------------------------------------------------------------
def test_importance_sources_on_text(bundles, windows):
    sources = [s for s in PER_IMAGE_SOURCES if s != "predicted"]
    orders = compute_orders(bundles["adjsccq_text"], windows, sources + ["static:oracle"], 10.0,
                            rankings={"oracle": list(range(16))})
    for source, order in orders.items():
        assert order.shape == (4, 16), source
        assert bool((order.sort(dim=1).values == torch.arange(16)).all()), source
    with pytest.raises(ValueError, match="attention"):
        compute_orders(bundles["deepjsccq_text"], windows, ["attention"], 10.0)


def test_loo_damage_and_mac_count_on_text(bundles, windows):
    bundle = bundles["deepjsccq_text"]
    info = encode_full(bundle.encoder, windows, 10.0)
    damage, clean = loo_damage_batch(bundle, bundle.encoder.quantizer.dequantize_indices(info["idx"]), windows, 10.0)
    assert damage.shape == (4, 16) and clean.shape == (4,)
    assert bool(((clean >= 0) & (clean <= 100)).all())
    macs = count_macs(bundle.encoder.encoder, lambda: bundle.encoder.encoder(bundle.example_input(), 10.0))
    assert macs > 1_000_000   # Conv1d layers are counted
    labels = compute_labels(bundle, windows, batch_size=2)
    assert labels["damage"].shape == (4, 16) and "attention" not in labels


@pytest.mark.parametrize("arch", ARCHS)
def test_text_cell_is_lossless_at_zero_loss_and_reports_word_accuracy(bundles, windows, arch):
    bundle = bundles[arch]
    specs = make_method_specs(["three_tier_25_50_25"], ["static:oracle", "energy"], include_no_rs=True,
                              include_random=True)
    prep = precompute(bundle, windows, ["static:oracle", "energy"], 2, {"oracle": list(range(16))})
    catalog = scheme_catalog(16, 64, 16, 0.5, names=sorted({s.scheme for s in specs}))
    context = make_context(bundle, windows, prep, catalog, 2)
    context.specs, context.master_n = specs, max(s.total_packets for s in catalog.values())
    context.args = {"dataset": "text8", "split": "val", "checkpoint": "tiny"}
    rows, per_image = run_cell(context, (0, 0.0))
    for row in rows:
        assert np.allclose(per_image[f"{row['method']}__s0__r0.000"], prep["clean_psnr"], atol=1e-3), row["method"]
        assert "word_acc_mean" in row and "ssim_mean" not in row and "si_sdr_db_mean" not in row
    lossy, _ = run_cell(context, (1, 0.4))
    summary = aggregate_over_seeds(rows + lossy, metrics=("psnr_db_mean_per_image", "word_acc_mean"))
    table = summary_markdown(summary)
    assert "Char acc (%)" in table and "Word acc (%)" in table


# ---- data pipeline --------------------------------------------------------------------------------
TINY_RANGES = {"train": (0, 9 * 4096), "val": (9 * 4096, 10 * 4096 + 100), "test": (10 * 4096 + 100, 12 * 4096)}


def _tiny_corpus() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, V, size=12 * 4096, dtype=np.uint8)


@pytest.fixture(scope="module")
def fake_data(tmp_path_factory):
    from data.text import build_split

    data_dir = str(tmp_path_factory.mktemp("text_data"))
    corpus = _tiny_corpus()
    manifests = {split: build_split(data_dir, split, cap=cap, corpus=corpus, ranges=TINY_RANGES)
                 for split, cap in (("train", 12), ("val", None), ("test", None))}
    return data_dir, manifests, corpus


def test_text_codec_round_trip_and_alphabet(tmp_path):
    from data.text import decode_ids, encode_text, read_text8

    assert decode_ids(encode_text("hello world")) == "hello world"
    with pytest.raises(ValueError):
        encode_text("Hello")
    path = os.path.join(tmp_path, "text8")
    with open(path, "wb") as handle:
        handle.write(b"abc def")
    assert read_text8(path, expected_length=7).tolist() == [1, 2, 3, 0, 4, 5, 6]
    with pytest.raises(ValueError):
        read_text8(path)                         # wrong length for the real corpus
    with open(path, "wb") as handle:
        handle.write(b"abc,def")
    with pytest.raises(ValueError):
        read_text8(path, expected_length=7)      # character outside the alphabet


def test_window_cache_is_deterministic_and_respects_split_ranges(fake_data, tmp_path):
    from data.text import TextSegments, build_split, load_text_tensor

    data_dir, manifests, corpus = fake_data
    assert manifests["train"]["num_segments"] == 12 and manifests["train"]["windows_available"] == 18
    assert manifests["val"]["num_segments"] == 2 and manifests["test"]["num_segments"] == 3
    for split, manifest in manifests.items():
        start, stop = TINY_RANGES[split]
        assert all(start <= o and o + L <= stop for o in manifest["offsets"]), split
    again = build_split(os.path.join(tmp_path, "again"), "train", cap=12, corpus=corpus, ranges=TINY_RANGES)
    assert again["array_sha256"] == manifests["train"]["array_sha256"]
    items = load_text_tensor("val", data_dir)
    assert items.shape == (2, L) and items.dtype == torch.int64
    offset = manifests["val"]["offsets"][0]
    assert np.array_equal(items[0].numpy(), corpus[offset:offset + L].astype(np.int64))
    assert torch.equal(load_eval_tensor("text8", "val", data_dir, 1), items[:1])
    dataset = TextSegments(data_dir, "val")
    assert len(dataset) == 2 and torch.equal(dataset[1][0], items[1])


def test_text_loaders(fake_data):
    from data.text import make_text_loaders

    train, val, test = make_text_loaders(fake_data[0], batch_size=2, num_workers=0, loader_seed=1)
    x, label = next(iter(train))
    assert x.shape == (2, L) and x.dtype == torch.int64 and int(label[0]) == 0
    assert len(val.dataset) == 2 and len(test.dataset) == 3


def test_one_epoch_training_writes_the_image_trainer_outputs(fake_data, tmp_path, monkeypatch):
    import experiments.train_text_jscc as trainer

    monkeypatch.setattr(trainer, "log_test_access", lambda *a, **k: None)   # keep the real audit log clean
    cfg = text_cfg("deepjsccq_text")
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
    for key in ("psnr_db_mean_per_image", "char_acc_mean", "word_acc_mean", "bits_per_char"):
        assert key in metrics["clean_validation"], key
    with open(os.path.join(out, "oracle_ranking.json"), encoding="utf-8") as handle:
        oracle = json.load(handle)
    assert sorted(oracle["ranking_channels_most_to_least_important"]) == list(range(16))
    assert oracle["source_split"] == "train" and len(oracle["psnr_drop_db_by_channel"]) == 16
    assert load_model_bundle(os.path.join(out, "best_val.pth"), "cpu").arch == "deepjsccq_text"


# ---- examples script and orchestrator ------------------------------------------------------------------
def test_highlight_marks_errors():
    from data.text import encode_text
    from experiments.eval_text_examples import highlight

    reference = encode_text("ab cd")
    predicted = encode_text("ax  d")
    assert highlight(reference, predicted) == "aX _d"


def test_text_pipeline_phase_is_well_formed():
    import argparse

    import run_pipeline

    cfg = argparse.Namespace(data_dir="./data", results="results", seed42_dir="checkpoints/train_split_adjsccq",
                             highres_dir=None, train_device="mps")
    steps = [s for s in run_pipeline.build_steps(cfg) if s.phase == "text"]
    names = {s.name for s in steps}
    for arch in run_pipeline.TEXT_ARCHS:
        assert {f"text_train_{arch}_seed{seed}" for seed in run_pipeline.TEXT_SEEDS} <= names
        assert {f"text_select_{arch}", f"text_final_{arch}_seed42", f"text_examples_{arch}"} <= names
    assert {"text_data", "text_smoke_train", "text_compare", "text_attention_vs_damage"} <= names
    for step in steps:
        command = run_pipeline.resolve(list(step.command), dry_run=True)
        if "experiments.train_text_jscc" in command:
            arch = command[command.index("--arch") + 1]
            assert command[command.index("--config") + 1] == run_pipeline.TEXT_CONFIGS[arch]
            assert command[command.index("--device") + 1] == "mps"
        if "experiments.eval_rs_uep" in command:
            assert command[command.index("--dataset") + 1] == "text8"
            split = command[command.index("--split") + 1]
            assert split == ("test" if "final" in step.name else "val"), step.name
            if "deepjsccq_text" in step.name:
                assert "attention" not in command[command.index("--importance") + 1]
        if "experiments.compare_architectures" in command:
            assert command[command.index("--min_gain_db") + 1] == "2.0"
    assert "text" in run_pipeline.EXTENSION_PHASES and "text" not in run_pipeline.PHASES
