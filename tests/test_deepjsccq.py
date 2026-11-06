"""G0 (model-agnostic interface) and G1 (DeepJSCC-Q attention ablation).

What must hold:

* DeepJSCC-Q is ADJSCC-Q minus the AF modules: identical conv / PReLU stack, same
  latent shape, same quantiser, and the same call signature;
* G0 is backward compatible: a config or checkpoint without ``model.arch`` is ADJSCC-Q,
  and the registry builds ADJSCC-Q with bit-identical initialisation to the pre-G0 code;
* the evaluation pipeline runs on DeepJSCC-Q (every importance source except
  ``attention``, which fails with a clear message), and a loss-free transmission is lossless;
* the comparison script applies the pre-registered decision rule correctly;
* the g1 pipeline phase is well formed and not part of ``--phase all``.

No data or checkpoints are needed: everything uses tiny random models.
"""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from coding.scaling import scheme_catalog
from experiments import compare_architectures as compare
from experiments.eval_rs_uep import run_cell
from experiments.importance_sources import PER_IMAGE_SOURCES, compute_orders
from experiments.model_io import ModelBundle, encode_full, load_model_bundle, reconstruct
from experiments.pipeline_core import make_context, make_method_specs, precompute
from experiments.train_adjsccq_train_split import resolve_device
from experiments.train_sensitivity_head import baseline_scores, compute_labels
from models.adjsccq import ADJSCCQDecoder, ADJSCCQEncoder
from models.Attention import AFModule
from models.deepjsccq import DeepJSCCQDecoder, DeepJSCCQEncoder
from models.registry import ARCHITECTURES, DEFAULT_ARCH, build_jscc, model_arch

CFG = {"model": {"k_over_n": 1 / 6, "image_size": 32},
       "quantizer": {"num_levels": 256, "init_range": [-3.0, 3.0], "sigma_start": 1.0},
       "af_conditioning": {"nominal_snr_db": 10.0}}


def cfg_for(arch=None):
    cfg = json.loads(json.dumps(CFG))
    if arch is not None:
        cfg["model"]["arch"] = arch
    return cfg


def tiny_bundle(arch: str) -> ModelBundle:
    torch.manual_seed(0)
    encoder, decoder = build_jscc(cfg_for(arch))
    encoder.calibrate_quantizer(torch.rand(8, 3, 32, 32))
    return ModelBundle(encoder, decoder, None, cfg_for(arch), "", torch.device("cpu")).eval()


@pytest.fixture(scope="module")
def deep_bundle():
    return tiny_bundle("deepjsccq")


@pytest.fixture(scope="module")
def images():
    return torch.rand(6, 3, 32, 32, generator=torch.Generator().manual_seed(1))


def layer_shapes(module: nn.Module, kind):
    return [tuple(m.weight.shape) for m in module.modules() if type(m) is kind]


# ---- the model itself -------------------------------------------------------------------
def test_deepjsccq_is_adjsccq_without_attention():
    adj_enc, adj_dec = build_jscc(cfg_for("adjsccq"))
    deep_enc, deep_dec = build_jscc(cfg_for("deepjsccq"))
    assert layer_shapes(deep_enc, nn.Conv2d) == layer_shapes(adj_enc, nn.Conv2d)
    assert layer_shapes(deep_dec, nn.ConvTranspose2d) == layer_shapes(adj_dec, nn.ConvTranspose2d)
    for deep, adj in ((deep_enc, adj_enc), (deep_dec, adj_dec)):
        assert sum(isinstance(m, nn.PReLU) for m in deep.modules()) == sum(isinstance(m, nn.PReLU) for m in adj.modules())
        assert not any(isinstance(m, AFModule) for m in deep.modules())
    count = lambda *modules: sum(p.numel() for m in modules for p in m.parameters())  # noqa: E731
    af = sum(p.numel() for m in list(adj_enc.modules()) + list(adj_dec.modules()) if isinstance(m, AFModule)
             for p in m.parameters())
    assert count(adj_enc, adj_dec) - count(deep_enc, deep_dec) == af
    assert deep_enc.quantizer.centers.numel() == adj_enc.quantizer.centers.numel() == 256


def test_latent_shape_channels_and_range(deep_bundle, images):
    encoder, decoder = deep_bundle.encoder, deep_bundle.decoder
    z_q, idx = encoder(images, 10.0, return_indices=True)
    assert z_q.shape == (6, 1024) and idx.shape == (6, 1024)
    assert int(idx.min()) >= 0 and int(idx.max()) < 256
    assert encoder.num_channels == deep_bundle.channels == 16 and deep_bundle.symbols_per_channel == 64
    x_hat = decoder(z_q, 10.0)
    assert x_hat.shape == images.shape and float(x_hat.min()) >= 0.0 and float(x_hat.max()) <= 1.0


def test_interface_matches_adjsccq_and_ignores_snr(deep_bundle, images):
    encoder, decoder = deep_bundle.encoder, deep_bundle.decoder
    z_q, attn, idx = encoder(images, 10.0, return_attn=True, return_indices=True)
    assert attn == {} and idx.dtype == torch.long
    z, attn = encoder.encoder(images, 10.0, return_attn=True)
    assert attn == {} and z.shape == (6, 1024)
    assert torch.equal(encoder(images, 0.0), encoder(images, 20.0))
    assert torch.equal(decoder(z_q, 0.0), decoder(z_q, 20.0))
    out, attn = decoder(z_q, 10.0, return_attn=True)
    assert attn == {} and out.shape == images.shape
    assert decoder(torch.randn(1, 16 * 16 * 24), 10.0, spatial_hw=(16, 24)).shape == (1, 3, 64, 96)
    info = encode_full(encoder, images, 10.0)
    assert info["attn"] == {} and torch.equal(info["z_q"], z_q)


def test_power_constraint_and_gradients():
    torch.manual_seed(0)
    encoder, decoder = build_jscc(cfg_for("deepjsccq"))
    x = torch.rand(4, 3, 32, 32)
    z = encoder.encoder(x)
    assert torch.allclose(z.pow(2).sum(1), torch.full((4,), float(encoder.k)), rtol=1e-4)
    ((decoder(encoder(x)) - x) ** 2).mean().backward()
    assert encoder.encoder.net[0].weight.grad.abs().sum() > 0
    assert encoder.quantizer.centers.grad is not None


# ---- G0 backward compatibility -------------------------------------------------------------
def test_registry_default_and_unknown_arch():
    assert DEFAULT_ARCH == "adjsccq" and set(ARCHITECTURES) >= {"adjsccq", "deepjsccq"}
    assert model_arch({"model": {"k_over_n": 1 / 6}}) == "adjsccq"
    assert model_arch(cfg_for("deepjsccq")) == "deepjsccq"
    with pytest.raises(ValueError):
        model_arch(cfg_for("bogus"))


def test_registry_builds_adjsccq_exactly_like_pre_g0_code():
    torch.manual_seed(123)
    old_encoder = ADJSCCQEncoder(k_over_n=1 / 6, image_size=32, num_levels=256, init_range=(-3.0, 3.0),
                                 init_sigma=1.0)
    old_decoder = ADJSCCQDecoder(k=old_encoder.k, image_size=32)
    torch.manual_seed(123)
    new_encoder, new_decoder = build_jscc(cfg_for(None))
    for old, new in ((old_encoder, new_encoder), (old_decoder, new_decoder)):
        old_state, new_state = old.state_dict(), new.state_dict()
        assert old_state.keys() == new_state.keys()
        assert all(torch.equal(old_state[k], new_state[k]) for k in old_state)


@pytest.mark.parametrize("arch", [None, "adjsccq", "deepjsccq"])
def test_checkpoint_round_trip(tmp_path, images, arch):
    """``arch=None`` is a pre-G0 checkpoint (no model.arch key): it must load as ADJSCC-Q."""
    source = tiny_bundle(arch or "adjsccq")
    path = os.path.join(tmp_path, "best_val.pth")
    torch.save({"encoder": source.encoder.state_dict(), "decoder": source.decoder.state_dict(),
                "config": cfg_for(arch), "split": {"seed": 42}}, path)
    loaded = load_model_bundle(path, "cpu")
    assert loaded.arch == (arch or "adjsccq")
    assert loaded.has_attention == (loaded.arch == "adjsccq")
    assert isinstance(loaded.encoder, ARCHITECTURES[loaded.arch][0])
    snr = loaded.snr_db
    with torch.no_grad():
        expected = reconstruct(source, source.encoder(images, snr), None, snr)
        actual = reconstruct(loaded, loaded.encoder(images, snr), None, snr)
    assert torch.equal(expected, actual)


def test_auto_device_never_selects_mps():
    assert resolve_device("auto").type in ("cuda", "cpu")
    assert resolve_device("cpu").type == "cpu"


# ---- the evaluation pipeline on DeepJSCC-Q -------------------------------------------------
def test_every_non_attention_source_works_and_attention_fails_clearly(deep_bundle, images):
    sources = [s for s in PER_IMAGE_SOURCES if s not in ("attention", "predicted")]
    orders = compute_orders(deep_bundle, images, sources + ["static:oracle"], 10.0,
                            rankings={"oracle": list(range(16))})
    for source, order in orders.items():
        assert order.shape == (6, 16), source
        assert bool((order.sort(dim=1).values == torch.arange(16)).all()), source
    with pytest.raises(ValueError, match="attention"):
        compute_orders(deep_bundle, images, ["attention"], 10.0)


def test_loss_free_transmission_is_lossless_on_deepjsccq(deep_bundle, images):
    specs = make_method_specs(["three_tier_25_50_25", "two_tier_50_50"], ["static:oracle", "energy"],
                              include_no_rs=True, include_random=True)
    prep = precompute(deep_bundle, images, ["static:oracle", "energy"], 4, {"oracle": list(range(16))})
    catalog = scheme_catalog(16, 64, 16, 0.5, names=sorted({s.scheme for s in specs}))
    context = make_context(deep_bundle, images, prep, catalog, 4)
    context.specs, context.master_n = specs, max(s.total_packets for s in catalog.values())
    context.args = {"dataset": "cifar10", "split": "val", "checkpoint": "tiny"}
    rows, per_image = run_cell(context, (0, 0.0))
    for row in rows:
        assert np.allclose(per_image[f"{row['method']}__s0__r0.000"], prep["clean_psnr"], atol=1e-3), row["method"]
    lossy, _ = run_cell(context, (1, 0.45))
    assert len({r["realized_erasure_rate"] for r in lossy if r["method"] != "no_rs"}) == 1


def test_head_labels_and_baselines_without_attention(deep_bundle, images):
    labels = compute_labels(deep_bundle, images, batch_size=3)
    assert "attention" not in labels and labels["damage"].shape == (6, 16)
    scores = baseline_scores(labels, list(range(16)), seed=0)
    assert "attention" not in scores and scores["energy"].shape == (6, 16)


# ---- the comparison script ---------------------------------------------------------------
def _write_model(folder, damage, best_epoch=60, gains=None, random_gains=None, arch="adjsccq"):
    os.makedirs(folder, exist_ok=True)
    damage = np.asarray(damage, dtype=float)
    with open(os.path.join(folder, "oracle_ranking.json"), "w", encoding="utf-8") as handle:
        json.dump({"psnr_drop_db_by_channel": damage.tolist(),
                   "ranking_channels_most_to_least_important": np.argsort(-damage).tolist()}, handle)
    with open(os.path.join(folder, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"best_epoch": best_epoch, "arch": arch, "clean_test": {"psnr_db_mean_per_image": 30.0},
                   "clean_validation": {"psnr_db_mean_per_image": 30.0}}, handle)
    rows = []
    for rate, gain in (gains or {}).items():
        rows.append({"erasure_rate": rate, "method": "three_tier_25_50_25@oracle", "baseline": "uniform",
                     "mean": gain, "ci95_half_width": 0.05})
        rows.append({"erasure_rate": rate, "method": "two_tier_50_50@oracle", "baseline": "uniform",
                     "mean": gain - 0.3, "ci95_half_width": 0.05})
        rows.append({"erasure_rate": rate, "method": "three_tier_25_50_25@oracle#rr", "baseline": "uniform",
                     "mean": gain + 9.0, "ci95_half_width": 0.05})   # interleaved: must be ignored
    for rate, gain in (random_gains or {}).items():
        rows.append({"erasure_rate": rate, "method": "three_tier_25_50_25@random", "baseline": "uniform",
                     "mean": gain, "ci95_half_width": 0.05})
    path = os.path.join(folder, "sweep.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"paired_vs_baseline": rows, "seed_summary": []}, handle)
    return path


CRITERIA = {"decision_rate": 0.4, "min_top4_share": 0.40, "min_gain_db": 1.5, "max_random_fraction": 0.5,
            "flat_top4_share": 0.30, "equivalence_margin_db": 0.5}


def test_comparison_applies_the_decision_rule(tmp_path):
    concentrated = [10, 8, 7, 6] + [1] * 12            # top-4 share 31/43 = 72%
    flat = [2.0] * 16                                   # top-4 share 25%
    models, sweeps = {}, {}
    for seed in (1, 2):
        folder = os.path.join(tmp_path, f"a{seed}")
        sweeps[f"adjsccq/s{seed}"] = _write_model(folder, concentrated, gains={0.35: 1.5, 0.4: 3.0, 0.5: 2.0},
                                                  random_gains={0.35: 0.2, 0.4: 0.5, 0.5: 0.3})
        models[f"adjsccq/s{seed}"] = folder
        folder = os.path.join(tmp_path, f"d{seed}")
        sweeps[f"deepjsccq/s{seed}"] = _write_model(folder, flat, gains={0.35: 0.3, 0.4: 0.4, 0.5: 0.3},
                                                    random_gains={0.35: 0.2, 0.4: 0.3, 0.5: 0.2}, arch="deepjsccq")
        models[f"deepjsccq/s{seed}"] = folder
    folder = os.path.join(tmp_path, "a_bad")                 # not converged: reported, excluded
    sweeps["adjsccq/bad"] = _write_model(folder, flat, best_epoch=21, gains={0.4: -5.0})
    models["adjsccq/bad"] = folder

    result = compare.build_result(models, sweeps, CRITERIA, reference="adjsccq")
    adj, deep = result["families"]["adjsccq"], result["families"]["deepjsccq"]
    assert adj["summary"]["models_excluded_not_converged"] == ["adjsccq/bad"]
    assert adj["decision"]["verdict"] == "criteria met"
    assert deep["decision"]["verdict"] == "no exploitable structure"
    assert result["models"]["adjsccq/s1"]["sweep"]["best_gain"]["oracle"]["0.4"]["gain_db"] == pytest.approx(3.0)
    assert result["models"]["adjsccq/s1"]["sweep"]["best_gain"]["oracle"]["0.4"]["scheme"] == "three_tier_25_50_25"
    assert adj["summary"]["top4_share"]["mean"] == pytest.approx(31 / 43)
    delta = result["deltas"]["deepjsccq - adjsccq"]["top4_share"]["difference"]
    assert delta == pytest.approx(0.25 - 31 / 43)

    compare.write_markdown(os.path.join(tmp_path, "compare.md"), result)
    assert "criteria met" in open(os.path.join(tmp_path, "compare.md"), encoding="utf-8").read()
    assert compare.make_figures(result, os.path.join(tmp_path, "figures"))


def test_comparison_is_incomplete_without_sweeps(tmp_path):
    folder = os.path.join(tmp_path, "m")
    _write_model(folder, [5.0] * 16)
    result = compare.build_result({"deepjsccq/m": folder}, {}, CRITERIA, reference="adjsccq")
    assert result["families"]["deepjsccq"]["decision"]["verdict"] in ("incomplete", "intermediate")


# ---- orchestrator ----------------------------------------------------------------------------
def test_g1_pipeline_phase_is_well_formed():
    import argparse

    import run_pipeline

    cfg = argparse.Namespace(data_dir="./data", results="results", seed42_dir="checkpoints/train_split_adjsccq",
                             highres_dir=None, train_device="mps")
    steps = [s for s in run_pipeline.build_steps(cfg) if s.phase == "g1"]
    names = {s.name for s in steps}
    assert {f"g1_train_seed{seed}" for seed in run_pipeline.G1_SEEDS} <= names
    assert {"g1_select", "g1_final_seed42", "g1_compare"} <= names
    for step in steps:
        command = run_pipeline.resolve(list(step.command), dry_run=True)
        if step.name.startswith("g1_train") or step.name == "g1_smoke_train":
            assert command[command.index("--arch") + 1] == "deepjsccq"
            assert command[command.index("--config") + 1] == "configs/deepjsccq_identity_train.yaml"
            assert command[command.index("--device") + 1] == "mps"
        if "--split" in command:
            split = command[command.index("--split") + 1]
            assert split == ("test" if "final" in step.name else "val"), step.name
    assert "g1" not in run_pipeline.PHASES and "g1" in run_pipeline.EXTENSION_PHASES
    for seed, path in run_pipeline.ADJSCC_PUBLISHED.items():
        assert os.path.exists(path), path
        assert os.path.exists(f"published_results/gpu/models/seed{seed}/oracle_ranking.json")
