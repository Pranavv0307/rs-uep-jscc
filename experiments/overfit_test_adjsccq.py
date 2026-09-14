"""
Phase 3 -- ADJSCC-Q identity-channel overfit smoke test.

Mirrors experiments/overfit_test.py's structure and conventions as
closely as the identity-channel setup allows: same device-selection
block, same W&B logging/artifact pattern, same nested config schema,
same get_overfit_subset_loader() call. Trains ADJSCCQEncoder +
ADJSCCQDecoder (models/adjsccq.py) end to end on a small fixed set of
CIFAR-10 images under the IDENTITY channel: quantize -> dequantize ->
decoder, no AWGN, no erasure. See models/adjsccq.py's module docstring
for why identity was chosen for this phase, and why a checkpoint from
this script should not be treated as robust to AWGN or erasure.

Two differences from overfit_test.py, both because there is no real
channel in this phase:
  1. No `evaluate_avg_psnr` "average over 10 noise realizations" step.
     Under identity channel, encode -> quantize -> decode is a
     deterministic function of the input and the current weights (the
     forward pass injects no randomness anywhere), so a single
     evaluation pass is exact, not a Monte Carlo estimate.
  2. No fixed 35 dB pass/fail bar. Some PSNR loss from 256-level
     quantization is expected; the informative number is the PSNR
     DELTA against the unquantized ADJSCC baseline (report_baseline_
     psnr_db in the config), not an absolute threshold.

Run this on Colab or Kaggle (GPU), not locally, same as overfit_test.py.
"""
import argparse
import os

import torch
import torch.nn as nn
import wandb
import yaml

from data.cifar10 import get_overfit_subset_loader
from models.adjsccq import ADJSCCQEncoder, ADJSCCQDecoder


def psnr(mse: torch.Tensor) -> float:
    if mse.item() <= 0:
        return float("inf")
    return 10 * torch.log10(1.0 / mse).item()


def anneal_sigma(epoch: int, sigma_start: float, sigma_end: float, anneal_epochs: int) -> float:
    if epoch >= anneal_epochs:
        return sigma_end
    frac = epoch / max(anneal_epochs, 1)
    return sigma_start + frac * (sigma_end - sigma_start)


def main(config_path: str, data_dir: str, snr_override: float = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected. On Colab: Runtime > Change runtime "
              "type > GPU. On Kaggle: Settings panel > Accelerator > GPU.")

    nominal_snr_db = snr_override if snr_override is not None else cfg["af_conditioning"]["nominal_snr_db"]

    # ---- wandb init: logs config + metrics to your project dashboard ----
    run = wandb.init(
        project=cfg["wandb"]["project"],
        name=f"phase3-adjsccq-identity-snr{nominal_snr_db}",
        tags=["phase3", "adjsccq", "identity-channel", "overfit-smoke-test"],
        config={
            "k_over_n": cfg["model"]["k_over_n"],
            "image_size": cfg["model"]["image_size"],
            "nominal_snr_db": nominal_snr_db,
            "num_levels": cfg["quantizer"]["num_levels"],
            "sigma_start": cfg["quantizer"]["sigma_start"],
            "sigma_end": cfg["quantizer"]["sigma_end"],
            "sigma_anneal_epochs": cfg["quantizer"]["sigma_anneal_epochs"],
            "lr": cfg["train"]["lr"],
            "epochs": cfg["train"]["epochs"],
            "n_overfit_images": cfg["data"]["n_overfit_images"],
            "batch_size": cfg["data"]["batch_size"],
            "seed": cfg["seed"],
        },
    )

    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)

    loader = get_overfit_subset_loader(
        data_dir=data_dir,
        n_images=cfg["data"]["n_overfit_images"],
        batch_size=cfg["data"]["batch_size"],
    )
    x_fixed, _ = next(iter(loader))
    x_fixed = x_fixed.to(device)

    encoder = ADJSCCQEncoder(
        k_over_n=cfg["model"]["k_over_n"],
        image_size=cfg["model"]["image_size"],
        num_levels=cfg["quantizer"]["num_levels"],
        init_range=tuple(cfg["quantizer"]["init_range"]),
        init_sigma=cfg["quantizer"]["sigma_start"],
    ).to(device)
    decoder = ADJSCCQDecoder(k=encoder.k, image_size=cfg["model"]["image_size"]).to(device)
    print(f"nominal SNR (AF conditioning only, no channel noise this phase) = {nominal_snr_db} dB"
          f"{' (overridden from CLI)' if snr_override is not None else ' (from config)'}")

    # Calibrate codebook centers on the same images the model will
    # overfit to (fine for a smoke test; a full-dataset run should
    # calibrate on a held-out calibration split instead, not the
    # training images themselves).
    encoder.calibrate_quantizer(
        x_fixed, snr_db=nominal_snr_db,
        low_pct=cfg["quantizer"]["calibrate_low_pct"],
        high_pct=cfg["quantizer"]["calibrate_high_pct"],
    )

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["train"]["lr"])
    criterion = nn.MSELoss()

    print(f"k = {encoder.k} channel symbols (ratio {cfg['model']['k_over_n']:.4f})")
    wandb.config.update({"k": encoder.k})

    sigma_start = cfg["quantizer"]["sigma_start"]
    sigma_end = cfg["quantizer"]["sigma_end"]
    sigma_anneal_epochs = cfg["quantizer"]["sigma_anneal_epochs"]

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        sigma = anneal_sigma(epoch - 1, sigma_start, sigma_end, sigma_anneal_epochs)
        encoder.quantizer.set_sigma(sigma)

        encoder.train(); decoder.train()
        optimizer.zero_grad()
        z_q = encoder(x_fixed, nominal_snr_db)
        x_hat = decoder(z_q, nominal_snr_db)
        loss = criterion(x_hat, x_fixed)
        loss.backward()
        optimizer.step()

        if epoch % cfg["train"]["log_every"] == 0 or epoch == 1:
            current_psnr = psnr(loss.detach())
            print(f"epoch {epoch:5d} | sigma_q {sigma:.4f} | MSE {loss.item():.6f} | PSNR {current_psnr:.2f} dB")
            wandb.log({"overfit/mse": loss.item(), "overfit/psnr_db": current_psnr,
                       "overfit/sigma_q": sigma}, step=epoch)

    # Final evaluation, hard-quantized (sigma at its annealed-to-hard end
    # value). Single deterministic pass -- see module docstring for why
    # this doesn't need averaging over multiple passes the way
    # overfit_test.py's AWGN-channel evaluation does.
    encoder.eval(); decoder.eval()
    encoder.quantizer.set_sigma(sigma_end)
    with torch.no_grad():
        z_q, idx = encoder(x_fixed, nominal_snr_db, return_indices=True)
        x_hat = decoder(z_q, nominal_snr_db)
        final_mse = criterion(x_hat, x_fixed).item()
        final_psnr = psnr(torch.tensor(final_mse))
        used_levels = idx.unique().numel()

    print(f"\nFinal identity-channel PSNR after {cfg['train']['epochs']} epochs: {final_psnr:.2f} dB")
    print(f"Codebook utilization: {used_levels}/{encoder.quantizer.num_levels} levels actually used")

    baseline = cfg.get("report_baseline_psnr_db")
    if baseline is not None:
        delta = final_psnr - baseline
        print(f"Delta vs unquantized ADJSCC baseline ({baseline:.2f} dB): {delta:+.2f} dB")
        wandb.summary["psnr_delta_vs_baseline_db"] = delta
    else:
        print("No report_baseline_psnr_db set in config -- fill it in from the matching "
              "unquantized ADJSCC run to see the quantization-only PSNR cost.")

    wandb.log({
        "final/psnr_db": final_psnr,
        "final/mse": final_mse,
        "final/codebook_levels_used": used_levels,
    })
    wandb.summary["final_psnr_db"] = final_psnr
    wandb.summary["codebook_levels_used"] = used_levels

    ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], "adjsccq_identity_overfit_final.pth")
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "config": cfg}, ckpt_path)

    artifact = wandb.Artifact("adjsccq-identity-overfit-checkpoint", type="model")
    artifact.add_file(ckpt_path)
    run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/adjsccq_identity_overfit.yaml")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--snr_db", type=float, default=None,
                         help="Override af_conditioning.nominal_snr_db from config. "
                              "NOTE: unlike overfit_test.py's --snr_db, this does NOT change "
                              "any actual channel noise (identity channel injects none) -- it "
                              "only changes what value the AF modules are conditioned on.")
    args = parser.parse_args()
    main(args.config, args.data_dir, snr_override=args.snr_db)
