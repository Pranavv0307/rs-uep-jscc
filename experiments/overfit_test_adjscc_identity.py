"""
Phase 3 -- UNQUANTIZED ADJSCC identity-channel overfit smoke test.

This is the missing baseline counterpart to experiments/overfit_test_adjsccq.py.
It exists only so that script's PSNR delta means something: it is byte-for-byte
the same setup (same 10 fixed CIFAR-10 images via get_overfit_subset_loader,
same seed, same nominal_snr_db, same epoch count, same lr, same device-
selection/W&B conventions) with exactly ONE thing removed -- the quantizer.

    overfit_test_adjsccq.py:      x -> ADJSCCEncoder -> quantize -> dequantize -> ADJSCCDecoder -> x_hat
    this script:                  x -> ADJSCCEncoder ------------------------> ADJSCCDecoder -> x_hat

No AWGNChannel is used here either -- "identity channel" means no corruption
of any kind between encoder and decoder, matching exactly what
overfit_test_adjsccq.py does to z_q (quantize -> dequantize, no noise, no
erasure). Removing the AWGNChannel call entirely is what makes this the
correct unquantized counterpart, not experiments/overfit_test.py (which
always injects real AWGN) and not experiments/train_adjscc.py (full dataset,
real AWGN, SNR swept per batch).

Because there is no channel noise AND no quantizer, this forward pass is
fully deterministic given the fixed images and current weights -- same
reasoning overfit_test_adjsccq.py gives for skipping the 10-pass noise-
averaging that plain overfit_test.py needs.

Run this, then paste its printed "Final PSNR" into
configs/adjsccq_identity_overfit.yaml's report_baseline_psnr_db field --
that is the number the delta printed by overfit_test_adjsccq.py needs to
actually mean "cost of quantization" rather than being confounded with a
different evaluation set and/or real channel noise (see the Week 3
planning discussion this script's config docstring links back to).
"""
import argparse
import os

import torch
import torch.nn as nn
import wandb
import yaml

from data.cifar10 import get_overfit_subset_loader
from models.adjscc import ADJSCCEncoder, ADJSCCDecoder


def psnr(mse: torch.Tensor) -> float:
    if mse.item() <= 0:
        return float("inf")
    return 10 * torch.log10(1.0 / mse).item()


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

    nominal_snr_db = snr_override if snr_override is not None else cfg["af_conditioning"]["nominal_snr_db"]

    run = wandb.init(
        project=cfg["wandb"]["project"],
        name=f"phase3-adjscc-identity-unquantized-baseline-snr{nominal_snr_db}",
        tags=["phase3", "adjscc", "identity-channel", "unquantized-baseline", "overfit-smoke-test"],
        config={
            "k_over_n": cfg["model"]["k_over_n"],
            "image_size": cfg["model"]["image_size"],
            "nominal_snr_db": nominal_snr_db,
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

    encoder = ADJSCCEncoder(k_over_n=cfg["model"]["k_over_n"], image_size=cfg["model"]["image_size"]).to(device)
    decoder = ADJSCCDecoder(k=encoder.k, image_size=cfg["model"]["image_size"]).to(device)
    print(f"nominal SNR (AF conditioning only, no channel noise -- identity channel) = {nominal_snr_db} dB"
          f"{' (overridden from CLI)' if snr_override is not None else ' (from config)'}")

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["train"]["lr"])
    criterion = nn.MSELoss()

    print(f"k = {encoder.k} channel symbols (ratio {cfg['model']['k_over_n']:.4f})")
    wandb.config.update({"k": encoder.k})

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        encoder.train(); decoder.train()
        optimizer.zero_grad()
        z = encoder(x_fixed, nominal_snr_db)          # no AWGNChannel call -- identity channel
        x_hat = decoder(z, nominal_snr_db)
        loss = criterion(x_hat, x_fixed)
        loss.backward()
        optimizer.step()

        if epoch % cfg["train"]["log_every"] == 0 or epoch == 1:
            current_psnr = psnr(loss.detach())
            print(f"epoch {epoch:5d} | MSE {loss.item():.6f} | PSNR {current_psnr:.2f} dB")
            wandb.log({"overfit/mse": loss.item(), "overfit/psnr_db": current_psnr}, step=epoch)

    # Deterministic forward pass (no noise, no quantizer) -- single evaluation
    # is exact, not a Monte Carlo estimate, same reasoning as overfit_test_adjsccq.py.
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        z = encoder(x_fixed, nominal_snr_db)
        x_hat = decoder(z, nominal_snr_db)
        final_mse = criterion(x_hat, x_fixed).item()
        final_psnr = psnr(torch.tensor(final_mse))

    print(f"\nFinal unquantized ADJSCC identity-channel PSNR after {cfg['train']['epochs']} epochs: {final_psnr:.2f} dB")
    print("Paste this into configs/adjsccq_identity_overfit.yaml's report_baseline_psnr_db "
          "to get a valid delta out of overfit_test_adjsccq.py.")

    wandb.log({"final/psnr_db": final_psnr, "final/mse": final_mse})
    wandb.summary["final_psnr_db"] = final_psnr

    ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], "adjscc_identity_overfit_final.pth")
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "config": cfg}, ckpt_path)

    artifact = wandb.Artifact("adjscc-identity-overfit-unquantized-baseline-checkpoint", type="model")
    artifact.add_file(ckpt_path)
    run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/adjscc_identity_overfit.yaml")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--snr_db", type=float, default=None,
                         help="Override af_conditioning.nominal_snr_db. Does NOT change any actual "
                              "channel noise (identity channel injects none) -- only what value the "
                              "AF modules are conditioned on. Must match the value used for the "
                              "quantized run you're comparing against.")
    args = parser.parse_args()
    main(args.config, args.data_dir, snr_override=args.snr_db)
