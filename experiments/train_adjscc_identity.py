"""
Phase 3 -- UNQUANTIZED ADJSCC identity-channel FULL CIFAR-10 training.

This is the missing full-scale baseline counterpart to
experiments/train_adjsccq_identity.py. Same relationship as
overfit_test_adjscc_identity.py has to overfit_test_adjsccq.py: identical
data pipeline, device selection, W&B logging pattern, epoch count, batch
size, and af_conditioning.nominal_snr_db -- the ONLY thing removed is the
quantizer.

    train_adjsccq_identity.py:   x -> ADJSCCEncoder -> quantize -> dequantize -> ADJSCCDecoder -> x_hat
    this script:                 x -> ADJSCCEncoder ------------------------> ADJSCCDecoder -> x_hat

No AWGNChannel is used here either -- identity channel means no corruption
of any kind between encoder and decoder, matching exactly what
train_adjsccq_identity.py does to z_q.

Run this, then paste its printed final PSNR into
configs/adjsccq_identity_train.yaml's report_baseline_psnr_db -- that is
the number needed to tell whether the ~0 dB quantization cost seen on the
10-image overfit smoke test (experiments/overfit_test_adjscc_identity.py)
actually holds at full-CIFAR-10 scale, where codebook utilization was
already observed to be lower (180/256 vs. 237/256) and training visibly
more volatile than the overfit run.

Run this on Colab/Kaggle (GPU) or comparable hardware -- same as
train_adjsccq_identity.py and train_adjscc.py. The analogous quantized run
(train_adjsccq_identity.py) took ~87 minutes (5221s) for 60 epochs on an
Apple M2 via MPS; expect roughly the same or slightly less here.
"""
import argparse
import os

import torch
import torch.nn as nn
import wandb
import yaml

from data.cifar10 import get_cifar10_loaders
from models.adjscc import ADJSCCEncoder, ADJSCCDecoder


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


@torch.no_grad()
def evaluate(encoder, decoder, test_loader, criterion, nominal_snr_db, device):
    encoder.eval(); decoder.eval()
    total_mse, n_batches = 0.0, 0

    for x, _ in test_loader:
        x = x.to(device)
        z = encoder(x, nominal_snr_db)          # no AWGNChannel call -- identity channel
        x_hat = decoder(z, nominal_snr_db)
        total_mse += criterion(x_hat, x).item()
        n_batches += 1

    avg_mse = total_mse / n_batches
    encoder.train(); decoder.train()
    return avg_mse, psnr_from_mse(avg_mse)


def main(config_path: str, data_dir: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected -- this run is full-dataset, "
              "not a smoke test. Switch to GPU runtime if on Colab/Kaggle.")

    nominal_snr_db = cfg["af_conditioning"]["nominal_snr_db"]

    run = wandb.init(
        project=cfg["wandb"]["project"],
        name=f"phase3-adjscc-identity-full-unquantized-baseline-snr{nominal_snr_db}",
        tags=["phase3", "adjscc", "identity-channel", "unquantized-baseline", "full-train"],
        config=cfg,
    )

    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    encoder = ADJSCCEncoder(k_over_n=cfg["model"]["k_over_n"], image_size=cfg["model"]["image_size"]).to(device)
    decoder = ADJSCCDecoder(k=encoder.k, image_size=cfg["model"]["image_size"]).to(device)

    print(f"nominal SNR (AF conditioning only, no channel noise -- identity channel) = {nominal_snr_db} dB")
    print(f"k = {encoder.k} channel symbols (ratio {cfg['model']['k_over_n']:.4f})")
    wandb.config.update({"k": encoder.k})

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["train"]["lr"])
    criterion = nn.MSELoss()

    step = 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        encoder.train(); decoder.train()

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            z = encoder(x, nominal_snr_db)          # no AWGNChannel call -- identity channel
            x_hat = decoder(z, nominal_snr_db)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            step += 1
            if step % cfg["train"]["log_every"] == 0:
                wandb.log({"train/mse": loss.item(),
                           "train/psnr_db": psnr_from_mse(loss.item())}, step=step)

        eval_mse, eval_psnr = evaluate(encoder, decoder, test_loader, criterion, nominal_snr_db, device)
        wandb.log({"eval/mse": eval_mse, "eval/psnr_db": eval_psnr}, step=step)
        print(f"epoch {epoch:3d} | eval PSNR: {eval_psnr:.2f} dB")

    print("\nRunning final evaluation...")
    final_mse, final_psnr = evaluate(encoder, decoder, test_loader, criterion, nominal_snr_db, device)
    print(f"Final unquantized ADJSCC identity-channel PSNR after {cfg['train']['epochs']} epochs: {final_psnr:.2f} dB")
    print("Paste this into configs/adjsccq_identity_train.yaml's report_baseline_psnr_db "
          "to get a valid full-scale delta out of train_adjsccq_identity.py's 29.82 dB result.")

    wandb.log({"final/psnr_db": final_psnr, "final/mse": final_mse})
    wandb.summary["final_psnr_db"] = final_psnr

    ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], "adjscc_identity_final.pth")
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "config": cfg}, ckpt_path)

    artifact = wandb.Artifact("adjscc-identity-full-unquantized-baseline-checkpoint", type="model")
    artifact.add_file(ckpt_path)
    run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/adjscc_identity_train.yaml")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()
    main(args.config, args.data_dir)
