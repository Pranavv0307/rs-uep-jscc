"""
Phase 3 -- ADJSCC-Q identity-channel full CIFAR-10 training script.

Trains ADJSCCQEncoder + ADJSCCQDecoder (models/adjsccq.py) end to end
on CIFAR-10 under the IDENTITY channel: quantize -> dequantize -> decoder,
no AWGN, no erasure.

Unlike the unquantized ADJSCC training (train_adjscc.py) which sweeps SNR
per batch, this script uses a single fixed `nominal_snr_db` for AF conditioning
as there is no real channel noise. See models/adjsccq.py's docstring for
why identity channel was chosen and the caveat about evaluating on AWGN/erasure.
"""
import argparse
import os
import yaml
import torch
import torch.nn as nn
import wandb

from data.cifar10 import get_cifar10_loaders
from models.adjsccq import ADJSCCQEncoder, ADJSCCQDecoder, NOMINAL_SNR_DB_IDENTITY


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def anneal_sigma(epoch: int, sigma_start: float, sigma_end: float, anneal_epochs: int) -> float:
    if epoch >= anneal_epochs:
        return sigma_end
    frac = epoch / max(anneal_epochs, 1)
    return sigma_start + frac * (sigma_end - sigma_start)


@torch.no_grad()
def evaluate(encoder, decoder, test_loader, criterion, nominal_snr_db, device, return_levels=False):
    encoder.eval(); decoder.eval()
    total_mse, n_batches = 0.0, 0
    unique_levels = set()

    for x, _ in test_loader:
        x = x.to(device)
        if return_levels:
            z_q, idx = encoder(x, nominal_snr_db, return_indices=True)
            unique_levels.update(idx.unique().tolist())
        else:
            z_q = encoder(x, nominal_snr_db)
            
        x_hat = decoder(z_q, nominal_snr_db)
        total_mse += criterion(x_hat, x).item()
        n_batches += 1
        
    avg_mse = total_mse / n_batches
    encoder.train(); decoder.train()
    
    if return_levels:
        return avg_mse, psnr_from_mse(avg_mse), len(unique_levels)
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
        name=f"phase3-adjsccq-identity-full-snr{nominal_snr_db}",
        tags=["phase3", "adjsccq", "identity-channel", "full-train"],
        config=cfg,
    )

    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    encoder = ADJSCCQEncoder(
        k_over_n=cfg["model"]["k_over_n"],
        image_size=cfg["model"]["image_size"],
        num_levels=cfg["quantizer"]["num_levels"],
        init_range=tuple(cfg["quantizer"]["init_range"]),
        init_sigma=cfg["quantizer"]["sigma_start"],
    ).to(device)
    decoder = ADJSCCQDecoder(k=encoder.k, image_size=cfg["model"]["image_size"]).to(device)
    
    print(f"nominal SNR (AF conditioning only, no channel noise) = {nominal_snr_db} dB")
    print(f"k = {encoder.k} channel symbols (ratio {cfg['model']['k_over_n']:.4f})")
    wandb.config.update({"k": encoder.k})

    # Calibrate codebook centers using a batch from the training data
    x_calib, _ = next(iter(train_loader))
    x_calib = x_calib.to(device)
    encoder.calibrate_quantizer(
        x_calib, snr_db=nominal_snr_db,
        low_pct=cfg["quantizer"]["calibrate_low_pct"],
        high_pct=cfg["quantizer"]["calibrate_high_pct"],
    )

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["train"]["lr"])
    criterion = nn.MSELoss()

    sigma_start = cfg["quantizer"]["sigma_start"]
    sigma_end = cfg["quantizer"]["sigma_end"]
    sigma_anneal_epochs = cfg["quantizer"]["sigma_anneal_epochs"]

    step = 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        sigma = anneal_sigma(epoch - 1, sigma_start, sigma_end, sigma_anneal_epochs)
        encoder.quantizer.set_sigma(sigma)
        encoder.train(); decoder.train()

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            z_q = encoder(x, nominal_snr_db)
            x_hat = decoder(z_q, nominal_snr_db)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            step += 1
            if step % cfg["train"]["log_every"] == 0:
                wandb.log({"train/mse": loss.item(),
                           "train/psnr_db": psnr_from_mse(loss.item()),
                           "train/sigma_q": sigma}, step=step)

        # Periodic evaluation
        eval_mse, eval_psnr = evaluate(encoder, decoder, test_loader, criterion, nominal_snr_db, device)
        wandb.log({"eval/mse": eval_mse, "eval/psnr_db": eval_psnr}, step=step)
        print(f"epoch {epoch:3d} | sigma_q {sigma:.4f} | eval PSNR: {eval_psnr:.2f} dB")

    # Final hard-quantized evaluation
    print("\nRunning final hard-quantized evaluation...")
    encoder.quantizer.set_sigma(sigma_end)
    final_mse, final_psnr, used_levels = evaluate(
        encoder, decoder, test_loader, criterion, nominal_snr_db, device, return_levels=True
    )
    print(f"Final PSNR after {cfg['train']['epochs']} epochs: {final_psnr:.2f} dB")
    print(f"Codebook utilization: {used_levels}/{encoder.quantizer.num_levels} levels used")

    baseline = cfg.get("report_baseline_psnr_db")
    if baseline is not None:
        delta = final_psnr - baseline
        print(f"Delta vs baseline ({baseline:.2f} dB): {delta:+.2f} dB")
        wandb.summary["psnr_delta_vs_baseline_db"] = delta

    wandb.log({
        "final/psnr_db": final_psnr,
        "final/mse": final_mse,
        "final/codebook_levels_used": used_levels,
    })
    wandb.summary["final_psnr_db"] = final_psnr
    wandb.summary["codebook_levels_used"] = used_levels

    ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], "adjsccq_identity_final.pth")
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "config": cfg}, ckpt_path)

    artifact = wandb.Artifact("adjsccq-identity-full-checkpoint", type="model")
    artifact.add_file(ckpt_path)
    run.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/adjsccq_identity_train.yaml")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()
    main(args.config, args.data_dir)
