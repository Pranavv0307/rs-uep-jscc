"""
Week 4-5: train ADJSCC end to end on full CIFAR-10, across a range of
SNRs (sampled per-sample per batch), and evaluate PSNR on a fixed SNR
grid so it's comparable against Week 3's fixed-SNR DeepJSCC baseline.

Run on Colab/Kaggle (GPU), same as experiments/overfit_test.py.
"""
import argparse
import os
import yaml
import torch
import torch.nn as nn
import wandb

from data.cifar10 import get_cifar10_loaders
from models.adjscc import ADJSCCEncoder, ADJSCCDecoder
from models.channel import AWGNChannel


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def sample_snr_batch(batch_size: int, snr_min: float, snr_max: float, device) -> torch.Tensor:
    """One SNR per sample, uniform over [snr_min, snr_max] -- this is
    what lets the AF modules learn a continuous SNR-to-gate mapping
    instead of memorizing a handful of discrete operating points."""
    return torch.empty(batch_size, device=device).uniform_(snr_min, snr_max)


@torch.no_grad()
def evaluate(encoder, decoder, channel, test_loader, criterion, eval_snrs, device):
    encoder.eval(); decoder.eval()
    results = {}
    for snr in eval_snrs:
        total_mse, n_batches = 0.0, 0
        for x, _ in test_loader:
            x = x.to(device)
            snr_tensor = torch.full((x.shape[0],), float(snr), device=device)
            z = encoder(x, snr_tensor)
            z_hat = channel(z, snr_tensor)
            x_hat = decoder(z_hat, snr_tensor)
            total_mse += criterion(x_hat, x).item()
            n_batches += 1
        avg_mse = total_mse / n_batches
        results[snr] = {"mse": avg_mse, "psnr_db": psnr_from_mse(avg_mse)}
    encoder.train(); decoder.train()
    return results


def main(config_path: str, data_dir: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected -- this run is full-dataset, "
              "not the Week 1 10-image overfit test. Switch to GPU runtime.")

    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    run = wandb.init(
        project=cfg["wandb"]["project"],
        name="week4-adjscc-multisnr",
        tags=["week4", "adjscc", "multi-snr"],
        config=cfg,
    )

    encoder = ADJSCCEncoder(k_over_n=cfg["model"]["k_over_n"],
                             image_size=cfg["model"]["image_size"]).to(device)
    decoder = ADJSCCDecoder(k=encoder.k,
                             image_size=cfg["model"]["image_size"]).to(device)
    channel = AWGNChannel().to(device)  # no fixed snr_db -- always overridden per batch
    print(f"k = {encoder.k} channel symbols (ratio {cfg['model']['k_over_n']:.4f})")
    wandb.config.update({"k": encoder.k})

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["train"]["lr"])
    criterion = nn.MSELoss()

    snr_min, snr_max = cfg["channel"]["snr_db_min"], cfg["channel"]["snr_db_max"]
    step = 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        for x, _ in train_loader:
            x = x.to(device)
            snr_batch = sample_snr_batch(x.shape[0], snr_min, snr_max, device)

            optimizer.zero_grad()
            z = encoder(x, snr_batch)
            z_hat = channel(z, snr_batch)
            x_hat = decoder(z_hat, snr_batch)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            step += 1
            if step % cfg["train"]["log_every"] == 0:
                wandb.log({"train/mse": loss.item(),
                           "train/psnr_db": psnr_from_mse(loss.item())}, step=step)

        eval_results = evaluate(encoder, decoder, channel, test_loader, criterion,
                                 cfg["train"]["eval_snrs"], device)
        log_dict = {}
        for snr, r in eval_results.items():
            log_dict[f"eval/psnr_db_snr{snr}"] = r["psnr_db"]
        wandb.log(log_dict, step=step)
        print(f"epoch {epoch:3d} | " +
              " | ".join(f"SNR{snr}dB: {r['psnr_db']:.2f}dB" for snr, r in eval_results.items()))

    ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], "adjscc_final.pth")
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, ckpt_path)
    artifact = wandb.Artifact("adjscc-checkpoint", type="model")
    artifact.add_file(ckpt_path)
    run.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/week4_adjscc.yaml")
    parser.add_argument("--data_dir", type=str, default="./cifar10_data")
    args = parser.parse_args()
    main(args.config, args.data_dir)