"""
Week 1 pipeline sanity check: overfit a 10-image CIFAR-10 subset. Run this
on Colab or Kaggle (GPU), not locally. Success: PSNR > 35 dB.
"""
import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data.cifar10 import get_overfit_subset_loader
from models.deepjscc import DeepJSCCEncoder, DeepJSCCDecoder
from models.channel import AWGNChannel


def psnr(mse: torch.Tensor) -> float:
    if mse.item() <= 0:
        return float("inf")
    return 10 * torch.log10(1.0 / mse).item()


def main(config_path: str, data_dir: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected. On Colab: Runtime > Change runtime "
              "type > GPU. On Kaggle: Settings panel > Accelerator > GPU.")

    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["log_dir"], exist_ok=True)
    writer = SummaryWriter(cfg["output"]["log_dir"])

    loader = get_overfit_subset_loader(
        data_dir=data_dir,
        n_images=cfg["data"]["n_overfit_images"],
        batch_size=cfg["data"]["batch_size"],
    )
    x_fixed, _ = next(iter(loader))
    x_fixed = x_fixed.to(device)

    encoder = DeepJSCCEncoder(k_over_n=cfg["model"]["k_over_n"],
                               image_size=cfg["model"]["image_size"]).to(device)
    decoder = DeepJSCCDecoder(k=encoder.k,
                               image_size=cfg["model"]["image_size"]).to(device)
    channel = AWGNChannel(snr_db=cfg["channel"]["snr_db"]).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["train"]["lr"])
    criterion = nn.MSELoss()

    print(f"k = {encoder.k} channel symbols (ratio {cfg['model']['k_over_n']:.4f})")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        encoder.train(); decoder.train()
        optimizer.zero_grad()
        z = encoder(x_fixed)
        z_hat = channel(z)
        x_hat = decoder(z_hat)
        loss = criterion(x_hat, x_fixed)
        loss.backward()
        optimizer.step()

        if epoch % cfg["train"]["log_every"] == 0 or epoch == 1:
            current_psnr = psnr(loss.detach())
            print(f"epoch {epoch:5d} | MSE {loss.item():.6f} | PSNR {current_psnr:.2f} dB")
            writer.add_scalar("overfit/mse", loss.item(), epoch)
            writer.add_scalar("overfit/psnr_db", current_psnr, epoch)

    final_psnr = psnr(loss.detach())
    print(f"\nFinal PSNR after {cfg['train']['epochs']} epochs: {final_psnr:.2f} dB")
    if final_psnr < 35:
        print("WARNING: PSNR below 35 dB — likely pipeline bug, debug before Week 2.")
    else:
        print("PASS — pipeline verified. Safe to proceed to Week 2.")

    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
                os.path.join(cfg["output"]["checkpoint_dir"], "overfit_final.pth"))
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/week1_overfit.yaml")
    parser.add_argument("--data_dir", type=str, default="./cifar10_data")
    args = parser.parse_args()
    main(args.config, args.data_dir)
