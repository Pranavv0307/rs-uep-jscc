import yaml
import torch
import torch.nn as nn
from data.cifar10 import get_cifar10_loaders
from models.adjscc import ADJSCCEncoder, ADJSCCDecoder
from models.channel import AWGNChannel
from experiments.train_adjscc import evaluate

def main():
    with open("configs/week4_adjscc.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    _, test_loader = get_cifar10_loaders(
        data_dir="./data",
        batch_size=256,
        num_workers=0,
    )

    encoder = ADJSCCEncoder(k_over_n=cfg["model"]["k_over_n"], image_size=cfg["model"]["image_size"]).to(device)
    decoder = ADJSCCDecoder(k=encoder.k, image_size=cfg["model"]["image_size"]).to(device)
    channel = AWGNChannel().to(device)

    ckpt = torch.load("results/checkpoints/week4_adjscc/adjscc_final.pth", map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    criterion = nn.MSELoss()
    eval_snrs = [0, 5, 10, 15, 20]

    print("Evaluating...")
    res = evaluate(encoder, decoder, channel, test_loader, criterion, eval_snrs, device)
    for snr, r in res.items():
        print(f"SNR {snr}dB: PSNR {r['psnr_db']:.2f}dB")

if __name__ == "__main__":
    main()
