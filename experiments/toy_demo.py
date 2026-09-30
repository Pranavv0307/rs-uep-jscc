"""Toy / sanity demo for the JSCC model, not an RS-UEP benchmark.

Runs a fixed small batch of CIFAR-10 images through an encoder, AWGN channel,
and decoder, then saves a side-by-side reconstruction figure. Use
``experiments.eval_rs_uep`` for the real packet-erasure and split study.

Usage::

        python -m experiments.toy_demo
        python -m experiments.toy_demo --model deepjscc --snr_db 10
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch

from data.cifar10 import get_cifar10_loaders
from models.adjscc import ADJSCCEncoder, ADJSCCDecoder
from models.deepjscc import DeepJSCCEncoder, DeepJSCCDecoder
from models.channel import AWGNChannel
from experiments.metrics import psnr_per_image, ssim

# Auto-detected checkpoint locations -- these are exactly the paths
# experiments/train_adjscc.py and experiments/overfit_test.py already write
# to (see their configs' output.checkpoint_dir), so a checkpoint produced by
# either script is picked up here with no extra wiring.
CHECKPOINT_CANDIDATES = {
    "adjscc": ["results/checkpoints/week4_adjscc/adjscc_final.pth"],
    "deepjscc": ["results/checkpoints/week1_overfit/overfit_final.pth"],
}

def build_model(model_name: str, device: torch.device):
    if model_name == "adjscc":
        encoder = ADJSCCEncoder(k_over_n=1 / 6, image_size=32).to(device)
        decoder = ADJSCCDecoder(k=encoder.k, image_size=32).to(device)
    else:
        encoder = DeepJSCCEncoder(k_over_n=1 / 6, image_size=32).to(device)
        decoder = DeepJSCCDecoder(k=encoder.k, image_size=32).to(device)
    return encoder, decoder


def load_checkpoint_if_available(encoder, decoder, model_name: str, checkpoint_arg: str, device) -> bool:
    path = checkpoint_arg or next((p for p in CHECKPOINT_CANDIDATES[model_name] if os.path.exists(p)), None)
    if path is None:
        print(f"No checkpoint found for --model {model_name} "
              f"(looked in: {CHECKPOINT_CANDIDATES[model_name]}).")
        print("Proceeding with RANDOMLY-INITIALIZED weights -- this demonstrates the "
              "pipeline *shape* only. Reconstructions will look like colored noise, "
              "not a working codec, until a real checkpoint exists (see "
              "experiments/train_adjscc.py / experiments/overfit_test.py).")
        return False
    ckpt = torch.load(path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    print(f"Loaded checkpoint: {path}")
    return True


def run_pipeline(encoder, decoder, channel, x, snr_db, model_name):
    with torch.no_grad():
        if model_name == "adjscc":
            z, attn = encoder(x, snr_db, return_attn=True)
        else:
            z, attn = encoder(x), None

        z_noisy = channel(z, snr_db=snr_db)

        if model_name == "adjscc":
            x_hat = decoder(z_noisy, snr_db)
        else:
            x_hat = decoder(z_noisy)
    return x_hat, attn


def main():
    parser = argparse.ArgumentParser(
        description="Toy end-to-end sanity demo for the RS-UEP-JSCC pipeline "
                     "(see the module docstring for what's fixed vs. tunable).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--snr_db", type=float, default=10.0, help="AWGN channel SNR in dB.")
    parser.add_argument("--model", choices=["adjscc", "deepjscc"], default="adjscc")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Override the auto-detected checkpoint path.")
    parser.add_argument("--n_images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out", type=str, default="results/toy_demo/reconstruction3.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Fixed inputs: first N images of the CIFAR-10 *test* split (shuffle=False
    # in data/cifar10.py), so the same images come back every run.
    _, test_loader = get_cifar10_loaders(data_dir=args.data_dir, batch_size=args.n_images, num_workers=0)
    x, _ = next(iter(test_loader))
    x = x[: args.n_images].to(device)

    encoder, decoder = build_model(args.model, device)
    encoder.eval()
    decoder.eval()
    has_checkpoint = load_checkpoint_if_available(encoder, decoder, args.model, args.checkpoint, device)

    channel = AWGNChannel().to(device)
    print(f"\nRunning: model={args.model} | snr_db={args.snr_db} | "
          f"checkpoint={'yes' if has_checkpoint else 'NO (random init)'}")

    x_hat, attn = run_pipeline(encoder, decoder, channel, x, args.snr_db, args.model)

    if attn is not None:
        gate = attn["af5_bottleneck"]
        print(f"\n[forward-looking, not part of this demo's claim] ADJSCC bottleneck "
              f"attention gate -- this is the per-channel importance signal Week 8's "
              f"RS-tiering will consume: mean={gate.mean().item():.3f}, "
              f"std={gate.std().item():.3f}, min={gate.min().item():.3f}, "
              f"max={gate.max().item():.3f}")

    psnr_vals = psnr_per_image(x_hat, x)
    ssim_vals = ssim(x_hat, x)

    print(f"\n{'image':<8}{'PSNR (dB)':>12}{'SSIM':>10}")
    for i in range(x.shape[0]):
        print(f"{i:<8}{psnr_vals[i].item():>12.2f}{ssim_vals[i].item():>10.4f}")
    print(f"{'mean':<8}{psnr_vals.mean().item():>12.2f}{ssim_vals.mean().item():>10.4f}")

    n = x.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.5))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        axes[0, i].imshow(
            x[i].permute(1, 2, 0).cpu().numpy(),
            interpolation="nearest"
        )

        axes[1, i].imshow(
            x_hat[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy(),
            interpolation="nearest"
        )

        axes[0, i].set_title(f"original #{i}")
        axes[0, i].axis("off")

        axes[1, i].set_title(f"PSNR {psnr_vals[i].item():.1f} dB\nSSIM {ssim_vals[i].item():.3f}")
        axes[1, i].axis("off")

    fig.suptitle(f"model={args.model} ({'checkpoint' if has_checkpoint else 'RANDOM INIT'}) | "
                  f"snr_db={args.snr_db}")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved side-by-side comparison to {args.out}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
