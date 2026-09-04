"""
Toy / sanity demo for the RS-UEP-JSCC pipeline -- NOT a training run, NOT a
benchmark, NOT the full RS-UEP experiment. Point of this script: let someone
who has never run the repo see the wire working end to end in one command,
with a picture and two numbers, in under a minute.

WHAT THIS SCRIPT SHOWS
-----------------------
Runs a fixed small batch of CIFAR-10 images through the currently-built part
of the pipeline --

    Deep JSCC / ADJSCC encoder -> [placeholder erasure + naive redundancy]
        -> AWGN channel -> Deep JSCC / ADJSCC decoder

-- and shows original vs. reconstructed images side by side, with PSNR/SSIM
printed per image. It's meant to be handed to an advisor as "here's the
pipeline running," not as evidence of model quality or that the RS-UEP
method itself works (it doesn't exist yet -- see PLACEHOLDER WARNING below).

WHAT'S FIXED (not exposed as flags -- these match configs/week4_adjscc.yaml
so numbers here stay comparable with the real training runs)
--------------------------------------------------------------
  - Dataset: CIFAR-10, image size 32x32.
  - Channel-bandwidth ratio k/n = 1/6.
  - The same N images every run: the first N images of the CIFAR-10 *test*
    split (no shuffling), so runs at different --snr_db are visually
    comparable against each other.

WHAT'S TUNABLE (CLI flags -- run --help)
------------------------------------------
  --snr_db          AWGN channel SNR in dB. Lower = noisier = worse
                     reconstruction. This part is real: models/channel.py.
  --erasure_rate     Fraction of latent "chunks" dropped by a placeholder
                     erasure model. See PLACEHOLDER WARNING.
  --rs_protection    {none, uniform}. Whether the placeholder erasure model
                     gets a naive redundancy layer. See PLACEHOLDER WARNING.
  --model            {adjscc, deepjscc}. Which encoder/decoder to run.
  --checkpoint       Path to a .pth checkpoint. Auto-detected (see
                     CHECKPOINT_CANDIDATES below) if omitted; falls back to
                     a randomly-initialized model with a loud warning.
  --n_images, --seed, --out, --data_dir   plumbing/convenience knobs.

PLACEHOLDER WARNING -- read this before showing the erasure/RS flags to
anyone as if they were the real thing
------------------------------------------------------------------------
Phase 3 of the 12-week plan (Weeks 6-7: a real `reedsolo`-based RS(n,k)
encoder/decoder over GF(256) latent symbols, plus a Bernoulli packet-erasure
channel simulator) has not been built yet -- coding/ is still an empty
package. `--erasure_rate` and `--rs_protection` here drive a small,
clearly-labeled stand-in implemented in apply_erasure_placeholder() below
(independent per-chunk zeroing; "uniform" protection = send each chunk
twice and keep whichever copy survives). It exists ONLY so this demo has
something honest to turn on/off today. It is NOT Reed-Solomon, does NOT
use GF(256) symbols, and has NO importance-tiering. Delete/replace that one
function once the real Phase 3 pipeline exists -- everything else in this
script (model, channel, PSNR/SSIM, plotting) stays valid.

WHAT THIS IS NOT
-----------------
  - Not a benchmark: no held-out sweep, no confidence intervals, no
    multi-scheme comparison table (that's Phase 5, Weeks 10-11).
  - Not evidence the RS-UEP method works: there is no importance signal and
    no per-tier redundancy here (that's Phase 4, Weeks 8-9) -- "uniform" is
    the only protection mode this script knows about.
  - Not necessarily a trained model: if no checkpoint is found for
    --model, this runs with random weights so the pipeline *shape* is still
    visible, but the reconstructions will look like colored noise, not a
    working codec. The script tells you loudly when this happens.

USAGE
-----
    ppython -m experiments.toy_demo
    ppython -m experiments.toy_demo --snr_db 2 --erasure_rate 0.3 --rs_protection uniform
    ppython -m experiments.toy_demo --snr_db 2 --erasure_rate 0.3 --rs_protection none
    ppython -m experiments.toy_demo --model deepjscc --snr_db 10
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data.cifar10 import get_cifar10_loaders
from models.adjscc import ADJSCCEncoder, ADJSCCDecoder
from models.deepjscc import DeepJSCCEncoder, DeepJSCCDecoder
from models.channel import AWGNChannel

# Auto-detected checkpoint locations -- these are exactly the paths
# experiments/train_adjscc.py and experiments/overfit_test.py already write
# to (see their configs' output.checkpoint_dir), so a checkpoint produced by
# either script is picked up here with no extra wiring.
CHECKPOINT_CANDIDATES = {
    "adjscc": ["results/checkpoints/week4_adjscc/adjscc_final.pth"],
    "deepjscc": ["results/checkpoints/week1_overfit/overfit_final.pth"],
}

# Number of equal-size pieces the latent vector is split into for the
# erasure placeholder below. Not exposed as a CLI flag -- it's a detail of
# the placeholder, not a real system parameter (the real Phase 3 packetizer
# will define this properly).
NUM_ERASURE_CHUNKS = 32


def apply_erasure_placeholder(z: torch.Tensor, erasure_rate: float, rs_protection: str,
                               generator: torch.Generator) -> torch.Tensor:
    """
    PLACEHOLDER for Phase 3 (Weeks 6-7): real packet-erasure channel +
    Reed-Solomon erasure coding over GF(256) latent symbols. Not built yet
    -- coding/ is still empty. TODO(Week 6-7): replace this entire function
    with a real Bernoulli/Gilbert-Elliott erasure-channel simulator feeding
    a reedsolo RS(n,k) encoder/decoder, per the deliverables plan.

    What this stand-in actually does: splits the latent vector into
    NUM_ERASURE_CHUNKS equal pieces and, independently per chunk:
      - rs_protection == "none":    erases the chunk (zeros it) with
                                     probability `erasure_rate`.
      - rs_protection == "uniform": simulates sending the chunk twice and
                                     only losing it if *both* copies are
                                     erased, i.e. probability
                                     `erasure_rate ** 2`. This is a naive
                                     2x-overhead repetition code, not RS --
                                     it exists purely to give
                                     --rs_protection a visible, honest
                                     effect before real RS exists.
    Returns a new tensor; does not modify z in place.
    """
    if erasure_rate <= 0.0:
        print("  [erasure placeholder] erasure_rate=0 -- no chunks dropped, pipeline runs clean.")
        return z

    B, L = z.shape
    num_chunks = min(NUM_ERASURE_CHUNKS, L)
    bounds = torch.linspace(0, L, num_chunks + 1).round().long()

    z_out = z.clone()
    n_erased, n_total = 0, 0
    for b in range(B):
        for c in range(num_chunks):
            lo, hi = bounds[c].item(), bounds[c + 1].item()
            if hi <= lo:
                continue
            n_total += 1
            draw = torch.rand(2, generator=generator).tolist()
            if rs_protection == "uniform":
                erased = (draw[0] < erasure_rate) and (draw[1] < erasure_rate)
            else:
                erased = draw[0] < erasure_rate
            if erased:
                z_out[b, lo:hi] = 0.0
                n_erased += 1

    print(f"  [erasure placeholder] {n_erased}/{n_total} latent chunks lost "
          f"(target rate {erasure_rate:.2f}, protection={rs_protection})")
    return z_out


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = (g1d / g1d.sum()).unsqueeze(1)
    g2d = g1d @ g1d.t()
    return g2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, data_range: float = 1.0) -> torch.Tensor:
    """Standard windowed SSIM (Wang et al. 2004), self-contained in torch so
    this demo has no extra dependency beyond what the rest of the repo
    already requires. Returns one SSIM value per image in the batch."""
    channels = img1.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean(dim=[1, 2, 3])


def psnr_per_image(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mse = ((x_hat - x) ** 2).mean(dim=[1, 2, 3])
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


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


def run_pipeline(encoder, decoder, channel, x, snr_db, erasure_rate, rs_protection, model_name, generator):
    with torch.no_grad():
        if model_name == "adjscc":
            z, attn = encoder(x, snr_db, return_attn=True)
        else:
            z, attn = encoder(x), None

        z_erased = apply_erasure_placeholder(z, erasure_rate, rs_protection, generator)
        z_noisy = channel(z_erased, snr_db=snr_db)

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
    parser.add_argument("--erasure_rate", type=float, default=0.0,
                         help="Fraction of latent chunks erased (PLACEHOLDER, see docstring).")
    parser.add_argument("--rs_protection", choices=["none", "uniform"], default="none",
                         help="Placeholder redundancy scheme, see docstring.")
    parser.add_argument("--model", choices=["adjscc", "deepjscc"], default="adjscc")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Override the auto-detected checkpoint path.")
    parser.add_argument("--n_images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./cifar10_data")
    parser.add_argument("--out", type=str, default="results/toy_demo/reconstruction.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    erasure_generator = torch.Generator().manual_seed(args.seed)

    print(f"\nRunning: model={args.model} | snr_db={args.snr_db} | "
          f"erasure_rate={args.erasure_rate} | rs_protection={args.rs_protection} | "
          f"checkpoint={'yes' if has_checkpoint else 'NO (random init)'}")

    x_hat, attn = run_pipeline(encoder, decoder, channel, x, args.snr_db,
                                args.erasure_rate, args.rs_protection, args.model, erasure_generator)

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
        axes[0, i].imshow(x[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title(f"original #{i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(x_hat[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title(f"PSNR {psnr_vals[i].item():.1f} dB\nSSIM {ssim_vals[i].item():.3f}")
        axes[1, i].axis("off")

    fig.suptitle(f"model={args.model} ({'checkpoint' if has_checkpoint else 'RANDOM INIT'}) | "
                  f"snr_db={args.snr_db} | erasure_rate={args.erasure_rate} | "
                  f"rs_protection={args.rs_protection}")
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
