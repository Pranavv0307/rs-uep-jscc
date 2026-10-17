"""Measure how fast one training epoch runs on this machine (GPU or CPU).

Times the real model for a few dozen optimiser steps with and without the fully
deterministic settings, and the CIFAR-10 data loader on its own, then projects
the seconds per 625-step epoch. Used to size the Kaggle jobs and to find out
*where* time goes (compute, determinism overhead, or data loading).

    python -m experiments.benchmark_training --data_dir ./data
"""
import argparse
import json
import time

import torch
import torch.nn as nn

from data.splits import make_loaders
from experiments.reproducibility import seed_everything
from models.adjsccq import NOMINAL_SNR_DB_IDENTITY, ADJSCCQDecoder, ADJSCCQEncoder

STEPS_PER_EPOCH = 625  # 40,000 images / batch 64


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_model(device, deterministic: bool, steps: int, warmup: int, batch: int) -> dict:
    seed_everything(0, deterministic=deterministic)
    encoder = ADJSCCQEncoder().to(device)
    decoder = ADJSCCQDecoder(k=encoder.k).to(device)
    x = torch.rand(batch, 3, 32, 32, device=device)
    encoder.calibrate_quantizer(x)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    criterion = nn.MSELoss()
    for step in range(warmup + steps):
        if step == warmup:
            sync(device)
            start = time.time()
        optimizer.zero_grad()
        loss = criterion(decoder(encoder(x, NOMINAL_SNR_DB_IDENTITY), NOMINAL_SNR_DB_IDENTITY), x)
        loss.backward()
        optimizer.step()
    sync(device)
    per_step = (time.time() - start) / steps
    return {"deterministic": deterministic, "seconds_per_step": per_step,
            "projected_seconds_per_epoch": per_step * STEPS_PER_EPOCH}


def time_loader(data_dir: str, batches: int, workers: int, batch: int) -> dict:
    train_loader, _, _ = make_loaders(data_dir, batch, workers, 42, 10_000)
    iterator = iter(train_loader)
    next(iterator)
    start = time.time()
    for _ in range(batches):
        next(iterator)
    per_batch = (time.time() - start) / batches
    return {"workers": workers, "seconds_per_batch": per_batch,
            "projected_seconds_per_epoch": per_batch * STEPS_PER_EPOCH}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}", flush=True)
    results = {"device": str(device)}
    for deterministic in (False, True):
        results[f"model_deterministic_{deterministic}"] = time_model(device, deterministic, args.steps, args.warmup,
                                                                       args.batch)
        print(json.dumps(results[f"model_deterministic_{deterministic}"]), flush=True)
    for workers in (0, 2):
        results[f"loader_workers_{workers}"] = time_loader(args.data_dir, 60, workers, args.batch)
        print(json.dumps(results[f"loader_workers_{workers}"]), flush=True)
    print("SUMMARY " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
