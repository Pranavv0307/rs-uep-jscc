# rs-uep-jscc: sensitivity-aware unequal error protection for deep JSCC over packet-erasure channels

Deep joint source-channel coding (ADJSCC-Q) whose quantised latent bytes are protected by
Reed-Solomon codes over a packet-erasure channel. The question: if some latent channels matter far
more than others, does giving them stronger protection (UEP) beat equal protection at the same
bandwidth, and how can a transmitter know which channels matter?

Working title: *When Attention Fails: Sensitivity-Aware Unequal Error Protection for Deep Joint
Source-Channel Coding over Packet-Erasure Channels.* Target: IEEE GLOBECOM / ICC, extended to TWC / TCOM.
The plan is [`documentations/TOP_TIER_RESEARCH_ROADMAP.md`](documentations/TOP_TIER_RESEARCH_ROADMAP.md).

---

## Project status

_Last updated: 2026-10-17._ **Honest summary: the foundation is done and measured; the full pipeline for the
remaining work is written but has not been run end to end yet.**

| Stage | State |
|---|---|
| ADJSCC-Q model, GF(256) Reed-Solomon codec, packet-interleaved layout, erasure channel | Done, unit-tested |
| Finding: attention gates do not predict erasure damage; leave-one-out "oracle" does (rho = -0.018 for attention) | Done (earlier runs) |
| Train/val/test protocol (40k / 10k / 10k), leak-free oracle from the training split | Done; one model trained (seed 42) |
| **Phase 0** hygiene: shared loss mask, side-info rule, determinism, provenance, seed-level statistics | Code done; 103 unit tests passed |
| **Phase 1** multi-seed models, attention figure, cross-dataset transfer (A.1), cheap proxies (A.2), sensitivity head (A.3) | Code written, **not yet run** |
| **Phase 2** erasure-aware decoder training (B.1) and inpainting head (B.2) | Code written, **not yet run** |
| **Phase 3** Kodak 768x512 (C.1) and bandwidth-ratio sweep (C.2) | Code written, **not yet run** |
| **Phase 4** Gilbert-Elliott bursts (D.1) and codec + RS baseline (D.2) | Code written, **not yet run** |
| **Phase 5** figures, headline table, auto-generated insights, orchestrator | Code written, **not yet run** |
| Manuscript | Not started |

What has been measured so far (earlier code, 3 channel seeds; to be re-measured with the new evaluator):

* clean PSNR 31.2 dB on the official test split (SSIM 0.965), no train/test gap;
* one latent channel costs ~12 dB when lost, the least important ~1.5 dB;
* oracle-guided UEP beats equal protection by about +2 dB at 40-45% packet loss, but **loses** by
  1-1.5 dB at 20-30% loss (the "low-loss penalty" that Phase 2 targets).

These numbers are not final; they are replaced by the full-protocol results.
Full chronology, decisions, deviations from the roadmap and known risks: [`documentations/research_progress/PROGRESS_LOG.md`](documentations/research_progress/PROGRESS_LOG.md).

## How to run

Everything is driven by [`run_pipeline.py`](run_pipeline.py), which **prints** the commands by default
and runs them only with `--execute`:

```bash
python run_pipeline.py --list
python run_pipeline.py --phase verify --execute      # unit tests + smoke test first
python run_pipeline.py --phase phase1 --execute --skip_existing --workers 8 --wandb
```

Per-phase commands, outputs and how to interpret them: [`documentations/research_progress/RUNBOOK.md`](documentations/research_progress/RUNBOOK.md).

## Setup

```bash
python -m pip install torch torchvision numpy pillow matplotlib pyyaml tqdm pytest wandb
python -m experiments.download_data --cifar10 --cifar100 --kodak
```

* **Weights & Biases (optional):** create a free account, copy your key from <https://wandb.ai/authorize>, then run
  `wandb login` once and paste it (or put `WANDB_API_KEY=...` in a git-ignored `.env`). Add `--wandb` to any script.
* **Kaggle (optional, for GPU training):** Kaggle -> Settings -> API -> *Create New Token*, save the file as
  `.kaggle/kaggle.json` inside this folder (git-ignored). Credentials are never committed.

## Scientific-integrity rules enforced by the code

1. The official test split is used for final tables only; every access is logged to `results/test_access_log.jsonl`.
2. Tier allocation and checkpoints are chosen on the validation split.
3. All methods face the **same** packet losses (one master mask per batch/seed/rate).
4. Failed Reed-Solomon blocks are filled with 0.0 in latent space; no ground truth reaches the receiver.
5. Overhead is reported with and without side-information (0 B for equal protection, one byte per channel for ranked methods).
6. Error bars are across channel seeds with paired differences; model-seed variance is reported separately.
7. Deterministic seeding everywhere; every result file carries git commit, versions, device, checkpoint hash and command line.

## Layout

```
coding/        Reed-Solomon codec, packetiser, erasure + burst channels, scheme scaling, interleaving
models/        ADJSCC-Q, quantiser, sensitivity head, latent inpainter
data/          CIFAR-10 40k/10k/10k split, CIFAR-100, STL-10, Kodak, high-res crops
experiments/   training, evaluation sweeps, analyses, reporting (see RUNBOOK.md)
tests/         unit and end-to-end tests (tiny random model, no data needed)
documentations/  roadmap, project report, research_progress/{PROGRESS_LOG,RUNBOOK}.md
run_pipeline.py  one entry point for every phase
```

Datasets, model weights and large outputs are git-ignored. Small results (JSON, markdown, figures) are committed.
