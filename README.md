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

_Last updated: 2026-10-19._ **All phases have been run end to end with the full protocol** (models trained from scratch on
Kaggle, validation-only scheme selection, test split evaluated once per table, 5 channel seeds, every test access logged in
`results/test_access_log.jsonl`). Small result files are in [`published_results/`](published_results/) (tables, figures,
checkpoints under 3 MB, logs, manifest with SHA-256); datasets, per-image arrays and large weights stay out of git.

| Stage | State |
|---|---|
| ADJSCC-Q model, GF(256) Reed-Solomon codec, packet-interleaved layout, erasure channel | Done, unit-tested (180 tests) |
| Phase 0 hygiene: shared loss mask, side-info rule, determinism, provenance, seed-level statistics | Done |
| Phase 1: 5 converged seeds, attention vs damage, CIFAR-100 transfer (A.1), cheap proxies (A.2), sensitivity head (A.3) | Done |
| Phase 2: erasure-aware decoder training (B.1) and inpainting head (B.2) | Done; **partial success** (see below) |
| Phase 3: Kodak 768x512 (C.1) and bandwidth-ratio sweep (C.2) | Done; one k/n model did not converge |
| Phase 4: Gilbert-Elliott bursts (D.1) and codec + RS baseline (D.2) | Done |
| Phase 5: figures, headline table, auto-generated insights, orchestrator | Done: [`published_results/gpu/report/`](published_results/gpu/report/) |
| Manuscript | Not started |

Headline results (CIFAR-10 test, 10,000 images x 5 channel seeds, gains are paired differences to equal-protection RS, 95% CI
about +-0.03-0.07 dB; best tiered scheme per loss rate):

| Packet loss | 0.2 | 0.3 | 0.35 | 0.4 | 0.45 | 0.5 |
|---|---|---|---|---|---|---|
| seed-42 model, best tiered scheme vs uniform RS | -1.15 | -0.87 | +1.42 | **+3.00** | +2.83 | +2.00 |

* **Gain exists only at high loss.** Tiered protection beats uniform RS from about 32% loss; below that it **loses** 0.9-1.8 dB
  (a block-length effect). Peak gain is +2.6 to +3.2 dB at 40% loss across the five converged seeds.
* **Which channels are protected matters.** Random-order tiers give only +0.50 dB against +2.19 dB for the oracle ranking
  (mean over loss >= 0.35). The ADJSCC **attention gates are no better than chance** (+0.31 dB; within-image
  Spearman -0.015). Latent **energy** (+2.24 dB), L1 and the 2048-MAC **sensitivity head** (+2.31 dB) match or beat the oracle.
* **The low-loss penalty is not removed.** Erasure-aware training (B.1) and the inpainting head (B.2) shrink the worst
  penalty from -0.87 dB to -0.64 / -0.41 dB but also cut the peak gain (+3.00 -> +1.17 / +1.22 dB), and the penalty remains.
* **Bursts:** the roadmap target (> 3 dB at burst length >= 4) is **not met** (gains 0.5-2.5 dB). Interleaving tiers across
  the timeline keeps most of the gain at long bursts; sending tiers in order loses it (35% loss, burst 16: -0.56 dB vs +0.80 dB).
* **Digital baselines (JPEG / JPEG2000 / WebP + RS)** are better with no loss (33-38 dB vs 30.9 dB) but fall below tiered Deep
  JSCC from 35% loss (25.7 dB vs 20.7-23.0 dB at 35%).
* **Convergence caveats:** under the original training recipe two runs (init seed 2024 and the k/n = 1/3 model) did not converge
  and are flagged, not hidden (`results/analysis/convergence_ledger.json`). Two alternative recipes (cosine+clip, lower lr)
  did not fix this and were rejected. The 5 converged seeds are used for conclusions.
* The Kodak experiment uses the CIFAR-trained model on 768x512 images (clean PSNR 24.3 dB), so absolute quality is low; only
  the relative behaviour is informative.

Full chronology, decisions, deviations from the roadmap, problems and fixes: [`documentations/research_progress/PROGRESS_LOG.md`](documentations/research_progress/PROGRESS_LOG.md).
Auto-generated findings: [`published_results/gpu/report/INSIGHTS.md`](published_results/gpu/report/INSIGHTS.md).

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
