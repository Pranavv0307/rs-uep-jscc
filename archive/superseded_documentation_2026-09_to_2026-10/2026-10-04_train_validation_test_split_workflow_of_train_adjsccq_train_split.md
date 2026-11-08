# Train/Validation/Test ADJSCC-Q and RS-UEP workflow

## Purpose

`experiments/train_adjsccq_train_split.py` is an isolated experiment entry
point for training the existing ADJSCC-Q model using a proper
train/validation/test protocol. It does not modify the existing model,
coding, data-loader, training, or evaluation modules.

The existing RS-UEP pipeline remains unchanged:

```text
image
  -> ADJSCC encoder
  -> 256-level scalar quantizer
  -> byte-valued latent indices
  -> importance ordering
  -> Reed-Solomon packetization
  -> packet-erasure channel
  -> Reed-Solomon decoding
  -> latent-space fallback for unrecovered symbols
  -> ADJSCC decoder
  -> reconstructed image
```

## Dataset protocol

The official CIFAR-10 training split contains 50,000 images. A deterministic
split is created using the configured seed:

- 40,000 images: optimization/training;
- 10,000 images: validation and best-checkpoint selection;
- 10,000 images from CIFAR-10's official test split: final evaluation only.

The official test set is not used for training, validation, checkpoint
selection, quantizer calibration, oracle ranking, or hyperparameter selection.
Quantizer calibration uses one batch from the 40,000-image training subset.

The split seed is recorded in the checkpoint and output metrics. The default
is `42`, inherited from `configs/adjsccq_identity_train.yaml`.

## Checkpoints and outputs

The default output directory is:

```text
results/train_split_adjsccq/
```

It contains:

- `best_val.pth`: checkpoint with the highest validation mean per-image PSNR;
- `final.pth`: checkpoint after the final training epoch;
- `oracle_ranking.json`: frozen training-derived oracle importance ranking;
- `metrics.json`: clean and RS-UEP metrics.

The existing `results/checkpoints/adjsccq_identity_train/` and other previous
experiment directories are not overwritten.

## Training

The existing ADJSCC-Q identity-channel training behavior is reused:

- fixed nominal SNR for AF conditioning;
- learnable 256-level scalar quantizer;
- quantizer sigma annealing;
- MSE reconstruction loss;
- Adam optimizer;
- existing model and decoder architecture.

Training logs include MSE, PSNR, and quantizer sigma. Validation logs include
MSE, mean per-image PSNR, and SSIM after every epoch.

## Importance strategies

Both strategies are evaluated on the untouched official test set:

### Attention importance

For each image, the existing bottleneck attention values are converted into a
per-image channel ordering from most to least important. This is the existing
attention-driven UEP behavior.

### Oracle importance

After selecting `best_val.pth`, the script computes a single global ranking
using only the 40,000 training images. For each latent channel and each
training image:

1. decode the clean hard-quantized latent;
2. set that one channel's latent values to zero;
3. decode again;
4. calculate the PSNR decrease relative to the clean reconstruction.

The oracle score for a channel is the mean PSNR decrease over the training
subset. Channels are ranked by descending score. This ranking is then frozen
and reused unchanged for validation and test/inference RS-UEP evaluation.

The oracle JSON also records per-channel MSE increase and SSIM decrease for
diagnostics, but PSNR drop is the ranking criterion.

## Evaluation metrics

Clean and RS-UEP evaluation records:

- dataset MSE;
- mean per-image PSNR;
- mean SSIM;
- failed-symbol fraction;
- block failure rates by tier;
- realized packet-erasure rate;
- coding and side-information overhead.

For oracle channel ablations, the output additionally records:

- PSNR drop by channel;
- MSE increase by channel;
- SSIM drop by channel.

PSNR is currently the oracle ranking metric because it is the primary
reconstruction-quality measure in the existing evaluation. MSE and SSIM are
reported alongside it rather than discarded.

## RS-UEP sweep

The validation and test sweeps evaluate both attention and oracle ordering
across every scheme in `coding.rs_pipeline.SCHEME_CANDIDATES` and the
configured erasure rates/seeds. The default rates and seeds match the
existing RS-UEP evaluator. The oracle ranking is the same frozen ranking in
both sweeps; it is never recomputed on validation or test images.

No changes are made to Reed-Solomon encoding, packet erasure, decoding,
permutation restoration, or latent fallback behavior.

## W&B compatibility

Run with `--wandb` to log:

- the experiment configuration and split metadata;
- training and validation metrics;
- final clean test MSE, PSNR, and SSIM;
- the frozen oracle ranking;
- the full RS-UEP test sweep as a W&B Table;
- best/final checkpoints and oracle JSON as a model artifact.

Use a new W&B run name/project if desired with `--wandb_name` and
`--wandb_project`. W&B is optional; local JSON/checkpoint outputs are always
written.

## Example

```bash
python -m experiments.train_adjsccq_train_split \
  --config configs/adjsccq_identity_train.yaml \
  --data_dir ./cifar10_data \
  --output_dir results/train_split_adjsccq \
  --wandb
```

For a quick smoke test before a full run, use `--oracle_max_images` and
`--eval_max_images`; these options do not change the default full evaluation
protocol.
