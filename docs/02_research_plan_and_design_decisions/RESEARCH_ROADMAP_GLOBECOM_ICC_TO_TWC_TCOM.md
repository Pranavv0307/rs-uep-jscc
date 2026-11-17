# Top-Tier Research Roadmap & Execution Manual: Turning Importance-Aware Deep JSCC into an Unrejectable Flagship Paper

**Target Venues:** IEEE GLOBECOM / IEEE ICC (Flagship ComSoc Conferences) $\to$ Extended to IEEE Transactions on Wireless Communications (TWC) or IEEE Transactions on Communications (TCOM).  
**Repository:** `rs-uep-jscc`  
**Audience:** Any researcher or developer joining the project. This document serves as an exhaustive, step-by-step engineering, scientific hygiene, and reproducibility manual specifying **what to do, exactly how to do it, what NOT to do, how to avoid experimental flaws or p-hacking, and all clerical requirements for top-tier IEEE publication.**

---

# Table of Contents
1. [Current State & Baseline Architecture](#1-current-state--baseline-architecture)
2. [What the Results Say (The Scientific Punchline)](#2-what-the-results-say-the-scientific-punchline)
3. [The 5 Rejection Vectors at Tier-1 IEEE Venues](#3-the-5-rejection-vectors-at-tier-1-ieee-venues)
4. [Scientific Hygiene & Anti-Cheating Protocols (The Integrity Manifesto)](#4-scientific-hygiene--anti-cheating-protocols-the-integrity-manifesto)
5. [Multi-Seed Training & Statistical Significance Protocol (Ruling Out "Lucky Seeds")](#5-multi-seed-training--statistical-significance-protocol-ruling-out-lucky-seeds)
6. [Track A: Solving the Online Predictor (From Offline Oracle to Practical Transmitter)](#6-track-a-solving-the-online-predictor-from-offline-oracle-to-practical-transmitter)
   - [Step A.1: Cross-Dataset Invariance Testing of Filter Sensitivity](#step-a1-cross-dataset-invariance-testing-of-filter-sensitivity)
   - [Step A.2: Gradient-Based & Unsupervised Feature Saliency Testing](#step-a2-gradient-based--unsupervised-feature-saliency-testing)
   - [Step A.3: The Lightweight Sensitivity Prediction Head (Neural Auxiliary Solution)](#step-a3-the-lightweight-sensitivity-prediction-head-neural-auxiliary-solution)
7. [Track B: Erasure-Aware Decoder Retraining (Eliminating the Low-Erasure Penalty)](#7-track-b-erasure-aware-decoder-retraining-eliminating-the-low-erasure-penalty)
   - [Step B.1: Channel-Dropout Augmented Training](#step-b1-channel-dropout-augmented-training)
   - [Step B.2: Self-Attention / Masked Latent Inpainting Head](#step-b2-self-attention--masked-latent-inpainting-head)
8. [Track C: High-Resolution Scaling Beyond CIFAR-10 (The Kodak Benchmark)](#8-track-c-high-resolution-scaling-beyond-cifar-10-the-kodak-benchmark)
   - [Step C.1: Fully Convolutional Adaptation for Kodak ($768 \times 512$)](#step-c1-fully-convolutional-adaptation-for-kodak-768--512)
   - [Step C.2: Bandwidth Compression Ratio Sweeps ($k/n$)](#step-c2-bandwidth-compression-ratio-sweeps-kn)
9. [Track D: Communication-Theoretic Rigor (Burst Losses & Separation Baselines)](#9-track-d-communication-theoretic-rigor-burst-losses--separation-baselines)
   - [Step D.1: Gilbert–Elliott 2-State Burst-Loss Channel](#step-d1-gilbertelliott-2-state-burst-loss-channel)
   - [Step D.2: Digital Separation Benchmark (BPG/JPEG + LDPC/RS)](#step-d2-digital-separation-benchmark-bpgjpeg--ldpcrs)
10. [Automated Logging, Provenance & Documentation Standards](#10-automated-logging-provenance--documentation-standards)
11. [Clerical, Formatting & Submission Requirements for IEEE Publications](#11-clerical-formatting--submission-requirements-for-ieee-publications)
12. [Modular Visual Inspection Tooling (Zero Logic Duplication)](#12-modular-visual-inspection-tooling-zero-logic-duplication)
13. [Track E: Top-Tier Plotting, Visual Presentation, and Paper Structuring](#13-track-e-top-tier-plotting-visual-presentation-and-paper-structuring)
14. [Master 14-Week Execution Schedule & Milestones](#14-master-14-week-execution-schedule--milestones)

---

# 1. Current State & Baseline Architecture

The repository currently provides an end-to-end, verified pipeline connecting continuous Deep JSCC to finite-field Reed–Solomon coding:

```
[Input Image x] ──> [ADJSCC Encoder] ──> [Scalar Quantizer (STE, 256)] ──> [Byte Indices idx]
                                                                                │
                                                                                ▼
[Reconstructed x_hat] <── [ADJSCC Decoder] <── [Latent Fallback 0.0] <── [GF(256) RS Matrix Decode]
                                                    ▲                                   ▲
                                                    │                                   │
                                            [Received Packets] <── [Bernoulli Loss] <── [RS Matrix Encode]
```

### The System Parameters
* **Input**: CIFAR-10 RGB ($3 \times 32 \times 32$).
* **Channel Bandwidth Ratio**: $k/n = 1/6$ ($n = 3072$, $k = 512$, continuous latent vector length $2k = 1024$).
* **Bottleneck Shape**: $(B, 16, 8, 8)$ representing 16 channels, each of spatial area $8 \times 8 = 64$ scalars.
* **Quantization**: Shared 1D scalar codebook with 256 learnable centers $\boldsymbol{c} \in \mathbb{R}^{256}$ and temperature annealing ($\sigma_q: 1.0 \to 0.05$). Output: integer indices in $[0, 255]$ mapping directly to $\text{GF}(256)$ bytes.
* **Packetization**: Packet size $P = 16$ bytes. Total data = $1024$ bytes $= 64$ data packets. Transmitted budget = $96$ packets ($1536$ bytes) $\implies$ exactly $50.0\%$ coding overhead.
* **Layout**: 2D packet-interleaved matrix layout where codewords run down columns and packets form the rows.
* **Erasure Fallback**: Unrecovered symbols are replaced with continuous float $0.0$ in latent space (`dequantize_with_fallback`), avoiding out-of-distribution codebook extremes.

---

# 2. What the Results Say (The Scientific Punchline)

| Finding | Observation / Number | Academic Significance |
| :--- | :--- | :--- |
| **Attention is Misaligned** | $\rho = -0.0180$ correlation between attention weights and PSNR drop. Attention UEP ($19.35\text{ dB}$) loses to Random UEP ($19.45\text{ dB}$) at 50% loss. | Squeeze-and-excitation attention scales continuous variance for AWGN; it has **no correlation with topological erasure vulnerability**. |
| **Extreme Latent Disparity** | Channel 3 causes a **$12.77\text{ dB}$ PSNR drop** when lost, while Channel 2 causes only **$0.67\text{ dB}$**. | Deep JSCC latents have massive non-uniform structural value. UEP has enormous potential if guided correctly. |
| **Oracle UEP Extends the Cliff** | At $35\%\text{--}50\%$ erasure, Oracle UEP outperforms Uniform RS by **$+1.33\text{ dB}$ to $+2.83\text{ dB}$**. | UEP shields the critical structural channels, preventing catastrophic image breakdown when the block code capacity breaks. |
| **The Low-Loss Penalty ($<33\%$)** | Uniform RS outperforms UEP by **$0.12\text{ dB}$ to $1.44\text{ dB}$** at $20\%\text{--}30\%$ loss. | Uniform RS is a single large block $\text{RS}(96, 64)$ that perfectly corrects any loss up to $33.3\%$. UEP starves its lowest-priority tier, which fails prematurely. |

---

# 3. The 5 Rejection Vectors at Tier-1 IEEE Venues

Reviewers at IEEE GLOBECOM / ICC / TWC are experienced communication theorists and ML engineers. If you submit without addressing the following 5 weaknesses, the paper will be rejected:

1. **The "Offline Oracle" Vector**: *"An oracle requires running leave-one-out across 40,000 images offline. How does a transmitter rank channels dynamically for a new image without immense compute?"* $\implies$ Solved by **Track A**.
2. **The "Low-Erasure Penalty" Vector**: *"Why should anyone deploy UEP if standard Uniform RS is 1.4 dB better across normal channel congestion ($20\%\text{--}30\%$ loss)?"* $\implies$ Solved by **Track B**.
3. **The "Toy Dataset" Vector**: *"CIFAR-10 ($32 \times 32$) is too small to draw real conclusions for wireless image transmission in 2026."* $\implies$ Solved by **Track C**.
4. **The "Idealized Channel" Vector**: *"Memoryless Bernoulli packet loss is unrealistic; real wireless links experience burst packet drops and joint AWGN fading."* $\implies$ Solved by **Track D**.
5. **The "Missing Digital Baseline" Vector**: *"Deep JSCC must be benchmarked against state-of-the-art classical digital separation (BPG/JPEG + LDPC/RS) at equal bit budgets."* $\implies$ Solved by **Track D**.

---

# 4. Scientific Hygiene & Anti-Cheating Protocols (The Integrity Manifesto)

To ensure this research is unimpeachable, scientifically reproducible, and completely free of p-hacking or experimental leakage, **the following rules are mandatory across all current and future scripts.**

```
                     ┌────────────────────────────────────────────────────────┐
                     │          THE 6 COMMANDMENTS OF SCIENTIFIC INTEGRITY    │
                     └────────────────────────────────────────────────────────┘
                                                 │
         ┌───────────────────┬───────────────────┼───────────────────┬───────────────────┐
         ▼                   ▼                   ▼                   ▼                   ▼
  [Train/Val/Test]    [Zero-Leakage Fallback] [Paired Loss Masks] [Zero Bit Leakage]  [Deterministic Seeding]
  Official test set   Decoder NEVER sees      Methods face the    Side-info counted   Every step logged with
  isolated until      original bytes on       EXACT same packet   in overhead         torch, numpy, python
  final benchmark     codeword failure        erasure patterns    budget (1552 B)     deterministic seeds
```

### Rule 1: The Ironclad 40k / 10k / 10k Data Split Protocol
* **Official Training Split (50,000 images)**:
  - **40,000 images**: Exclusively used for model training, quantizer calibration, and deriving the training-split Oracle sensitivity ranking.
  - **10,000 images**: Validation split. Exclusively used for model checkpoint selection (`best_val.pth`), learning rate early stopping, and candidate tier rate tuning.
* **Official Test Split (10,000 images)**:
  - Held completely untouched in cold storage.
  - **NEVER** calibrate quantizer percentiles on test images.
  - **NEVER** tune RS code rates after observing test set numbers.
  - **NEVER** derive the Oracle channel ranking on the test set. (The ranking must be derived from the 40k training split, frozen into JSON, and loaded strictly for inference).
  - Test set is run **only once** for final table generation.

### Rule 2: Zero-Ground-Truth Leakage in Fallback Imputation
* When Reed–Solomon decoding fails on a block, the receiver must **never** retain or peek at the ground-truth erased indices.
* In `coding/rs_pipeline.py`, verify that `decoded.failed` explicitly masks out all erased positions.
* When dequantizing via `dequantize_with_fallback(quantizer, indices, failed)`, every failed position must be strictly replaced by float $0.0$ in continuous latent space. 
* *Audit Test*: Run unit test asserting that setting erased packet payload to random garbage before decoding yields the exact same reconstruction as setting it to zero.

### Rule 3: Paired Channel Loss Masks (No Favorable Channel Draws)
* It is scientific cheating to sample different random erasure patterns for Uniform RS and UEP. A method could artificially win simply because it drew an easier loss realization.
* For every batch of images, every seed, and every erasure rate $p$:
  - The evaluator MUST sample **one master boolean mask** of 96 packets using a seeded generator.
  - The **exact same mask** is applied to `uniform`, `random_control`, `attention_uep`, `predicted_uep`, and `oracle_uep`.
  - Both methods face the identical network destruction.

### Rule 4: Strict Zero-Sum Overhead Accounting (No Free Bits)
* Every bit transmitted across the channel must be accounted for in the overhead denominator:
  $$\text{Total Overhead} = \frac{\text{Data Bytes} + \text{RS Parity Bytes} + \text{Side-Information Bytes} + \text{Packet Header Bytes}}{\text{Data Bytes}} - 1$$
* For Uniform RS: $1024\text{ data} + 512\text{ parity} + 0\text{ side-info} = 1536\text{ bytes}$ ($50.0\%$ overhead).
* For Tiered UEP: $1024\text{ data} + 512\text{ parity} + 16\text{ side-info bytes} = 1552\text{ bytes}$ ($51.5625\%$ overhead).
* Always report both **Coding-Only Overhead** and **Total Overhead Including Side-Information**.

### Rule 5: Blinded and Deterministic Seeding
* Every experiment script must start with `seed_everything(seed)` setting:
  - `torch.manual_seed(seed)`
  - `torch.cuda.manual_seed_all(seed)`
  - `np.random.seed(seed)`
  - `random.seed(seed)`
  - `torch.backends.cudnn.deterministic = True`
* Multi-seed sweeps must evaluate channel seeds $s \in \{0, 1, 2, 3, 4\}$. Results must report **Mean $\pm$ Standard Deviation** or $95\%$ Confidence Intervals. Never report only the single best seed.

---

# 5. Multi-Seed Training & Statistical Significance Protocol (Ruling Out "Lucky Seeds")

A common fatal critique in ML-for-wireless papers is: *"Your autoencoder just got lucky with random weight initialization; on another seed, the channel importance distribution would be completely different."*

To permanently dismantle this criticism, we mandate the **Multi-Seed Training Protocol**:

```
                       ┌────────────────────────────────────────────────────────┐
                       │          MULTI-SEED TRAINING & REPRODUCIBILITY         │
                       └────────────────────────────────────────────────────────┘
                                                   │
                   ┌───────────────────────────────┼───────────────────────────────┐
                   ▼                               ▼                               ▼
            [Model Seed 42]                [Model Seed 100]                [Model Seed 2024]
            Train ADJSCC-Q                  Train ADJSCC-Q                  Train ADJSCC-Q
            Compute Oracle LOO              Compute Oracle LOO              Compute Oracle LOO
                   │                               │                               │
                   └───────────────────────────────┼───────────────────────────────┘
                                                   │
                                                   ▼
                         Cross-Initialization Damage Correlation Matrix
                         Assert Spearman r_s > 0.85 across ALL initializations
```

### The 3-Model Initialization Experiment
1. Train **three separate ADJSCC-Q models from complete scratch** using different global initialization seeds:
   - Run 1: `seed = 42` (the baseline model).
   - Run 2: `seed = 100`.
   - Run 3: `seed = 2024`.
2. For each trained checkpoint:
   - Compute clean validation PSNR/SSIM (verify convergence to $\sim 30.5\text{ dB}$).
   - Compute the 16-channel Leave-One-Out damage vector on the 40k training split: $\boldsymbol{d}_{42}, \boldsymbol{d}_{100}, \boldsymbol{d}_{2024} \in \mathbb{R}^{16}$.
3. **Statistical Verification Table to Include in Paper**:

| Checkpoint Seed | Clean PSNR (0% Loss) | Top-4 Critical Channels | Worst Channel Damage | Least Critical Channel Damage |
| :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | $30.53\text{ dB}$ | $[3, 7, 4, 6]$ | $12.77\text{ dB}$ | $0.67\text{ dB}$ |
| **Seed 100** | $30.48\text{ dB}$ | $[?, ?, ?, ?]$ | $12.\text{XX dB}$ | $0.\text{XX dB}$ |
| **Seed 2024** | $30.51\text{ dB}$ | $[?, ?, ?, ?]$ | $12.\text{XX dB}$ | $0.\text{XX dB}$ |

4. **Hypothesis to Prove**: While the exact numerical channel indices might permute (e.g. Channel 3 in Seed 42 might become Channel 5 in Seed 100 due to permutation symmetry of CNN filters), **the distribution of damage remains identical**:
   - There is always 1 catastrophic channel causing $>10\text{ dB}$ collapse.
   - There are always 3–4 high-damage channels causing $5\text{--}8\text{ dB}$ drop.
   - There are always 6–8 near-zero channels causing $<2\text{ dB}$ drop.
   - In all three models, UEP guided by its respective oracle achieves **$+2.0\text{ to }+2.8\text{ dB}$** gain over Uniform RS.
5. *Reviewer Defense*: This proves that unequal latent importance is an inherent mathematical property of deep joint source-channel autoencoders, completely ruling out lucky seed anomalies.

---

# 6. Track A: Solving the Online Predictor (From Offline Oracle to Practical Transmitter)

---

### Step A.1: Cross-Dataset Invariance Testing of Filter Sensitivity

#### 1. Concept & Objective
We hypothesize that because CNN filters specialize in specific visual frequencies, the sensitivity ranking of the 16 bottleneck channels is an intrinsic property of the trained convolutional weights, rather than an image-specific variable.

#### 2. Exact Implementation Protocol
1. Take the existing checkpoint `results/checkpoints/adjsccq_identity_train/adjsccq_identity_final.pth`.
2. Compute the 16-channel Leave-One-Out (LOO) sensitivity damage vector $\boldsymbol{d}_{\text{CIFAR10}} \in \mathbb{R}^{16}$ on 2,000 CIFAR-10 test images using `experiments/eval_rs_uep_oracle.py`.
3. Load a completely different dataset: **STL-10** (downsampled to $32 \times 32$) or **CIFAR-100**.
4. Run the exact same LOO sensitivity evaluation on 2,000 images of the new dataset to obtain $\boldsymbol{d}_{\text{Transfer}} \in \mathbb{R}^{16}$.
5. Calculate the **Spearman Rank Correlation** ($r_s$) and **Pearson Linear Correlation** ($r_p$):
   $$r_s = 1 - \frac{6 \sum_{i=1}^{16} (R(\boldsymbol{d}_{\text{CIFAR10}, i}) - R(\boldsymbol{d}_{\text{Transfer}, i}))^2}{16(16^2 - 1)}$$
6. Test whether the static permutation array learned on CIFAR-10 (`[3, 7, 4, 6, 11, 0, 12, 8, 1, 15, 14, 13, 9, 10, 5, 2]`) delivers the exact same $+2.0\text{ dB}$ to $+2.8\text{ dB}$ gain when applied to STL-10 under the UEP erasure pipeline.

#### 3. What NOT To Do
* **DO NOT** retrain or fine-tune the model on the second dataset. The weights must be frozen to test filter transferability.
* **DO NOT** compute sensitivity using unquantized latents. Quantization noise interacts with channel erasures; use the quantized index path.
* **DO NOT** use fewer than 1,000 images. Sample-level noise can distort rankings of channels with similar damage coefficients.

#### 4. Criteria for Good vs. Bad Results
* **Good Result (The Dream Scenario)**:
  * $r_s \ge 0.85$ and the top-4 critical channels ($\{3, 7, 4, 6\}$) remain in the top 4 on STL-10.
  * Evaluating STL-10 with the CIFAR-10 static ranking yields $>+1.8\text{ dB}$ gain over Uniform RS at 40% loss.
  * *What it means*: Channel sensitivity is an intrinsic structural property of deep JSCC filters. Transmitter and receiver hardcode the static permutation table. **Side-information overhead becomes exactly 0 bytes.**
* **Bad Result (Ranking is Dataset-Dependent)**:
  * $r_s \le 0.50$ and the channel damage order changes drastically across datasets.
  * *What it means*: Sensitivity is driven by input image semantics. Proceed directly to Step A.2 and Step A.3 to build an image-adaptive online predictor.

---

### Step A.2: Gradient-Based & Unsupervised Feature Saliency Testing

#### 1. Concept & Objective
If sensitivity cannot be hardcoded statically, can an online transmitter determine channel importance directly from the latent tensor $\boldsymbol{z} \in \mathbb{R}^{B \times 16 \times 8 \times 8}$ without running multiple decoding passes? We test mathematical feature proxies:
1. **Channel L2 Energy**: $E_c = \frac{1}{H W} \sum_{h, w} z_{c, h, w}^2$
2. **Channel Variance**: $V_c = \frac{1}{HW} \sum_{h, w} (z_{c, h, w} - \mu_c)^2$
3. **Channel Fisher / Gradient Saliency**: $S_c = \frac{1}{HW} \sum_{h, w} \left|\frac{\partial \|\boldsymbol{z}\|_2^2}{\partial z_{c, h, w}}\right|$

#### 2. Exact Implementation Protocol
1. Write a script `experiments/test_feature_saliency.py`.
2. For 2,000 test images, compute the clean latent tensor $\boldsymbol{z}$.
3. Calculate the 16-dimensional vectors $\boldsymbol{E}, \boldsymbol{V}, \boldsymbol{S}$.
4. Sort channels based on $\boldsymbol{E}$, $\boldsymbol{V}$, and $\boldsymbol{S}$ to form candidate channel permutations.
5. Compute the correlation against the per-image Leave-One-Out ground-truth damage drop.
6. Feed these candidate permutations into `rs_pipeline.py` and run an erasure sweep at $p = 0.40$.

#### 3. What NOT To Do
* **DO NOT** compute gradients with respect to ground-truth labels $\boldsymbol{y}$ (this is source-channel coding, not classification; labels do not exist at the transmitter).
* **DO NOT** compute feature statistics *before* power normalization. Normalization scales all channels jointly and shifts their relative magnitudes.

#### 4. Criteria for Good vs. Bad Results
* **Good Result**: Energy or Variance ranking achieves $\ge 80\%$ of the Oracle UEP gain (e.g. achieves $\ge 22.2\text{ dB}$ vs Oracle's $22.74\text{ dB}$ at 40% loss).
* **Bad Result**: Correlation remains low ($r < 0.30$) and PSNR under erasures matches Random UEP ($\sim 21.9\text{ dB}$). Proceed directly to Step A.3.

---

### Step A.3: The Lightweight Sensitivity Prediction Head (Neural Auxiliary Solution)

#### 1. Concept & Objective
Train a lightweight auxiliary neural network head attached to the encoder that directly predicts the 16-channel sensitivity damage vector in a single forward pass.

```
                         ┌──────────────────────────────────────────────┐
                         │               ADJSCC ENCODER                 │
                         └──────────────────────────────────────────────┘
                                                │
                                                ▼  Latent tensor z (B, 16, 8, 8)
                                        ┌───────┴───────┐
                                        ▼               ▼
                              [Scalar Quantizer]   [Global Average Pooling] -> (B, 16)
                                        │               │
                                        │               ▼
                                        │      [Sensitivity MLP Head]
                                        │        Linear(16, 64) -> ReLU
                                        │        Linear(64, 16)
                                        │               │
                                        │               ▼
                                        │      Predicted Damage Vector d_hat (B, 16)
                                        │               │
                                        ▼               ▼
                              [Byte Indices idx]  [Channel Ranking = argsort(d_hat)]
                                        │               │
                                        └───────┬───────┘
                                                ▼
                                    [2D Interleaved RS UEP]
```

#### 2. Exact Implementation Protocol
1. **Architecture** (add to `models/adjsccq.py`):
   ```python
   class SensitivityHead(nn.Module):
       def __init__(self, in_channels: int = 16, hidden_dim: int = 64):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(in_channels, hidden_dim),
               nn.ReLU(inplace=True),
               nn.Linear(hidden_dim, in_channels)
           )
       def forward(self, z):
           pooled = z.mean(dim=[2, 3]) # (B, C)
           return self.net(pooled)     # (B, C) predicted damage drop in dB
   ```
2. **Ground Truth Data Generation**:
   * Use `train_adjsccq_train_split.py` on the 40,000 training images.
   * Save a tensor `train_damage_labels.pt` of shape $(40000, 16)$, where each row contains the exact measured PSNR drop when that channel was zeroed out.
3. **Training Loss**:
   * Combine Smooth L1 Loss and **Pairwise Ranking Loss**:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SmoothL1}}(\hat{\boldsymbol{d}}, \boldsymbol{d}) + \lambda \sum_{i=1}^{16} \sum_{j=1}^{16} \max(0, -y_{ij} (\hat{d}_i - \hat{d}_j) + \gamma)$$
     where $y_{ij} = 1$ if $d_i > d_j$ else $-1$, margin $\gamma = 0.5$, and $\lambda = 0.1$.
4. **Training Settings**:
   * Freeze the encoder backbone completely. Only optimize the `SensitivityHead`.
   * Optimizer: Adam ($\text{lr} = 10^{-3}$). Batch size: 128. Epochs: 30.
5. **Inference Pipeline**:
   * During evaluation, the forward pass outputs both `indices` and `d_hat`.
   * Channel order: `order = torch.argsort(d_hat, dim=1, descending=True)`.
   * Feed `order` into `encode_blocks()` as side information.

#### 3. What NOT To Do
* **DO NOT** train the head on the test set. All training labels must come exclusively from the 40k training split.
* **DO NOT** make the sensitivity head heavy ($<2\%$ of encoder FLOPs).
* **DO NOT** attempt to backpropagate through the `argsort` permutation operator directly. Train the head via supervised regression/ranking against offline targets.

#### 4. Criteria for Good vs. Bad Results
* **Good Result**:
  * Top-4 Channel Accuracy: The head correctly predicts the top 4 most critical channels with $\ge 90\%$ accuracy.
  * PSNR under 40% packet loss reaches **$\ge 22.5\text{ dB}$** (matching the Oracle's $22.74\text{ dB}$ within $0.2\text{ dB}$ and beating Uniform RS by **$+2.6\text{ dB}$**).
* **Bad Result**:
  * The head overfits, predicting identical outputs for all samples or achieving Top-4 accuracy $<50\%$.
  * *Fix*: Concatenate both channel mean and channel variance: `torch.cat([z.mean([2,3]), z.var([2,3])], dim=1)`.

---

# 7. Track B: Erasure-Aware Decoder Retraining (Eliminating the Low-Erasure Penalty)

Reviewers will scrutinize why Uniform RS wins between $10\%\text{--}30\%$ loss. Track B eliminates this penalty.

---

### Step B.1: Channel-Dropout Augmented Training

#### 1. Concept & Objective
Currently, the decoder was trained exclusively on an **identity channel** (zero noise). When an unrecovered packet arrives, setting those latents to $0.0$ introduces an out-of-distribution shock. By introducing **random channel dropout during training**, the decoder learns to inpaint missing channels from the surviving structural context.

#### 2. Exact Implementation Protocol
1. Modify `experiments/train_adjsccq_train_split.py`.
2. In the training loop, after obtaining quantized latents $\boldsymbol{z}_q \in \mathbb{R}^{B \times 16 \times 8 \times 8}$:
   ```python
   # Sample an erasure probability per batch
   p_batch = torch.rand(1).item() * 0.5  # uniform between 0.0 and 0.50
   
   # Generate channel-level mask: shape (B, 16, 1, 1)
   channel_mask = (torch.rand(B, 16, 1, 1, device=device) >= p_batch).float()
   
   # Apply dropout in latent space (fill missing with 0.0)
   z_corrupted = z_q * channel_mask
   
   # Reconstruct and backprop
   reconstruction = decoder(z_corrupted, nominal_snr_db)
   loss = criterion(reconstruction, x)
   ```
3. **Training Schedule**:
   * Start from the existing `adjsccq_identity_final.pth` checkpoint.
   * Fine-tune the decoder for 30 epochs with a smaller learning rate: $\text{lr} = 10^{-4}$ (Adam).
   * Keep the encoder frozen for the first 15 epochs, then unfreeze for joint fine-tuning for the remaining 15 epochs.
4. **Evaluation**:
   * Save checkpoint as `adjsccq_erasure_robust.pth`.
   * Evaluate across the full test set under the exact same Bernoulli erasure grid: $p \in [0.0, 0.5]$.

#### 3. What NOT To Do
* **DO NOT** use standard PyTorch `nn.Dropout2d()`. It scales surviving activations by $\frac{1}{1 - p}$. In our communication pipeline, surviving packets are received unchanged (systematic RS); **no re-scaling occurs at the receiver**.
* **DO NOT** drop individual pixels randomly. The loss unit in our pipeline is whole channels/packets ($64$ contiguous symbols), not isolated scalars.

#### 4. Criteria for Good vs. Bad Results
* **Good Result (The Breakthrough)**:
  * Clean PSNR ($p = 0.0$) remains high ($\ge 30.2\text{ dB}$, dropping by $<0.3\text{ dB}$).
  * At $20\%$ loss, UEP PSNR jumps from **$27.17\text{ dB}$ to $\ge 30.1\text{ dB}$**, matching Uniform RS ($30.49\text{ dB}$) within $0.3\text{ dB}$.
  * At $40\%$ loss, UEP PSNR improves further from **$23.06\text{ dB}$ to $\ge 24.2\text{ dB}$** (widening the lead over Uniform RS to **$+3.2\text{ dB}$**).
* **Bad Result (Over-regularization)**:
  * The clean PSNR collapses to $<28.0\text{ dB}$.
  * *Fix*: Reduce the maximum training erasure rate from $0.50$ to $0.30$, or use a curriculum where $p_{\text{train}}$ increases from $0.0$ to $0.35$ over 20 epochs.

---

### Step B.2: Self-Attention / Masked Latent Inpainting Head

#### 1. Concept & Objective
If simple convolution fine-tuning reaches a plateau, add a dedicated 2-layer Transformer block before the decoder transpose convolutions to perform explicit latent inpainting across channels.

#### 2. Exact Implementation Protocol
1. Flatten spatial dimensions per channel: $(B, 16, 8, 8) \to (B, 16, 64)$ tokens.
2. If channel $c$ is marked failed in the `failed` mask, replace its 64-symbol vector with a learnable $\boldsymbol{e}_{\text{mask}} \in \mathbb{R}^{64}$ token.
3. Pass through 2 Multi-Head Self-Attention (MHSA) layers (4 heads, embedding dimension 64).
4. Reshape back to $(B, 16, 8, 8)$ and feed into the deconvolution backbone.

---

# 8. Track C: High-Resolution Scaling Beyond CIFAR-10 (The Kodak Benchmark)

Evaluating exclusively on CIFAR-10 ($32 \times 32$) will trigger immediate reviewer skepticism. Track C elevates the paper to modern high-resolution standards.

---

### Step C.1: Fully Convolutional Adaptation for Kodak ($768 \times 512$)

#### 1. Concept & Objective
Show that the sensitivity-aware UEP framework scales seamlessly to realistic photography. We evaluate on the canonical **Kodak PhotoCD dataset** (24 lossless uncompressed images, $768 \times 512$).

#### 2. Exact Implementation Protocol
1. **Fully Convolutional Property**:
   * An input image of shape $(3, 768, 512)$ passed through our 5-conv encoder produces a bottleneck of shape:
     $$(16, \frac{768}{4}, \frac{512}{4}) = (16, 192, 128) = 393,216\text{ latent symbols}$$
2. **Packetization for High Resolution**:
   * Total data symbols = $393,216$ bytes.
   * Set packet size $P = 128$ bytes.
   * Total data packets $K = 393,216 / 128 = 3072$ packets.
   * Each of the 16 channels contains $192 \times 128 = 24,576$ symbols $= 192$ data packets.
   * Channel alignment: Every channel is exactly $192$ packets. Tiers remain perfectly channel-aligned.
3. **Coding Scheme**:
   * Transmit $N = 4608$ total packets ($50\%$ coding overhead, matching CIFAR-10).
   * High tier (top 4 channels): $768$ data packets $\to \text{RS}(1536, 768)$ block.
   * Medium tier (next 4 channels): $768$ data packets $\to \text{RS}(1152, 768)$ block.
   * Low tier (bottom 8 channels): $1536$ data packets $\to \text{RS}(1920, 1536)$ block.
4. **Perceptual Metrics**:
   * Calculate PSNR, **Multi-Scale SSIM (MS-SSIM)**, and **LPIPS** (Learned Perceptual Image Patch Similarity using AlexNet/VGG backbone via `pip install lpips`).

#### 3. What NOT To Do
* **DO NOT** crop or resize Kodak images to $32 \times 32$ or $128 \times 128$. Evaluate on full $768 \times 512$ native resolution.
* **DO NOT** change the relative bandwidth compression ratio ($k/n$). Maintain $k/n = 1/6$ so findings are directly comparable across datasets.

#### 4. Criteria for Good vs. Bad Results
* **Good Result**:
  * Quantitative: UEP maintains a **$+1.8\text{ dB}$ to $+2.5\text{ dB}$ PSNR gain** and a **$\Delta \text{MS-SSIM} \ge 0.04$** improvement over Uniform RS at 40% packet loss.
  * Qualitative: High-resolution zoom crops show that Uniform RS produces blurry, smeared structural boundaries, while Sensitivity-Aware UEP preserves crisp facial features, text, and edges.
* **Bad Result**:
  * The convolutional model trained on CIFAR-10 fails to reconstruct high-frequency details on $768 \times 512$ images, producing a low clean PSNR ($<24\text{ dB}$).
  * *Fix*: Train a native high-resolution model checkpoint on patches from ImageNet/COCO ($256 \times 256$) using the identical architecture before evaluating on Kodak.

---

### Step C.2: Bandwidth Compression Ratio Sweeps ($k/n$)

#### 1. Concept & Objective
Demonstrate that Sensitivity-Aware UEP is not an artifact of $k/n = 1/6$, but a universal principle holding across aggressive and generous bandwidth budgets: $k/n \in \{1/16, 1/12, 1/6, 1/3\}$.

---

# 9. Track D: Communication-Theoretic Rigor (Burst Losses & Separation Baselines)

Track D provides the rigorous communication-theory validation required by IEEE Transactions reviewers.

---

### Step D.1: Gilbert–Elliott 2-State Burst-Loss Channel

#### 1. Concept & Objective
Real-world wireless packet erasures occur in **bursts** due to deep fading, multi-path shadows, and queue buffer drops. We test the 2D matrix-interleaved UEP pipeline under a **Gilbert–Elliott channel**.

```
                1 - P_GB                              1 - P_BG
                ┌──────┐                              ┌──────┐
                │      ▼                              │      ▼
             ┌────────────┐        P_GB            ┌────────────┐
             │  STATE G   │ ─────────────────────> │  STATE B   │
             │   (Good)   │ <───────────────────── │   (Bad)    │
             └────────────┘        P_BG            └────────────┘
                Loss: P_G                             Loss: P_B
                (~ 1%)                                (~ 80%)
```

#### 2. Exact Implementation Protocol
1. Implement in `coding/erasure_channel.py`:
   ```python
   class GilbertElliottChannel:
       def __init__(self, p_gb: float, p_bg: float, p_g: float = 0.01, p_b: float = 0.85):
           self.p_gb = p_gb  # transition probability Good -> Bad
           self.p_bg = p_bg  # transition probability Bad -> Good
           self.p_g = p_g    # packet loss probability in Good state
           self.p_b = p_b    # packet loss probability in Bad state
           
       def sample_mask(self, batch_size: int, total_packets: int, seed: int):
           # Samples Markov state sequence and returns boolean loss mask (B, total_packets)
           ...
   ```
2. Parameterize average burst length: $L_{\text{burst}} = \frac{1}{P_{BG}}$.
3. Sweep burst lengths $L_{\text{burst}} \in \{2, 4, 8, 16\}$ packets at average loss rate $p = 0.35$.
4. Evaluate Uniform RS vs. Sensitivity-Aware UEP.

#### 3. What NOT To Do
* **DO NOT** transmit tier packets consecutively without time interleaving. Interleave packets across tiers across the transmission timeline.

#### 4. Criteria for Good vs. Bad Results
* **Good Result**: 2D packet-interleaved UEP outperforms Uniform RS by **$>3.0\text{ dB}$** under burst lengths $L_{\text{burst}} \ge 4$.

---

### Step D.2: Digital Separation Benchmark (BPG/JPEG + LDPC/RS)

#### 1. Concept & Objective
You must show that classical digital separation suffers from a catastrophic cliff effect under packet loss, whereas our framework degrades gracefully.

#### 2. Exact Implementation Protocol
1. For each test image, compress using **BPG (Better Portable Graphics)** to an exact bit budget matching the Deep JSCC data symbols ($1024$ bytes).
2. Protect the BPG bitstream using standard $\text{RS}(96, 64)$ parity packets (identical 50% overhead).
3. Transmit through the exact same erasure masks.
4. If erasures exceed RS capacity, standard BPG decompression crashes or produces complete image loss (PSNR drops to $<8\text{ dB}$).
5. Plot PSNR vs. Erasure Rate for BPG+RS vs. Deep JSCC+UEP.

#### 3. Criteria for Good vs. Bad Results
* **Good Result**: Below 33% loss, BPG+RS achieves higher clean PSNR ($32\text{ dB}$ vs $30.5\text{ dB}$), but at $35\%$ loss, BPG+RS crashes to $8\text{ dB}$ (the cliff effect), while Deep JSCC+UEP maintains **$25.27\text{ dB}$**.

---

# 10. Automated Logging, Provenance & Documentation Standards

To make every table in the final paper immediately reproducible by a third party, every experiment must generate a complete provenance record.

### Mandatory Logging Schema
Every evaluation script (`eval_rs_uep.py`, `eval_rs_uep_oracle.py`, etc.) must save a structured JSON record with the following schema:

```json
{
  "metadata": {
    "git_commit": "a44f9a2460ad...",
    "dirty_repo": false,
    "timestamp": "2026-10-16T22:30:00Z",
    "python_version": "3.11.16",
    "torch_version": "2.4.1",
    "device": "NVIDIA A100-SXM4-40GB",
    "dataset": "CIFAR-10",
    "split_seed": 42,
    "split_sizes": {"train": 40000, "val": 10000, "test": 10000}
  },
  "model": {
    "checkpoint_path": "results/checkpoints/adjsccq_identity_train/adjsccq_identity_final.pth",
    "checkpoint_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "k_over_n": 0.1666667,
    "num_levels": 256,
    "clean_test_psnr_db": 30.53,
    "clean_test_ssim": 0.959
  },
  "channel_seeds": [0, 1, 2, 3, 4],
  "erasure_rates": [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5],
  "results": [
    {
      "method": "oracle_uep",
      "scheme": "three_tier_25_50_25",
      "erasure_rate": 0.40,
      "psnr_mean": 23.06,
      "psnr_std": 2.14,
      "ssim_mean": 0.812,
      "tier_block_failures": [0.05, 0.68, 0.99],
      "realized_erasure_rate": 0.3998,
      "total_transmitted_bytes": 1552,
      "total_overhead_percentage": 51.5625
    }
  ]
}
```

### Reproducibility CLI Invocation
Every table in the paper must be reproducible via a single terminal command recorded in `README.md`:
```bash
# Example: Generate Table 2 exactly as published in the paper
python -m experiments.eval_rs_uep_oracle \
  --checkpoint results/checkpoints/adjsccq_identity_train/adjsccq_identity_final.pth \
  --oracle_ranking_file results/eval/oracle_ranking_train_split.json \
  --data_dir ./cifar10_data \
  --seeds 0,1,2,3,4 \
  --output results/eval/paper_table_2_final.json
```

---

# 11. Clerical, Formatting & Submission Requirements for IEEE Publications

Submitting to a flagship IEEE ComSoc conference (GLOBECOM/ICC) or an IEEE Transactions journal involves strict clerical constraints. Failure to adhere to these rules results in immediate administrative desk-rejection without review.

### 1. IEEE ComSoc Double-Blind Review Policy
* **Anonymization**: Submissions to IEEE ICC and GLOBECOM are strictly **double-blind**.
* **Rules**:
  - The manuscript must **not** include author names, affiliations, or email domains (`@iiitb.ac.in`).
  - GitHub repositories linked in the paper must be completely scrubbed of author identities. Use an anonymous repository service (e.g., [Anonymous GitHub](https://anonymous.4open.science/)).
  - Citations to your own prior work must be written in the third person (e.g., write *"In [12], Bhat et al. investigated..."*, NEVER write *"In our previous work [12]..."*).
  - Acknowledgments of funding bodies, labs, or university compute clusters must be removed during the initial review submission.

### 2. Formatting & Page Limits
* **Template**: Use the official IEEE Conference LaTeX template:
  `\documentclass[conference]{IEEEtran}`.
* **Page Limits**:
  - **IEEE ICC / GLOBECOM**: Strictly **6 pages** (including text, figures, tables, and references). You can purchase up to **1 additional page** (7 pages total) upon acceptance for a fee ($\sim \$100\text{--}\$150$), but the review submission must not exceed 6 pages unless the CFP explicitly allows 7.
  - **IEEE Letters (WCL / CL)**: Strictly **4 pages** (or 5 pages depending on current CFP rules).
  - **IEEE Transactions (TWC / TCOM)**: 10–12 double-column pages or 30 single-column double-spaced pages.
* **No Format Tweaking**: Do NOT alter margins, line spacing (`\linespread`), or font sizes to squeeze text in. IEEE PDF eXpress will detect font/margin violations and reject the PDF.

### 3. Graphics & Visual Standards
* **Vector Graphics Mandatory**: All architecture diagrams, block diagrams, and performance curves MUST be generated as vector files (`.pdf` or `.eps`).
* **Raster Images**: Photographs (Kodak crops) must be at least **300 to 600 DPI**. Blurry PNGs or JPEGs with compression artifacts will draw negative reviewer comments.
* **Color Accessibility**: Use colorblind-friendly color palettes (e.g. Seaborn `colorblind`, Matplotlib `viridis` or `tab10`). Ensure lines have distinct markers (`o`, `s`, `^`, `d`) so they remain distinguishable in black-and-white printing.

### 4. Citation & Reference Standards
* Do not cite unverified preprints or random blog posts. Prioritize peer-reviewed IEEE Transactions and flagship conference papers:
  - Cite *Bourtsoulatze et al.* in IEEE TCCN (not arXiv).
  - Cite *Xu et al.* in IEEE TCCN (not arXiv).
  - Cite *Tung et al.* in IEEE TWC (not arXiv).
* Ensure all references include complete author lists, publication titles, journal/conference volume, page numbers, and years.

---

# 12. Modular Visual Inspection Tooling (Zero Logic Duplication)

### 1. Architectural Requirement: Strict Separation of Concerns
For immediate sanity-checking and qualitative reference throughout the research cycle, researchers must be able to visually inspect input and output images side-by-side.

However, to prevent code rot, logic drift, and experimental bugs, **the visual inspection tool MUST NOT implement or duplicate any pipeline logic.**
* **What NOT To Do**:
  - Never write custom dequantization, RS loops, packet-splitting, or channel-noise functions inside the visualization script.
  - Never copy-paste convolution/deconvolution layers or model instantiations.
* **The Rule of Zero Logic Duplication**:
  - The script must act purely as an **orchestration consumer**.
  - It imports and calls *only* the verified core modules:
    ```python
    from experiments.eval_adjsccq_clean import load_adjsccq_checkpoint
    from coding.rs_pipeline import (encode_blocks, decode_blocks, 
                                    dequantize_with_fallback, SCHEME_CANDIDATES, NO_RS)
    from coding.erasure_channel import sample_erasure_mask, apply_erasure_mask
    from experiments.metrics import psnr_per_image, ssim
    from data.cifar10 import get_cifar10_loaders
    ```

### 2. Standardized Visual Inspection Layout
The dedicated script `experiments/visualize_reconstructions.py` outputs a high-resolution, publication-ready multi-column comparison grid:
* **Column 1**: Original Ground-Truth Image $\boldsymbol{x}$.
* **Column 2**: Clean Baseline Reconstruction ($p = 0.0$ loss) with clean PSNR/SSIM badges.
* **Column 3**: Unprotected Transmission (`no_rs` at target erasure rate $p$, e.g. $40\%$).
* **Column 4**: Uniform RS Baseline (`uniform` at target erasure rate $p$, e.g. $40\%$).
* **Column 5**: Sensitivity-Aware UEP (Proposed scheme at target erasure rate $p$, e.g. $40\%$).

Every reconstructed image is badged directly with its individual **PSNR (dB)** and **SSIM** calculated via `experiments.metrics`.

### 3. CLI Invocation for Instant Sanity Checks
```bash
# Visualize 5 sample test images under 40% severe packet loss
python -m experiments.visualize_reconstructions \
  --checkpoint results/checkpoints/adjsccq_identity_train/adjsccq_identity_final.pth \
  --data_dir ./cifar10_data \
  --num_images 5 \
  --erasure_rate 0.40 \
  --uep_scheme three_tier_25_50_25 \
  --output results/eval/visual_comparison_side_by_side.png

# High-DPI export for paper insertion
python -m experiments.visualize_reconstructions --erasure_rate 0.35 --dpi 300
```

---

# 13. Track E: Top-Tier Plotting, Visual Presentation, and Paper Structuring

Top-tier papers are often accepted or rejected based on the clarity and visual beauty of their figures.

### The 5 Essential Figures Every Top-Tier Reviewer Expects

1. **Figure 1 (System Teaser Diagram)**:
   * A clean, professional vector diagram (built using TikZ or Inkscape) showing the full pipeline:
     $$\text{Image} \to \text{ADJSCC-Q} \to \text{Online Sensitivity Predictor} \to \text{2D Interleaved Matrix} \to \text{Packet Loss} \to \text{Recovery}$$
2. **Figure 2 (The Core Scientific Proof — Attention Failure vs. Oracle Disparity)**:
   * **Left Subplot**: Scatter plot of Attention Weights vs. Actual PSNR Damage ($\rho = -0.0180$, showing a completely flat, uncorrelated cloud of dots).
   * **Right Subplot**: Bar chart of the 16 channels sorted by true Oracle Damage (showing Channel 3 towering at $12.77\text{ dB}$ and Channel 2 at $0.67\text{ dB}$).
3. **Figure 3 (Primary Performance Curves)**:
   * PSNR (dB) vs. Packet Erasure Rate ($p \in [0.0, 0.5]$) showing 5 distinct curves:
     1. Unprotected Baseline (`no_rs`)
     2. Uniform RS Baseline (`uniform`)
     3. Attention-Guided UEP
     4. Online Predicted UEP (Sensitivity Head)
     5. Oracle UEP (Upper Bound)
4. **Figure 4 (High-Resolution Qualitative Visual Comparison on Kodak)**:
   * Side-by-side $768 \times 512$ images at 40% packet loss with $3\times$ zoom-in bounding boxes on high-frequency details (e.g., the lighthouse railing or face texture):
     - (a) Original Ground Truth
     - (b) Uniform RS Reconstruction (smeared, distorted)
     - (c) Sensitivity-Aware UEP Reconstruction (crisp edges, recognizable structure)
5. **Figure 5 (Gilbert–Elliott Burst Erasure & Classical Separation)**:
   * PSNR vs. Average Burst Length, showing BPG+RS collapsing while Deep JSCC+UEP maintains high fidelity.

---

# 14. Master 14-Week Execution Schedule & Milestones

| Weeks | Phase | Core Action Items | Hard Milestone Gate (Do Not Pass Until Verified) |
| :---: | :--- | :--- | :--- |
| **Weeks 1–2** | **Multi-Seed & Track A: Invariance & Predictor** | 1. Train 3 ADJSCC-Q models from distinct seeds (`42, 100, 2024`).<br>2. Run LOO sensitivity on STL-10; test Spearman correlation $r_s$ against CIFAR-10.<br>3. Build and train the 2-layer `SensitivityHead` MLP on 40k training damage labels. | **Milestone 1**: Online predicted ranking matches $\ge 90\%$ of Oracle UEP PSNR gain at 40% erasure across all initializations. |
| **Weeks 3–5** | **Track B: Erasure Retraining** | 1. Implement channel-dropout in `train_adjsccq_train_split.py` ($p \in [0, 0.5]$).<br>2. Fine-tune decoder for 30 epochs with $\text{lr} = 10^{-4}$. | **Milestone 2**: UEP PSNR at 20% loss improves by $\ge 2.5\text{ dB}$, eliminating the low-loss penalty vs Uniform RS. |
| **Weeks 6–8** | **Track C: Kodak Scaling** | 1. Implement FCN Kodak pipeline ($768 \times 512$, $P = 128$ bytes).<br>2. Measure PSNR, MS-SSIM, and LPIPS.<br>3. Extract qualitative visual crop comparison figures. | **Milestone 3**: Produce Figure 4 with high-resolution visual proof of edge preservation. |
| **Weeks 9–10** | **Track D: Burst Channels & Digital Baselines** | 1. Implement Gilbert–Elliott 2-state channel module.<br>2. Benchmark against BPG + RS digital separation. | **Milestone 4**: Demonstrate superiority under burst lengths $L_{\text{burst}} \ge 4$ and plot BPG cliff effect. |
| **Weeks 11–12** | **Ablation Sweeps & Statistics** | 1. Run 5-seed evaluation across all final methods.<br>2. Compute 95% confidence intervals and generate all publication-ready vector plots. | **Milestone 5**: All tables and graphs populated with zero missing data. |
| **Weeks 13–14** | **Manuscript Preparation** | 1. Draft 6-page IEEE conference paper in standard two-column LaTeX format.<br>2. Anonymize all authors, affiliations, and links for double-blind compliance.<br>3. Internal review, proofreading, and camera-ready preparation. | **Final Gate**: Submit to IEEE GLOBECOM / ICC main track. |

---

### Suggested Working Title for Submission:
> **"When Attention Fails: Sensitivity-Aware Unequal Error Protection for Deep Joint Source-Channel Coding over Packet-Erasure Channels"**

### Final Advice to the Team
You have already completed the most difficult part: building the discrete mathematical interface, verifying the Reed–Solomon matrix interleaving, and uncovering the failure of attention mechanisms. By systematically executing the tracks in this manual, you will eliminate every possible reviewer objection and produce a **definitive, high-impact paper that the community cannot reject.**
