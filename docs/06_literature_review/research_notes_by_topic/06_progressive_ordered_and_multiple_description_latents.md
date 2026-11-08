# Progressive, Successive-Refinement, Ordered and Nested-Dropout Representations in Deep JSCC, Learned Compression and Semantic Communication

Scope note: our paper does NOT train the latent to be ordered. We measure naturally unequal channel importance by leave-one-out (top-4 of 16 channels carry about 50% of erasure damage) and protect important channels more with Reed-Solomon UEP over packet-erasure channels. These notes ask what exists on deliberately creating an importance ordering, and whether any of it already pairs ordering with unequal erasure protection.

Verification legend: "Verified" means I opened the arXiv abstract or full-text page, or a publisher or proceedings page, in this session. "Search-verified" means the paper's existence and metadata were confirmed through search-result listings (arXiv, CVF, NeurIPS, IEEE Xplore) but I did not read the full text. Anything from memory alone is flagged as such.

## Q1: Does any work combine an ordered/progressive Deep JSCC representation with unequal erasure protection over packet channels?

### Takeaway
No paper I found combines all three of: a trained ordered/progressive Deep JSCC latent, a packet-erasure channel, and explicit erasure-code (RS/FEC) unequal protection. Two 2026 preprints come close from different directions, and they are the main novelty threats.
- **Esfahanizadeh et al. (Nokia Bell Labs, arXiv 2601.20707).** They train a JSCC autoencoder over block erasures with an *increasing erasure-probability vector*, which induces an ordering among packets. They leave UEP to the network (for example per-block MCS) and use no FEC.
- **Zhan et al. ISFR (arXiv 2604.00595).** They use *nested dropout* to create the ordering and then apply semantic-importance UEP. Their channel is bit-error/AWGN, their UEP levers are modulation, power and channel matching, and they use no erasure codes.
- **Closest RS/FEC-over-packets precedent: Grad-FEC (ICIP 2023).** It applies FEC selectively to important deep-feature packets in collaborative intelligence (split inference). It is not JSCC and the features are not trained to be ordered.

### Cited Findings

#### (a) Deep JSCC with successive refinement / layered / variable-rate transmission

- **Kurka & Gündüz, "Successive Refinement of Images with Deep Joint Source-Channel Coding"** (conference version, arXiv 1903.06333). It presents three progressive deep JSCC strategies built on convolutional autoencoders, with different complexity-performance trade-offs — [ar5iv 1903.06333](https://ar5iv.labs.arxiv.org/html/1903.06333); [Imperial PDF](https://t4awstest.cc.ic.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/Kurka_succesiverefinement.pdf); also summarised in [arXiv 2009.12480](https://arxiv.org/abs/2009.12480). (The venue, IEEE SPAWC 2019, is from memory and was not verified this session.) Closeness: **related**. AWGN, no erasure codes.
- **D. B. Kurka & D. Gündüz, "Bandwidth-Agile Image Transmission with Deep Joint Source-Channel Coding" (DeepJSCC-l), IEEE Trans. Wireless Communications, 2021; arXiv 2009.12480** — [arXiv](https://arxiv.org/abs/2009.12480); [Imperial Spiral](https://spiral.imperial.ac.uk/handle/10044/1/92917); [journal PDF](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/layered_JSCC_journal.pdf).
  - **Two scenarios.** In successive refinement (SR), layers arrive in order. In multiple descriptions (MD), layers are independent and any subset can be decoded (2^L − 1 decoders) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2009.12480).
  - **How the ordering arises.** The loss is the average MSE over the L progressive decoders (or over all 2^L − 1 subset decoders for MD). The base layer is pushed to carry coarse content. A single-decoder variant is trained by randomly zeroing *trailing* blocks, which works like tail/nested dropout — [ar5iv](https://ar5iv.labs.arxiv.org/html/2009.12480).
  - **Channel.** AWGN and slow Rayleigh. Layer loss is not modelled as a channel effect; the receiver chooses which subset to use. There are no erasure codes and no UEP — [ar5iv](https://ar5iv.labs.arxiv.org/html/2009.12480).
  - **Closeness: related.** It is the canonical answer to "why not train it ordered?", but it was never evaluated with packet erasures plus FEC.
- **M. Yang & H.-S. Kim, "Deep Joint Source-Channel Coding for Wireless Image Transmission with Adaptive Rate Control", ICASSP 2022; arXiv 2110.04456** — [arXiv](https://arxiv.org/abs/2110.04456v1); [ICASSP 2022 video page](https://rc.signalprocessingsociety.org/conferences/icassp-2022/spsicassp22vid0116).
  - **Method.** A policy network picks how much of the feature map to send, made differentiable with Gumbel-Softmax. One model covers many rates and sends less at high SNR or for simple images. For each target rate it performs about as well as a model trained for that rate. Code: Dynamic_JSCC — [arXiv listing / search summary](https://arxiv.org/pdf/2110.04456).
  - **Ordering.** Implicit: selecting feature groups requires prefix-style importance.
  - **Closeness: related/distant.** AWGN only, no erasure, no UEP.
- **J. Dai, S. Wang, K. Tan, Z. Si, X. Qin, K. Niu, P. Zhang, "Nonlinear Transform Source-Channel Coding for Semantic Communications" (NTSCC), IEEE JSAC 40(8):2300–2316, Aug. 2022, DOI 10.1109/JSAC.2022.3180802; arXiv 2112.10961** — [arXiv](https://arxiv.org/abs/2112.10961v3); [catalog record](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c359t-d6af9430f7c079b482eaa8c01e5050c8a9c5c6f1fae0b8e0d911555df41ac9b3).
  - **Method.** A learned analysis transform plus a hyperprior entropy model drive *variable-length* JSCC. Each latent element gets channel bandwidth according to its estimated entropy.
  - **Ordering.** Rate allocation is content-adaptive and per element. It is not a nested or progressive order.
  - **Closeness: distant.** It is relevant only because it uses side information (the hyperprior) to allocate resources unequally. Follow-up: Improved NTSCC, [arXiv 2303.14637](https://www.arxiv.org/abs/2303.14637v3).
- **Esfahanizadeh, Fayaz, Du, Viswanathan, "Block Erasure-Aware Semantic Multimedia Compression via JSCC Autoencoder", arXiv 2601.20707, Jan 2026 (submitted to IEEE Trans. Multimedia)** — [arXiv HTML](https://arxiv.org/html/2601.20707v1); [arXiv abs](https://arxiv.org/abs/2601.20707). **Verified.**
  - **Latent and erasure model.** The latent is split into K blocks: on CIFAR-10, K=8 blocks of 2×8×8, a 16×8×8 latent. Each block goes through a block-erasure channel, and an erased block is replaced by −1 so the decoder can tell it is missing.
  - **How the ordering arises.** The training erasure vector is increasing (ε1 < … < εK). With ε1 = 0, earlier blocks become more important; a steeper slope behaves more like successive refinement. A uniform ε gives no ordering.
  - **No FEC.** UEP is "delegated to the network", e.g. 5G MCS per block. In experiments, UEP is simulated by scaling the training erasure vector, ε_test = min(a·ε_train, 1).
  - **Results.** It beats per-layer SR at 10% erasure, using one encoder/decoder pair for any number of received blocks. It was also integrated into DVC for video (UVG).
  - **Citations.** It cites Kurka & Gündüz and the Koike-Akino tail-drop "stochastic bottleneck". It does not cite multiple description coding or nested dropout by name.
  - **Closeness: CLOSE, the main novelty threat.** It is packet/block-erasure Deep JSCC with importance created by training and unequal protection per block. The differences from us are that they train the ordering in rather than measure natural importance, and they use no RS/erasure codes, no leave-one-out measurement and no explicit parity allocation.
- **Fayaz, Esfahanizadeh, Mortaheb, Du, Viswanathan, "Towards Robust Semantic Video Transmission over Block Erasure Channels", arXiv 2607.07823, July 2026 (accepted, IEEE VTC-Fall 2026)** — [arXiv](https://arxiv.org/abs/2607.07823). Neural video JSCC over block erasures, comparing spatial-domain and feature-domain partitioning, with a two-level semantic-guided non-uniform erasure strategy. It does not mention FEC or UEP by name. Closeness: **related**.
- **Mortaheb, Esfahanizadeh, Du, Viswanathan, "Semantic-Aware Neural Video Codec for Error-Resilient Low-Latency Transmission", arXiv 2609.16279, Sept 2026** — [arXiv HTML](https://arxiv.org/html/2609.16279); [abs](https://arxiv.org/abs/2609.16279). **Verified.**
  - **Method.** Built on DCVC-RT. The latent is split into 8 channel slices, and packets are assigned to priority streams over a *multi-level packet-erasure channel* (ε1 ≤ … ≤ εL). Assignment uses learnable logits that combine semantic score and slice index.
  - **Decoding and side information.** The entropy model makes every packet independently decodable. The assignment matrix is sent as 384 bits per frame of side information.
  - **No codes.** No FEC or RS; protection comes only from stream assignment.
  - **Closeness: close (video).** It uses priority streams over erasure channels and explicit side information, but no erasure codes.
- **Tung, Esfahanizadeh, Du, Viswanathan, "Multi-level Reliability Interface for Semantic Communications over Wireless Networks", arXiv 2407.05487, July 2024** — [arXiv](https://arxiv.org/pdf/2407.05487). The source mapper is trained over several bit pipes, each with a different bit-flip probability. The authors suggest packetizing source bits by reliability level, with a header carrying the level index. This is the conceptual parent of the Bell Labs block-erasure line. Closeness: **related** (bit errors, not erasures). Only the abstract and summaries were read.

#### (b) Nested dropout / ordered representations, and their use in communication

- **O. Rippel, M. A. Gelbart, R. P. Adams, "Learning Ordered Representations with Nested Dropout", ICML 2014; arXiv 1402.0915** — [arXiv PDF](https://arxiv.org/pdf/1402.0915); [Princeton PDF](https://www.cs.princeton.edu/~rpa/pubs/rippel2014nested.pdf). Nested dropout removes coherent nested sets of units: a truncation index is sampled and every unit after it is dropped. This enforces identifiability, and for linear autoencoders it is exactly equivalent to PCA — [search summary of paper](https://www.researchgate.net/publication/260062234_Learning_Ordered_Representations_with_Nested_Dropout). Closeness: **related (foundational)**.
- **Variational Nested Dropout, arXiv 2101.11353.** It learns the nested-dropout rate and ordering from data instead of fixing it as a hyperparameter — [arXiv](https://arxiv.org/abs/2101.11353). Authors and venue were not verified this session.
- **"Ordering Dimensions with Nested Dropout Normalizing Flows", arXiv 2006.08777** — [arXiv](https://arxiv.org/pdf/2006.08777). Nested dropout applied to normalizing flows. Authors and venue were not verified.
- **A. Kusupati et al., "Matryoshka Representation Learning", NeurIPS 2022; arXiv 2205.13147** — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract.html).
  - **Method.** It jointly optimizes O(log d) nested prefixes of one embedding. A prefix of the first m dimensions is about as accurate as an independently trained m-dimensional model, with no extra inference cost, and embeddings are up to 14× smaller at the same ImageNet-1K accuracy — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract.html); [Google Research](https://research.google/pubs/matryoshka-representation-learning/).
  - **Use in comms.** I found **no** Matryoshka-based Deep JSCC or semantic-comms paper (one targeted search returned none).
  - **Closeness: distant.**
- **E. W. Staley & J. Markowitz, "Triangular Dropout: Variable Network Width without Retraining", arXiv 2205.01235, ICML 2022 DyNN workshop** — [arXiv](https://arxiv.org/abs/2205.01235v1); [workshop PDF](https://dynn-icml2022.github.io/spapers/paper_3.pdf). A lower-triangular binary mask on fully connected layers lets a trained network be cut down to a narrower dense one. Tested on autoencoders, VGG19/ImageNet and RL policies, with small accuracy loss. Closeness: **distant**.
- **T. Koike-Akino & Y. Wang, "Stochastic Bottleneck: Rateless Auto-Encoder for Flexible Dimensionality Reduction", IEEE ISIT 2020; arXiv 2005.02870 (MERL TR2020-075)** — [arXiv](https://arxiv.org/pdf/2005.02870); [MERL](https://merl.com/publications/TR2020-075). The "TailDrop" regularizer gives higher priority to the upper latent units, so the least-principal dimensions can be dropped at test time (MNIST/CIFAR-10). Follow-up: "Rateless Autoencoder Codes: Trading off Decoding Delay and Reliability", [arXiv 2301.12231](https://arxiv.org/pdf/2301.12231), which applies dropout-matched training to channel coding. Closeness: **related**. These are cited by Esfahanizadeh et al. as the closest dropout-based antecedent.
- **Zhan, Cao, Zhu, Pappas, Qin, Feng, "Toward Robust Semantic Communications: Proactive Importance-Ordered Restructuring for Enhanced Unequal Error Protection" (ISFR), arXiv 2604.00595, April 2026** — [arXiv abs](https://arxiv.org/abs/2604.00595); [HTML](https://arxiv.org/html/2604.00595). **Verified.**
  - **How the ordering arises.** Nested dropout (Rippel et al.) gives the i-th feature retention probability (N+1−i)/N. A "BER-matching" module trains front features at the lowest sampled BERs, with N BERs drawn from [0, 0.2] and sorted.
  - **Two-stage training.** Stage 1 is analog transmission with nested dropout. Stage 2 adds learned scalar quantization and BER matching.
  - **Importance is still measured after training.** Each feature is masked on a validation set, the PSNR drop is recorded, and the normalized weights are stored as a prior. This is essentially leave-one-out, even though the representation is ordered by construction.
  - **Channel.** BSC in training; block-fading AWGN with per-feature SNR at inference.
  - **UEP levers.** Greedy channel matching, feature truncation, a QAM order in {4, 16, 64}, and KKT power allocation. **No channel coding and no erasure model.**
  - **Baselines.** Five "uniform-importance" baselines include feature masking (UI-M, Liu et al.), gradients (UI-G), gradients plus correlation (UI-GC), Taylor approximation (UI-EA) and equal protection (UI-EEP, Bourtsoulatze).
  - **Closeness: CLOSE.** It is the nested-dropout plus UEP counterpart of our work, over bit-error/fading channels rather than packet erasure with RS.

#### (c) Progressive learned image compression

- **Y. Lu, Y. Zhu, Y. Yang, A. Said, T. S. Cohen, "Progressive Neural Image Compression with Nested Quantization and Latent Ordering" (PLONQ), arXiv 2102.02913, Feb 2021** — [arXiv](https://arxiv.org/abs/2102.02913).
  - **Method.** Nested quantization grids refine all latents from coarse to fine. Elements are ordered "in the rate-distortion sense", per element rather than per channel. It beats SPIHT.
  - **Venue.** The arXiv page lists none. I recall ICIP 2021 from memory, which is unverified.
  - **Channels.** No erasure or channel evaluation.
  - **Closeness: related.**
- **J.-H. Lee, S. Jeon, K. P. Choi, Y. Park, C.-S. Kim, "DPICT: Deep Progressive Image Compression Using Trit-Planes", CVPR 2022, DOI 10.1109/CVPR52688.2022.01564; arXiv 2112.06334** — [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_DPICT_Deep_Progressive_Image_Compression_Using_Trit-Planes_CVPR_2022_paper.html); [arXiv](https://arxiv.org/abs/2112.06334).
  - **Method.** The latent is converted to ternary digits and sent trit-plane by trit-plane, in order of significance. Within a plane, trits are sorted by R-D priority, and a postprocessing net helps at low rates. The authors describe it as the first learned codec with fine granular scalability.
  - **Channels.** No erasure or UEP evaluation found.
  - **Closeness: related.**
- **DeepHQ: "Learned Hierarchical Quantizer for Progressive Deep Image Coding", arXiv 2408.12150; ACM DOI 10.1145/3773994** — [arXiv](https://arxiv.org/pdf/2408.12150); [ACM DL](https://dl.acm.org/doi/10.1145/3773994). Learned nested quantization intervals in place of handcrafted ones (search-verified only). Closeness: **related**.
- **Bound-Conditioned Latent Inference for Progressive Image Compression (BLI), arXiv 2610.30966, 2026** — [arXiv HTML](https://arxiv.org/html/2610.30966). It reports the best R-D among progressive approaches over most of the bitrate range against FTIC-DPICT (search snippet only). Closeness: **distant**.
- **A. Hojjat, J. Haberer, O. Landsiedel, "ProgDTD: Progressive Learned Image Compression with Double-Tail-Drop Training", CVPR Workshops 2023, pp. 1130–1139, DOI 10.1109/CVPRW59228.2023.00120; arXiv 2305.02145** — [CVPRW / mlanthology](https://mlanthology.org/cvprw/2023/hojjat2023cvprw-progdtd); [arXiv](https://arxiv.org/abs/2305.02145); [code](https://github.com/ds-kiel/ProgDTD). Tail-drop (nested-dropout-style) training on both the latent and the hyperprior bottlenecks of a Ballé hyperprior model, so channels are ordered by priority and sent in index order. This is the closest analogue in learned compression to "channel-wise progressive". Closeness: **related**. No erasure or UEP evaluation.
- **Wei, Zhou, Shi, Wang, Lu, "Every Packet Counts: Dispersing Information for Loss-Resilient Learned Image Compression", ACM MM 2026; arXiv 2608.11096** — [arXiv HTML](https://arxiv.org/html/2608.11096). **Verified.**
  - **Approach.** The opposite philosophy to ordering. Inter-Channel Redistribution and strided Interleaved Channel Grouping make every packet carry *comparable* information, with packets of at most 1500 bytes.
  - **Loss models.** Uniform loss at 5/10/20%, and Gilbert-Elliott at about 15% average loss.
  - **Results.** +1.84 dB over LossResilientLIC at 20% loss (Kodak low rate: 27.211 vs 25.578 dB).
  - **FEC.** None in the main experiments, and the hyperprior is assumed loss-free. Appendix C analyses RS(12,8) protection of the hyperprior only, at about 7% overhead.
  - **Related works it cites.** LossResilientLIC (Sha et al., AAAI 2025), ResiComp (Wang et al., IEEE TCSVT 2025), GRACE (Cheng et al., USENIX NSDI 2024) and ProgDTD.
  - **Closeness: related.** It is a competing "equalize importance" design, the opposite of UEP.
- **ResiTok (Z. Liu, Y. Ma, R. Tafazolli, arXiv 2505.01870, 2025)** — [arXiv](https://arxiv.org/abs/2505.01870). Hierarchical 1-D tokens (key tokens plus detail tokens) give progressive encoding. "Zero-out" training simulates token loss, and channel-adaptive coding and modulation allocate resources. Resi-VidTok ([arXiv 2510.25002](https://fugumt.com/fugumt/paper_check/2510.25002v1_enmode), metadata via mirror) extends it to video under high packet loss. Closeness: **related**. It is an ordered token hierarchy with loss-aware training, but I found no RS/FEC UEP.

#### (d) Importance-aware feature protection under packet loss / neural multiple description coding

- **K. Uyanik, S. F. Yeganli, I. V. Bajić, "Grad-FEC: Unequal Loss Protection of Deep Features in Collaborative Intelligence", IEEE ICIP 2023; arXiv 2307.01846** — [arXiv](https://arxiv.org/abs/2307.01846); [PDF](https://arxiv.org/pdf/2307.01846). A feature-importance estimator ranks feature packets. FEC protects a percentage of the important packets, and the least important packets are sacrificed to pay for the parity packets (FEC_A_B scheme). The search snippet describes systematic RS across packets; the abstract says only "FEC". Closeness: **CLOSE in mechanism** (importance-ranked FEC over packet loss, the same idea as ours), but the setting is split inference, not Deep JSCC reconstruction, and the features are not trained to be ordered.
- **Bo, Zhang, Tao, He, Huang, "Lightweight Generative Image Semantic Communication over Packet Erasure Channels" (LGSemCom), arXiv 2609.06989, Sept 2026** — [arXiv](https://arxiv.org/abs/2609.06989). Packet-level generative JSCC with adversarial inpainting of erased regions and an erasure-aware loss weighting. No UEP, FEC or ordering. Closeness: **related** (a competing "conceal instead of protect" approach).
- **Classical baseline: Mohr, Riskin, Ladner, "Unequal loss protection: graceful degradation of image quality over packet erasure channels through forward error correction"** — [IEEE Xplore 848236](https://ieeexplore.ieee.org/document/848236/). RS-style FEC with unequal protection for a progressive (SPIHT) bitstream over packet erasure. With exponential loss at a 20% mean rate and 0.2 bpp on Lenna, good quality survives even 40% packet loss, and the algorithm applies to any progressive codec — [search summary](https://www.researchgate.net/publication/3234351_Unequal_loss_protection_Graceful_degradation_of_image_quality_over_packet_erasure_channels_through_forward_error_correction). (IEEE JSAC 2000 is from memory.) This is the classical precedent for "progressive source + unequal erasure protection".

### Inferences
- The "ordered Deep JSCC + UEP" combination is being actively explored in 2026, through Esfahanizadeh et al. 2601.20707 and Zhan et al. 2604.00595. Neither uses erasure codes or measures naturally emerging importance in an *unordered* model, so our specific combination (natural importance, leave-one-out measurement, RS parity allocation over packet erasure) still appears unclaimed. Both must be cited and discussed explicitly.
- **Rebuttal point.** ISFR trains an ordered representation with nested dropout and *still* measures per-feature importance afterwards by masking and stores it as a prior. Ordering by construction fixes the *rank* but not the *magnitude* of importance, and UEP parity allocation needs magnitudes. Some measurement step (or an assumed decay profile) remains, so measuring importance is not made redundant by ordered training.
- **Side information.** If our leave-one-out ranking is computed once per model (not per image), it is a static prior, like ISFR's stored weights, so side information is negligible either way. Per-image or content-adaptive schemes do pay a cost: Mortaheb et al. send a 384-bit/frame assignment matrix, and NTSCC sends a hyperprior.
- **Unreviewed design.** Training the ordering in (Esfahanizadeh's increasing ε, or nested dropout) and then applying RS UEP is a natural ablation reviewers may ask for. No paper reports it, so it is a gap we could fill rather than a prior-art threat.
- **Philosophically opposite designs.** "Every Packet Counts" and DeepJSCC-l MD mode equalize information across packets, the opposite of UEP. They are useful as contrasting baselines.

### Gaps
- I could not open the full text of Grad-FEC to confirm the exact code (RS vs other), the importance estimator (gradient-based per the name), or numbers.
- I did not verify whether DPICT, PLONQ, CTC (context-based trit-plane coding, Jeon et al., recalled from memory and not verified) or DeepHQ were ever evaluated with packet loss and FEC. No such evaluation surfaced.
- The publication venues of PLONQ (ICIP 2021?) and of Kurka & Gündüz's successive-refinement conference paper (SPAWC 2019?) are from memory and need checking.
- No Matryoshka-based JSCC or semantic-communication paper was found. Absence of evidence from limited searching is not proof that none exists.

## Q2: What is the cost of imposing ordering (rate-distortion loss) reported in these papers?

### Takeaway
Reported costs are small but rarely quantified cleanly. Most papers say "negligible" or "slightly lower PSNR" without a dB number. Where numbers exist, they show a modest penalty at good channel conditions:
- ISFR is about 1.3% below the best unordered baseline in PSNR at high SNR.
- In DeepJSCC-l MD mode, the full decoder falls below a jointly trained single decoder.
- ProgDTD has slightly lower PSNR than its non-progressive base model.
- Steeper ordering profiles hurt full-bandwidth quality (Esfahanizadeh).

### Cited Findings
- **DeepJSCC-l (SR mode).** Progressive transmission has "negligible losses in the end-to-end performance compared with a single transmission". The curves stay close to the single-layer baseline, but the text gives no explicit dB loss figure. With L=2 and k_i/n = 1/12, the second layer adds about 2–3 dB. With L=5 the per-layer gains shrink, and first-layer performance barely changes as L grows — [arXiv 2009.12480](https://arxiv.org/abs/2009.12480); [ar5iv](https://ar5iv.labs.arxiv.org/html/2009.12480).
- **DeepJSCC-l (MD mode).** Decoding all layers improves a lot over single-layer decoding but "stays below a joint decoder trained only for the full bandwidth". Kodak, L=2: 26.85 dB with both layers vs about 25 dB from either layer alone. It beats a repetition-code baseline at all SNRs — [ar5iv](https://ar5iv.labs.arxiv.org/html/2009.12480).
- **Yang & Kim adaptive rate.** One adaptive model performs about as well as models trained for each specific rate — [arXiv 2110.04456](https://arxiv.org/pdf/2110.04456).
- **ISFR (nested dropout + SI-UEP).**
  - At average SNR > 10 dB on CIFAR-10, ISFR-SI-UEP PSNR is about 1.3% below the best uniform-importance baseline.
  - On DIV2K it is slightly below UI at high SNR on all metrics, and its LPIPS advantage disappears.
  - At −6 dB it gains 24.7–46.4% PSNR over the five baselines.
  - The first feature alone recovers over 80% of full performance, and importance weights span up to two orders of magnitude.
  - Note that this compares whole systems, each with its own UEP, not a clean ordered-vs-unordered ablation — [arXiv 2604.00595 HTML](https://arxiv.org/html/2604.00595).
- **Esfahanizadeh et al.**
  - For uniform training, a smaller ε gives higher PSNR when all blocks arrive.
  - Steeper non-uniform (more ordered) profiles hurt high-bandwidth quality, and the paper gives no explicit no-erasure dB penalty.
  - The redundancy cost for video is called "incremental".
  - Per-layer SR is best when only one block is received but becomes suboptimal as more blocks arrive, and it is fragile to loss of early blocks.
  - Video, non-uniform: 28.11 dB (1 block), 35.46 dB (2 blocks), 39.55 dB (4 blocks). The text also says "nearly 41 dB" with 8 blocks, a minor internal inconsistency — [arXiv 2601.20707](https://arxiv.org/html/2601.20707v1).
- **ProgDTD.** The progressive model is comparable in MS-SSIM to its non-progressive counterpart and to other progressive models, and its PSNR is slightly lower than the non-progressive baseline — [arXiv 2305.02145 / CVPRW](https://arxiv.org/abs/2305.02145).
- **Matryoshka.** Nested prefixes are about as accurate as independently trained low-dimensional models (classification, not R-D) — [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract.html).
- **Triangular Dropout.** VGG19 can be cut substantially in parameters with only small accuracy loss — [arXiv 2205.01235](https://arxiv.org/abs/2205.01235v1).
- **DPICT.** When it disables the channel-wise autoregressive entropy model of its base (FTIC), it gives up some coding efficiency to gain progressivity (from a BLI search snippet; magnitude not verified) — [BLI arXiv 2610.30966](https://arxiv.org/html/2610.30966).

### Inferences
- The literature consensus is that ordering costs "a little": a fraction of a dB, or about 1–2% PSNR, at full rate or high SNR. The cost grows with how strongly the ordering is imposed (steeper ε slope, more layers, MD vs SR). Ordering buys large gains only when channels are poor or truncated.
- For our paper, this supports a measured position. Ordered training is a valid alternative, but it trades clean-channel quality for known-by-construction importance. Our approach keeps the unconstrained model's clean-channel R-D and still gets unequal importance "for free", because natural importance is already highly skewed. ISFR's own finding that importance spans two orders of magnitude even under nested dropout also shows that ordered training *amplifies* skew; it does not create something an unordered model lacks.

### Gaps
- No paper reports a clean ablation: the same architecture trained with and without nested dropout, evaluated over packet erasure with the same RS UEP. The exact dB cost of ordering in our setting is unknown and would need our own experiment.
- I found no quantified dB penalty for DPICT, PLONQ or DeepHQ against their non-progressive counterparts in the material I read.

## Q3: Is there neural multiple description coding for Deep JSCC under packet loss that would be a competing approach to UEP?

### Takeaway
Yes, but it is thin. DeepJSCC-l (Kurka & Gündüz, TWC 2021) includes an MD mode that is described as the first practical MD-JSCC scheme, but it was evaluated over AWGN/fading with receiver-chosen subsets rather than random packet erasures. The block-erasure JSCC with *uniform* erasure probabilities from Esfahanizadeh et al. is effectively a learned MD-style code over packet erasure. Several neural MDC works exist for separation-based compression and for a 2026 semantic-comms MDC design. "Every Packet Counts" and LGSemCom are further "equalize / conceal" competitors to UEP.

### Cited Findings
- **DeepJSCC-l, MD mode.** Layers are independent and decodable in any order, with 2^L − 1 decoders trained jointly. The paper is described as the first practical scheme for the multiple-description JSCC case. Single-layer decoders perform about as well as a single-layer encoder of the same dimension. The full-subset decoder falls below a full-bandwidth joint model but beats repetition coding — [Imperial journal PDF](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/layered_JSCC_journal.pdf); [ar5iv](https://ar5iv.labs.arxiv.org/html/2009.12480).
- **Esfahanizadeh et al.** Uniform training erasure gives no ordering: every block is equally important, which is MD-like behaviour. Uniform ε_train = 0.1 shows minimal degradation up to ε_test ≈ 0.3 on video. At 10% erasure it significantly outperforms per-layer SR — [arXiv 2601.20707](https://arxiv.org/html/2601.20707v1).
- **"Deep Multiple Description Coding by Learning Scalar Quantization", arXiv 1811.01504 (2019).** Deep MDC for image compression in which the quantizers are learned by minimizing a multiple-description compressive loss (separation-based, not JSCC) — [arXiv](https://arxiv.org/pdf/1811.01504).
- **"Deep Optimized Multiple Description Image Coding via Scalar Quantization Learning", arXiv 2001.03851.** A neural MD framework for robust transmission over lossy packet networks, with an SSIM-based loss supervising the decoded descriptions (separation-based) — [arXiv](https://arxiv.org/pdf/2001.03851).
- **Samarathunga et al., "Semantic Communication-Based Multiple Description Coding for Robust Image Transmission Over Wireless Channels", IEEE Access (2026 per repository).** End-to-end training of an encoder, a description generator and a fusion decoder, with path diversity — [Oulu repository](https://oulurepo.oulu.fi/handle/10024/64607). Only repository metadata was seen; I did not read the content or the erasure model.
- **Thesis "Deep Multiple Description Coding for Semantic Communication", DiVA portal** — [DiVA PDF](https://www.diva-portal.org/smash/get/diva2:1713315/FULLTEXT01.pdf). Per the search snippet, it studies multiple encoders over a packet-level erasure channel with erasure probability p_i per channel. The fetch failed (connection refused), so author, institution and details are **unverified**.
- **Competing "equalize" approach: "Every Packet Counts" (ACM MM 2026).** Interleaving and redistribution give packets equal information, which beats LossResilientLIC by 1.84 dB at 20% loss without FEC — [arXiv 2608.11096](https://arxiv.org/html/2608.11096).
- **Competing "conceal" approach: LGSemCom (arXiv 2609.06989).** Generative recovery of erased regions — [arXiv](https://arxiv.org/abs/2609.06989). Also ResiTok, which uses zero-out training on tokens — [arXiv 2505.01870](https://arxiv.org/abs/2505.01870).

### Inferences
- A reviewer could argue that MD-style training (uniform-erasure JSCC, or DeepJSCC-l MD) removes the need for UEP. The literature shows that MD trades full-reception quality for robustness: the DeepJSCC-l MD full decoder falls below the joint decoder. It also makes protection uniform, which wastes parity on unimportant channels. A head-to-head between MD-style erasure training and RS UEP on the same Deep JSCC backbone does not appear in the literature and would strengthen our paper.
- Our work sits at a different design point:
  - Ordered training (SR, nested dropout, increasing ε) builds the importance skew in.
  - MD / uniform-erasure / interleaving designs remove the skew.
  - We *exploit the skew that emerges naturally*, using classical RS erasure codes and no retraining.
  - Stating this taxonomy explicitly is the cleanest answer to the reviewer question.

### Gaps
- I did not find a peer-reviewed paper explicitly titled as "multiple description Deep JSCC" over random packet erasure with quantitative comparison to FEC/UEP. The DiVA thesis may be one, but it could not be read.
- The Samarathunga et al. IEEE Access paper's channel model (packet erasure vs path failure) and results are unverified.
- Searches were limited (about 30 tool calls). IEEE Xplore full text was not accessible, so 2025–2026 journal versions of these preprints may exist without being captured here.
