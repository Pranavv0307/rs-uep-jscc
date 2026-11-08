# Deep JSCC under packet erasure, and digital / quantized Deep JSCC with conventional channel codes

Scope note: these notes compare the literature with our paper: ADJSCC latent quantized to 256 levels (1 byte per value), 16-byte packets, Reed-Solomon (RS) over i.i.d. and Gilbert-Elliott packet erasure, zero-fill in latent space on block failure, equal protection vs importance-tiered UEP, JPEG/JPEG2000/WebP + RS baselines, erasure-aware decoder fine-tuning (channel/packet dropout), and a latent inpainting transformer. Closeness labels: near-duplicate / close / related / distant. Unless noted, details come from arXiv abstract pages or HTML full text fetched in this session (Nov 2026). I found no near-duplicate: no verified paper combines a quantized Deep JSCC latent, RS packet FEC, Gilbert-Elliott erasures, importance-tiered UEP and image-codec+RS baselines.

## Q1. Which papers evaluate Deep JSCC / semantic communication over packet-erasure / packet-loss channels, and what do they find (graceful degradation vs digital cliff)?

### Takeaway
Packet-erasure Deep JSCC is now an active niche, mostly from 2025-2026. Every verified paper reports graceful degradation when the network is trained with simulated erasures, and sharp collapse when it is not. The closest works are Esfahanizadeh et al. (2026, block-erasure JSCC with 8-bit inference quantization and UEP-by-training) and LGSemCom (Bo et al., 2026, i.i.d. plus bursty Markov packet loss, compared against codec + ideal packet-FEC envelopes). Neither uses RS on a quantized JSCC latent or ADJSCC.

### Cited Findings

**Esfahanizadeh, Fayaz, Du, Viswanathan, "Block Erasure-Aware Semantic Multimedia Compression via JSCC Autoencoder," arXiv:2601.20707, Jan 2026 (submitted to IEEE Trans. Multimedia; not yet published). Closeness: CLOSE. Moderate novelty threat to our erasure-aware training and UEP claims.**
- Content is split into packets and an unknown number may be dropped. The authors use JSCC for "graceful quality degradation as channel conditions worsen" without retransmission, and claim support for congestion control and unequal error protection — [arXiv abs](https://arxiv.org/abs/2601.20707)
- Channel: each block is erased independently with probability ε_i (one block per packet). Bursty loss is not modelled — [arXiv HTML](https://arxiv.org/html/2601.20707)
- Erased blocks are filled with −1, a value outside the normal [0,1] latent range, so the decoder can detect erasures. (We zero-fill instead.) — [arXiv HTML](https://arxiv.org/html/2601.20707)
- Training uses real-valued latents. At inference each latent element is quantized to 8 bits, "which preserves performance". This is the same byte-per-value choice as ours — [arXiv HTML](https://arxiv.org/html/2601.20707)
- Importance and UEP come from a non-uniform erasure-probability vector during training. Testing uses ε_test = min(a·ε_train, 1). No FEC of any kind (no RS, LDPC or fountain codes) is used or compared: the redundancy sits inside the JSCC code — [arXiv HTML](https://arxiv.org/html/2601.20707)
- Baselines are successive refinement, a genie bound, and averaging-fill. JPEG, BPG, WebP and DeepJSCC are not compared. Data: CIFAR-10 (images); Vimeo-90k/UVG (video) — [arXiv HTML](https://arxiv.org/html/2601.20707)
- A model trained with ε=0 "degrades sharply" as test erasures rise. The video model trained with ε=0.10 shows minimal loss up to ε_test ≈ 0.3, while baseline DVC drops to negative PSNR at high erasure — [arXiv HTML](https://arxiv.org/html/2601.20707)
- It cites earlier latent-dropout variable-rate autoencoders, including Koike-Akino & Wang, "Stochastic bottleneck: rateless auto-encoder for flexible dimensionality reduction," IEEE ISIT 2020 (tail dropout) — [arXiv HTML](https://arxiv.org/html/2601.20707)

**Fayaz, Esfahanizadeh, Mortaheb, Du, Viswanathan, "Towards Robust Semantic Video Transmission over Block Erasure Channels," arXiv:2607.07823, Jul 2026 (accepted, IEEE VTC-Fall 2026). Closeness: RELATED (video).**
- Compares spatial-domain block JSCC (uniform or two-level semantic-guided non-uniform erasure) with feature-domain latent partitioning. Spatial-domain handles random localized losses better; feature-domain is more robust to distributed erasures — [arXiv abs](https://arxiv.org/abs/2607.07823)

**Bo, Zhang, Tao, He, Huang, "Lightweight Generative Image Semantic Communication over Packet Erasure Channels" (LGSemCom), arXiv:2609.06989, Sep 2026 (preprint). Closeness: CLOSE. Threatens the novelty of "digital baselines under bursty packet loss" and of mask-token/inpainting recovery.**
- Packet-level JSCC for images. Recovery of erased regions is treated as conditional generation, with adversarial training and erasure-aware weighting (EAW) — [arXiv abs](https://arxiv.org/abs/2609.06989)
- Channel: (i) i.i.d. loss, with bits interleaved evenly across N=16 packets; training masks k% of bits with k ~ U[0, 87.5]. (ii) Bursty loss from a first-order 3-state Markov chain taken from ResiComp: moderate case PLR 13.8%, mean burst 1.69; severe case PLR 32.3%, mean burst 2.71 — [arXiv HTML](https://arxiv.org/html/2609.06989)
- Missing bits are replaced by a learnable mask token ρ (masked-token-modelling style). Training has three stages: lossless, random erasure, then adversarial with the encoder frozen — [arXiv HTML](https://arxiv.org/html/2609.06989)
- Baselines are separate source coding (JPEG AI, Cheng2020-GMM, BPG, JPEG2000) plus *ideal* packet FEC at parity ratios of 12.5–87.5%, plotted as an envelope. No specific code (RS/LDPC) is named. DeepJSCC and SwinJSCC are not packet-erasure baselines — [arXiv HTML](https://arxiv.org/html/2609.06989)
- FEC-based schemes degrade "rapidly" as PLR rises, showing "either quality degradation or total collapse". The term "digital cliff" is not used explicitly — [arXiv HTML](https://arxiv.org/html/2609.06989)

**Yao, Dai, Wang, Lu, Niu, Xu, Zhang, Zhang, "Loss-Resilient Semantic Communication over Packet-Loss Networks at Extreme-Low Bandwidth" (ResiGLC), arXiv:2608.19590, Aug 2026 (accepted to IEEE Trans. Mobile Computing). Closeness: RELATED/CLOSE (generative latent codec, BUPT group).**
- Packet loss in compact latents damages fidelity, and errors propagate across latent contexts. The fix is LM-style in-context prediction with masked learning plus progressive decoding with a multimodal semantic prior. The authors claim "graceful performance when dealing with dynamic packet losses". The abstract does not state the loss model or baselines — [arXiv abs](https://arxiv.org/abs/2608.19590)

**Yang, Ma, Liang, Shi, "Feature Partitioning and Semantic Equalization for Intrinsic Robustness in Semantic Communication under Packet Loss," arXiv:2511.16983, Nov 2025 (submitted to IEEE). Closeness: CLOSE on the packetization question. Supports our importance/partitioning analysis rather than threatening it.**
- Studies how high-dimensional features are partitioned into packets. Transformers partitioned along the channel dimension are inherently robust. A CNN baseline has imbalanced channel use and degrades severely when dominant channels are lost. The proposed Semantic Equalization Mechanism lets a CNN keep "about 85% of lossless PSNR at 40% packet loss" — [arXiv abs](https://arxiv.org/abs/2511.16983)

**Teng, Fan, Dong, Liang, Bao, Xu, Meng, Zhang, "Conquering High Packet-Loss Erasure: MoE Swin Transformer-Based Video Semantic Communication" (MSTVSC), arXiv:2508.01205, Aug 2025. Closeness: RELATED (video).**
- Semantic vectors are packetized at the upper protocol layer. A 3D CNN rebuilds missing content from the received data plus a packet-loss mask matrix, and semantic-level interleaving disperses burst losses. Reported results: MS-SSIM > 0.6 and PSNR > 20 dB at 90% packet loss — [arXiv abs](https://arxiv.org/abs/2508.01205)

**Bourtsoulatze, Kurka, Gündüz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," IEEE TCCN, 2019, arXiv:1809.01733. Closeness: DISTANT (foundational).**
- CNN encoder/decoder trained jointly over AWGN and slow Rayleigh fading. It is the canonical claim that deep JSCC "does not suffer from the cliff effect" and degrades gracefully — [arXiv](https://arxiv.org/pdf/1809.01733)

### Inferences
- The standard finding: erasure-trained JSCC degrades gracefully, an erasure-naive network collapses, and FEC/separation shows a cliff or total collapse beyond its design PLR. Our results should be framed against this. Our distinct angle is putting a real RS code on the JSCC latent, which brings its own cliff at the RS block level, and asking whether the cliff matters once latent-space zero-fill and erasure-aware fine-tuning are added.
- Gilbert-Elliott/bursty evaluation exists in LGSemCom (3-state Markov) and indirectly in MSTVSC (interleaving), so "first to evaluate bursty loss" is not a safe claim. RS + GE on a quantized ADJSCC latent appears to be unclaimed.
- Esfahanizadeh et al. already make the 8-bit inference quantization and UEP-via-non-uniform-erasure-training points. We should cite it, and distinguish our explicit RS-based UEP (parity allocation) from their implicit UEP built into training.

### Gaps
- Exact PSNR curves in LGSemCom and ResiGLC were not extracted, and I did not verify whether ResiGLC compares with codec+FEC.
- ResiComp ("Loss-resilient image compression via dual-functional masked visual token modeling," IEEE TCSVT 2025) is cited in LGSemCom, but I did not verify its own page or authors in this session.

## Q2. Which papers combine learned/quantized latents with classical channel codes (RS, LDPC, polar), or digital constellations, and compare against separation baselines (BPG/JPEG2000 + LDPC)?

### Takeaway
Digital Deep JSCC (DeepJSCC-Q, MDJCM, D²-JSCC, VQ-DeepSC, Park et al.) mostly quantizes onto modulation constellations or bits over AWGN/fading, and compares against BPG + LDPC. It does not use RS over packet erasures. The closest classical-FEC-on-learned-features work is Grad-FEC: RS-based unequal loss protection of deep features ranked by importance, for collaborative intelligence rather than JSCC. Rateless NTRSCC puts LT codes with unequal protection on learned latents. I found no paper applying importance-tiered RS to a quantized Deep JSCC latent.

### Cited Findings

**Tung, Kurka, Jankowski, Gündüz, "DeepJSCC-Q: Constellation Constrained Deep Joint Source-Channel Coding," IEEE JSAIT 2022 (journal version), arXiv:2206.08100. Conference version: "DeepJSCC-Q: Channel Input Constrained Deep Joint Source-Channel Coding," IEEE ICC 2022, pp. 3880–3885, DOI 10.1109/ICC45855.2022.9838671, arXiv:2111.13042. Closeness: RELATED (digital/constellation, AWGN, no packets).**
- Fixed channel-input alphabet with end-to-end training. With high enough modulation order it approaches unconstrained DeepJSCC. It beats BPG + LDPC separation while avoiding the cliff effect, and a learned constellation further improves performance — [arXiv 2206.08100](https://arxiv.org/pdf/2206.08100); [arXiv 2111.13042](https://arxiv.org/pdf/2111.13042)
- Note: I did not re-verify the JSAIT venue in this session. The search confirmed the ICC 2022 DOI.

**Fu, Xie, Qin, Slabaugh, Tao, "Vector Quantized Semantic Communication System" (VQ-DeepSC), IEEE Wireless Communications Letters, 2023, arXiv:2209.11519. Closeness: RELATED.**
- CNN transceiver with multi-scale features quantized against multi-scale semantic codebooks, plus PatchGAN adversarial training. More robust than BPG in digital systems and comparable to DeepJSCC on MS-SSIM. (The authors are Fu, Xie, Qin et al., not Shao or Huang.) — [arXiv](https://arxiv.org/pdf/2209.11519)

**Hu, Zhang, Qin, Cai, Yu, Li, "Robust Semantic Communications with Masked VQ-VAE Enabled Codebook," IEEE TWC, 2023, DOI 10.1109/TWC.2023.3265201, arXiv:2206.04011. Closeness: RELATED (masking and codebook robustness to semantic noise, not erasure).**
- Masked VQ-VAE with a shared discrete codebook. Only the indices of task-relevant codebook entries are sent, selected by a feature-importance module — [arXiv](https://arxiv.org/abs/2206.04011)

**Zhang, Yang, Cai, Hu, Yu, "From Analog to Digital: Multi-Order Digital Joint Coding-Modulation for Semantic Communication" (MDJCM), IEEE Trans. Commun., 2025, arXiv:2406.05437. Closeness: RELATED.**
- Adds a multi-order modulator/demodulator to NTSCC, and one model serves several modulation orders. Modulation is trained as constrained quantization approximated by scaling plus hand-crafted noise. It outperforms digital and analog JSCC — [arXiv](https://arxiv.org/html/2406.05437v2)

**Huang, Yuan, Huang, Huang, "D²-JSCC: Digital Deep Joint Source-channel Coding for Semantic Communications," arXiv:2403.07338, Mar 2024 (preprint). Closeness: RELATED.**
- Deep source coding with an adaptive density model, plus digital channel coding protecting the features. Source and channel rates are jointly allocated from a distortion model. It mitigates the cliff and leveling-off effects of separation and outperforms classic deep JSCC. The abstract does not name the channel code — [arXiv](https://arxiv.org/abs/2403.07338)

**Park, Oh, Kim, Jeon, "Joint Source-Channel Coding for Channel-Adaptive Digital Semantic Communications," arXiv:2311.08146, Nov 2023 (v2 Mar 2024). Closeness: RELATED (bit-level erasure training).**
- Models the digital link as binary symmetric *erasure* channels with sampled parameters during training. Uses an uncertainty-aware demodulator and channel-adaptive modulation order — [arXiv](https://arxiv.org/abs/2311.08146)

**Choi, Tatwawadi, Grover, Weissman, Ermon, "Neural Joint Source-Channel Coding" (NECST), ICML 2019, PMLR 97:1182–1192, arXiv:1811.07557. Closeness: RELATED (early discrete-latent JSCC over BSC/BEC).**
- Discrete VAE trained with channel noise on the latent bits. The code supports BSC or BEC channels. Baselines are WebP plus LDPC-style codes. I did not find its BEC-specific results — [arXiv](https://www.arxiv.org/abs/1811.07557v3); [GitHub](https://github.com/ermongroup/necst)

**"Grad-FEC: Unequal Loss Protection of Deep Features in Collaborative Intelligence," arXiv:2307.01846, 2023. Closeness: CLOSE in mechanism (importance-ranked RS UEP of learned features over packet loss), different application (split inference, not image reconstruction JSCC). Partial threat to "importance-tiered RS UEP of learned latents".**
- Systematic RS codes over deep-feature packets. Packets are ranked by estimated importance, and for each added FEC packet the least important packet is dropped — [arXiv](https://arxiv.org/pdf/2307.01846)
- Authors and venue were not verified in this session (search snippet only).

**Qin, Huang, Fei, Ding, Shao, Chen, "Rateless DeepJSCC for Broadcast Channels: a Rate-Distortion-Complexity Tradeoff" (NTRSCC), arXiv:2603.21616, Mar 2026. Closeness: RELATED (classical rateless code on learned latents, with unequal protection).**
- Learned nonlinear transforms combined with physical-layer LT codes and unequal protection that exploits decoder side information. Receivers trade rateless symbols collected and BP iterations against distortion — [arXiv](https://arxiv.org/abs/2603.21616)

**Singh, Arif, "Reinforcement-Learned Unequal Error Protection for Quantized Semantic Embeddings," arXiv:2601.00186, Jan 2026. Closeness: DISTANT (text embeddings, repetition coding).**
- RL-allocated per-dimension repetition coding gives +6.8% chrF and +9.3% entity preservation over uniform protection at 1 dB. The authors claim per-dimension allocation is "unattainable with conventional codes such as LDPC or Reed-Solomon" — [arXiv](https://arxiv.org/abs/2601.00186)

**Separation baselines in the Deep JSCC lineage.**
- DeepJSCC-f compares against BPG followed by a capacity-achieving code. In its appendix, failed transmissions are reconstructed as the mean value and images are coded at the largest rate R ≤ R_max — [arXiv](https://arxiv.org/pdf/1911.11174)
- An improved NTSCC study reports about 16.35% channel-bandwidth saving over BPG/VTM + 5G LDPC — [arXiv 2409.17557 survey](https://arxiv.org/html/2409.17557v1)
- SwinJSCC is better than or comparable to BPG + 5G LDPC, with faster coding — [arXiv](https://arxiv.org/pdf/2308.09361)

### Inferences
- Most digital Deep JSCC work quantizes to the *constellation* (physical layer, AWGN/fading). Our quantization is at the *packet/byte* layer with RS over erasures, a network-layer view that reviewers may see as closer to Grad-FEC and classic ULP than to DeepJSCC-Q. Cite DeepJSCC-Q, MDJCM and D²-JSCC as "digital JSCC at the PHY", and Grad-FEC plus classic RS-ULP as precedent for importance-tiered RS.
- The usual digital baseline is BPG (or VTM/JPEG AI) + LDPC or ideal FEC, not JPEG/WebP + RS. Reviewers may ask for BPG, or at least a learned codec (e.g., Cheng2020, as in LGSemCom), in our baseline set.

### Gaps
- I found no verified paper that uses polar codes with quantized Deep JSCC latents on erasure channels. Polar-code digital SemCom papers likely exist but were not verified in this session.
- I could not verify Grad-FEC's authors or venue, or whether D²-JSCC uses LDPC specifically.

## Q3. Which papers train with erasure/dropout augmentation, or use latent inpainting / masked reconstruction, to handle lost packets?

### Takeaway
Erasure-aware training is the dominant approach: random block or bit erasure in Esfahanizadeh 2026, LGSemCom 2026, Park 2023 and Tian 2025, and simulated packet loss in GRACE 2024. Masked-token or transformer inpainting of lost latents is also established: learnable mask token in LGSemCom and ResiComp, an MAE-style transformer in Tian et al., a mask-matrix 3D CNN in MSTVSC, masked in-context prediction in ResiGLC. Our decoder fine-tuning and latent-inpainting transformer are therefore incremental relative to these. The novelty must come from combining them with RS-coded quantized latents and from the comparison itself.

### Cited Findings
- **Tian et al., "Synchronous Multi-modal Semantic Communication System with Packet-level Coding," IEEE TWC 2025 (venue per citation in LGSemCom), arXiv:2408.04535. Closeness: RELATED/CLOSE.** Treats packet-level coding as a masked-autoencoder problem: a transformer predicts lost semantics, a dropout layer simulates packet drops in training, and packets follow RTP — [arXiv](https://arxiv.org/pdf/2408.04535); venue as cited in [LGSemCom HTML](https://arxiv.org/html/2609.06989)
- **Cheng et al., "GRACE: Loss-Resilient Real-Time Video through Neural Codecs," USENIX NSDI 2024, pp. 509–531, arXiv:2305.12333. Closeness: RELATED (networking-side analogue of our erasure-aware fine-tuning).** The neural video encoder and decoder are trained jointly under simulated packet losses. Against FEC baselines it gives 95% fewer undecodable frames and 90% less stall, about 3 dB SSIM over neural error concealment, and +38% MOS (240 users) — [USENIX](https://www.usenix.org/conference/nsdi24/presentation/cheng); [arXiv](https://arxiv.org/abs/2305.12333)
- LGSemCom replaces missing bits with a learnable token ρ and trains with k ~ U[0, 87.5]% random masking — [arXiv HTML](https://arxiv.org/html/2609.06989)
- Esfahanizadeh et al. simulate stochastic block erasures in training (−1 placeholder) and note that tail-only dropout (stochastic bottleneck) handles only contiguous losses — [arXiv HTML](https://arxiv.org/html/2601.20707)
- MSTVSC reconstructs from received semantics plus a packet-loss mask matrix — [arXiv](https://arxiv.org/abs/2508.01205)
- ResiGLC combines masked learning with LM-style in-context latent prediction to limit error propagation — [arXiv](https://arxiv.org/abs/2608.19590)
- Yang et al. show that architecture choice alone matters: channel-partitioned Transformers are intrinsically robust, while CNNs need equalization — [arXiv](https://arxiv.org/abs/2511.16983)
- **Kurka & Gündüz, "Bandwidth-Agile Image Transmission with Deep Joint Source-Channel Coding" (DeepJSCC-l), IEEE TWC 20(12):8081–8095, 2021, arXiv:2009.12480. Closeness: RELATED.** Layered transmission with successive-refinement and multiple-description (layers retrievable in any order) variants. The multiple-description variant is the natural JSCC precedent for "decode from whatever packets arrive" — [arXiv](https://arxiv.org/abs/2009.12480)

### Inferences
- ADJSCC is a CNN with channel-wise attention. Yang et al.'s finding that CNNs unevenly concentrate information across channels predicts that our zero-filled channel/packet losses hurt more without fine-tuning, which our dropout fine-tuning should mitigate. This is a useful mechanistic citation.
- Zero-fill (ours) vs −1 sentinel (Esfahanizadeh) vs learnable token (LGSemCom, ResiComp): a reviewer may ask why zero-fill was chosen, and an ablation would pre-empt that.

### Gaps
- I found no verified Deep JSCC image paper that, like ours, explicitly evaluates a separate post-hoc latent inpainting transformer on top of a frozen JSCC decoder. The cited works train the recovery jointly.

## Q4. Are there studies of the cliff effect of separate source-channel coding versus Deep JSCC under erasures?

### Takeaway
The cliff effect is a standard motivation, shown mostly over AWGN (Bourtsoulatze 2019, DeepJSCC-Q, D²-JSCC). Under packet erasures, the comparison against codec + FEC appears in LGSemCom (ideal-FEC envelopes collapse as PLR rises), GRACE (vs FEC for video) and Esfahanizadeh (erasure-naive models degrade sharply). No verified paper quantifies the cliff for *RS-coded quantized JSCC latents* against RS-coded JPEG/JPEG2000/WebP under GE channels, which is our niche.

### Cited Findings
- Deep JSCC "does not suffer from the cliff effect" (AWGN/fading) — [Bourtsoulatze et al. 2019](https://arxiv.org/pdf/1809.01733)
- DeepJSCC-Q beats BPG + LDPC while avoiding the cliff effect, even with constrained constellations — [arXiv](https://arxiv.org/pdf/2206.08100)
- D²-JSCC claims to mitigate the "cliff and leveling-off effects" of separation-based approaches — [arXiv](https://arxiv.org/abs/2403.07338)
- Under packet erasure, codec + ideal FEC baselines degrade rapidly with "quality degradation or total collapse", and failed decoding yields a random image — [LGSemCom HTML](https://arxiv.org/html/2609.06989)
- GRACE vs FEC: 95% fewer undecodable frames under the same congestion control — [arXiv](https://arxiv.org/abs/2305.12333)
- Classic precedent for graceful degradation via FEC: "Unequal loss protection: Graceful degradation of image quality over packet erasure channels through forward error correction" (RS-based ULP for embedded image streams; authors and venue not verified in this session) — [ResearchGate](https://www.researchgate.net/publication/3234351_Unequal_loss_protection_Graceful_degradation_of_image_quality_over_packet_erasure_channels_through_forward_error_correction)

### Inferences
- Our study can be pitched as follows: with a real RS code on JSCC bytes, the system inherits a digital cliff at the RS block level. Importance-tiered UEP plus latent zero-fill and inpainting turns the cliff into a staircase or slope, which is the JSCC analogue of classic ULP (Mohr/Riskin/Ladner-style ULP for SPIHT/JPEG2000). The classic ULP literature must be cited.

### Gaps
- I did not find a verified systematic "cliff" study specific to Gilbert-Elliott burst erasures with Deep JSCC.

## Q5 (objective c). Core Deep JSCC architectures reviewers will expect, and their channel models

### Takeaway
The expected citations are DeepJSCC (2019), DeepJSCC-f (2020), DeepJSCC-l (2021), DeepJSCC-l++ (2023), ADJSCC (our backbone), WITT (2023), SwinJSCC (2024/25), NTSCC (2022), DeepJSCC-Q (2022), VQ-DeepSC (2023) and MDJCM (2025). All were designed and evaluated over AWGN and/or fading, with none over packet erasure. Closeness: DISTANT, except ADJSCC (our backbone, cite as such) and DeepJSCC-Q (RELATED).

### Cited Findings
- **Bourtsoulatze, Kurka, Gündüz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," IEEE TCCN 2019.** CNN autoencoder over AWGN and slow Rayleigh fading; graceful degradation — [arXiv:1809.01733](https://arxiv.org/pdf/1809.01733)
- **Kurka, Gündüz, "DeepJSCC-f: Deep Joint Source-Channel Coding of Images with Feedback," IEEE JSAIT 2020 (ICASSP 2020 conference version).** Uses channel-output feedback and significantly outperforms ideal separation (BPG + capacity-achieving code) — [arXiv:1911.11174](https://arxiv.org/pdf/1911.11174); [ICASSP version](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/Kurka_ICASSP20.pdf). (Journal venue from my prior knowledge, not re-verified.)
- **Kurka, Gündüz, "Bandwidth-Agile Image Transmission with Deep JSCC" (DeepJSCC-l), IEEE TWC 2021.** Successive refinement and multiple descriptions; graceful degradation with SNR — [arXiv:2009.12480](https://arxiv.org/abs/2009.12480)
- **Bian, Shao, Gündüz, "DeepJSCC-l++: Robust and Bandwidth-Adaptive Wireless Image Transmission," IEEE GLOBECOM 2023.** Swin/ViT backbone with bandwidth ratio and SNR as side information; dynamically weighted multi-ratio loss; beats BPG + capacity-achieving code — [arXiv:2305.13161](https://arxiv.org/abs/2305.13161v2)
- **Xu, Ai, Chen, Yang, Sun, Rodrigues, "Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules" (ADJSCC), IEEE TCSVT vol. 32, no. 4, pp. 2315–2328 (early access 2021, issue 2022).** SNR-adaptive channel-wise attention over AWGN/fading. This is our backbone. The citation was verified via SwinJSCC's reference list, which also lists ADJSCC at 14.66M parameters — [SwinJSCC arXiv](https://arxiv.org/pdf/2308.09361). (The ADJSCC paper page itself was not fetched. The commonly cited arXiv id 2012.00533 is from my prior knowledge, not re-verified.)
- **Yang et al., "WITT: A Wireless Image Transmission Transformer for Semantic Communications," IEEE ICASSP 2023.** Swin Transformer backbone with a spatial modulation module for channel adaptation — [arXiv:2211.00937](https://arxiv.org/pdf/2211.00937)
- **Yang, Wang, Dai, et al., "SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding," IEEE TCCN vol. 11, no. 1, pp. 90–104, 2025 (arXiv 2023).** Channel ModNet and Rate ModNet; better than or comparable to BPG + 5G LDPC — [arXiv:2308.09361](https://arxiv.org/pdf/2308.09361)
- **Dai, Wang, Tan, Si, Qin, Niu, Zhang, "Nonlinear Transform Source-Channel Coding for Semantic Communications" (NTSCC), IEEE JSAC 40(8):2300–2316, 2022, DOI 10.1109/JSAC.2022.3180802.** Learned analysis transform plus an entropy-model prior that drives variable-length deep JSCC; beats analog deep JSCC and separation — [arXiv:2112.10961](https://arxiv.org/pdf/2112.10961). (Search confirmed "Dai and six co-authors". The middle-author list is from prior knowledge.)
- DeepJSCC-Q, VQ-DeepSC and MDJCM: see Q2.

### Inferences
- Reviewers may ask why ADJSCC rather than SwinJSCC. Yang et al. (2511.16983) suggest Transformer backbones are intrinsically more packet-loss-robust, so this choice affects the conclusions, and a SwinJSCC/WITT ablation or discussion would strengthen the paper.

### Gaps
- Full author lists for WITT and SwinJSCC were not fetched. The first author (Ke Yang, BUPT) is from the search summary, and the rest is from prior knowledge.
- I did not verify any DeepSC (Xie & Qin) packet-erasure evaluation. The DeepSC text models are AWGN/fading and are distant from this work.
