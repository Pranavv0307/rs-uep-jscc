# Speech Semantic Communication, Deep JSCC for Speech, and Packet-Loss-Robust Neural Speech Codecs

Context being judged: a 1-D conv Deep JSCC on 16 kHz LibriSpeech (1.024 s -> 16 x 64 latent = 1024 bytes, 8 kbit/s), with and without squeeze-and-excitation (SE) gates; byte-quantized latent; Reed-Solomon (RS) over a packet-erasure channel; importance-ranked UEP. Metrics SNR, SI-SDR, PESQ-WB, STOI. Planned baseline: Opus + RS at the same byte budget.

Verification note: entries marked [verified] were confirmed this session from arXiv / ISCA / IEEE-indexed pages or the authors' institutional repositories. Entries marked [prior knowledge, not re-fetched] are canonical works whose existence is not in doubt, but whose link/venue was not re-opened this session. The report writer should treat them as lower-confidence on bibliographic detail.

---

## Q1. Has importance-aware UEP of a learned speech latent over erasure channels been done?

### Takeaway
I found no paper that applies importance-ranked unequal error protection (e.g., RS with different code rates by latent importance) to a learned speech latent over a packet-erasure channel. The nearest work does one of two things. It uses importance only for *rate allocation* over AWGN/fading channels (DSST). Or it targets packet loss with *equal-importance* in-band FEC plus learned concealment (Glaris, NESC-FEC, DRED, SoundSpring, FD-PLC). Glaris (BUPT, Dec 2025) is the closest threat: it uses LibriSpeech at 16 kHz with packet loss and FEC, but it uses no RS and no importance ranking. The specific combination in our work therefore looks open, but the space around it is crowded and moving fast. Most of it comes from the Dai/Niu/Zhang group at BUPT.

### Cited Findings

**(a) Speech semantic communication / Deep JSCC for speech**

- **DeepSC-S.** Z. Weng, Z. Qin, "Semantic Communication Systems for Speech Transmission," IEEE JSAC, 2021 (arXiv 2102.12605). It is cited as IEEE JSAC 2021 by both DSST and Glaris. An earlier conference-style version is Z. Weng, Z. Qin, G. Y. Li, "Semantic Communications for Speech Signals," arXiv 2012.05369. [verified] — [arXiv 2102.12605](https://arxiv.org/abs/2102.12605); [arXiv 2012.05369](https://ar5iv.labs.arxiv.org/html/2012.05369)
  - Architecture: an attention mechanism based on a **squeeze-and-excitation network** gives higher weight to essential speech information. A general model handles various channel conditions without retraining. Metrics are SDR and PESQ. — [arXiv 2102.12605](https://arxiv.org/abs/2102.12605)
  - Details from 2012.05369: Edinburgh DataShare speech dataset (>10,000 training wav files, 800 test files), 16 kHz originally, **down-sampled to 8 kHz**, frame length 128. Channels are AWGN, Rayleigh and Rician. Metrics are SDR and PESQ (ITU-T P.862). The baseline is G.711 8-bit A-law PCM + turbo code (rate 1/3) + 64-QAM. Encoder/decoder use 4 SE-ResNet modules; the channel encoder/decoder are 2-D CNNs. — [ar5iv 2012.05369](https://ar5iv.labs.arxiv.org/html/2012.05369)
  - Training over Rician fading gave the most robust model across conditions. — [search summary of 2012.05369](https://arxiv.org/abs/2012.05369v2)
  - **Closeness: close (architecturally).** DeepSC-S already uses SE attention in a speech JSCC. Our "with/without SE gates" ablation therefore has direct precedent and is not novel by itself. DeepSC-S uses analog AWGN/fading channels, with no erasures, no quantization, no RS and no UEP. Novelty threat to the UEP claim: low. Threat to any "SE attention for speech JSCC" claim: high.
- **DSST.** Z. Xiao, S. Yao, J. Dai, S. Wang, K. Niu, P. Zhang, "Wireless Deep Speech Semantic Transmission," ICASSP 2023 (arXiv 2211.02283), BUPT. [verified] — [arXiv 2211.02283](https://arxiv.org/abs/2211.02283v1); [ar5iv](https://ar5iv.labs.arxiv.org/html/2211.02283)
  - A nonlinear transform maps speech into a latent space. A hyperprior entropy model estimates the importance of each feature, and features get **different coding rates**: per-frame entropy −log p(y_i|z) sets the per-frame bandwidth, and silent frames get fewer symbols. An SNR token adapts the model to channel state. — [ar5iv 2211.02283](https://ar5iv.labs.arxiv.org/html/2211.02283)
  - Dataset: TIMIT at 16 kHz, frames of 512 samples. Channels: AWGN and COST2100 fading (indoor, 5.3 GHz). Baselines: AMR-WB + 5G LDPC, Opus + 5G LDPC, and DeepSC-S. Metrics: PESQ and MUSHRA; STOI is not used. Bandwidth is reported in kHz of channel bandwidth, not bit/s. Claimed savings: about 75% bandwidth vs DeepSC-S, and more than 45% vs Opus/AMR-WB + LDPC. — [ar5iv 2211.02283](https://ar5iv.labs.arxiv.org/html/2211.02283)
  - **Closeness: close.** This is the clearest precedent for "importance-aware" allocation over a learned speech latent. It is *rate/bandwidth* allocation on an analog channel, not UEP over erasures. Reviewers will likely ask how our importance ranking relates to DSST's entropy-based importance. Its "Opus + 5G LDPC" and "AMR-WB + LDPC" baselines are the direct analogue of our "Opus + RS".
- **DeepSC-ST.** "Deep Learning Enabled Semantic Communications with Speech Recognition and Synthesis" (Weng, Qin, Tao, Pan, Liu, et al.; arXiv 2205.04603). Per the search result, the author list includes Geoffrey Ye Li, not Hanzo. I did not verify the final venue; it is commonly cited as IEEE TWC 2023, which is unconfirmed here. — [arXiv 2205.04603](https://arxiv.org/pdf/2205.04603); [ar5iv](https://ar5iv.arxiv.org/html/2205.04603)
  - A CNN + BRNN joint semantic-channel encoder extracts text-related features. The receiver recovers the text (ASR), then synthesizes speech (TTS) using the user ID. Gains are largest at low SNR. — [ar5iv 2205.04603](https://ar5iv.arxiv.org/html/2205.04603)
  - **Closeness: distant.** It is a speech-to-text/TTS task, not waveform fidelity.
- **Semantic-preserved speech transmission.** T. Han, Q. Yang, Z. Shi, S. He, Z. Zhang, "Semantic-preserved Communication System for Highly Efficient Speech Transmission," arXiv 2205.12727. A catalog record says IEEE JSAC 2023. The system sends only ASR-relevant semantics, plus a small amount of semantic-irrelevant information for speech-to-speech. It adds language-model correction, and the authors claim 16% of the symbols with about 10% lower WER. — [arXiv 2205.12727](https://arxiv.org/pdf/2205.12727); [DeepAI](https://deepai.org/publication/semantic-preserved-communication-system-for-highly-efficient-speech-transmission)
  - **Closeness: distant.** It is task/ASR-oriented and does not use an erasure channel.
- **Aalborg/Oldenburg low-latency speech JSCC.** M. Bokaei, J. Jensen, S. Doclo, J. Østergaard:
  - "Channel-Configurable Deep Wireless Speech Transmission," IEEE WCNC 2024, DOI 10.1109/WCNC57260.2024.10570880. A single network with CSI-aware JSCC encoder/decoder and about 2 ms latency. — [AAU VBN](https://vbn.aau.dk/da/publications/channel-configurable-deep-wireless-speech-transmission/)
  - "Deep Digital Joint Source-Channel Based Wireless Speech Transmission," EUSIPCO 2024, pp. 341–345, DOI 10.23919/eusipco63174.2024.10715034. Constellation-constrained (digital) deep JSCC for speech; it beats separate source-channel coding in poor channels. — [AAU VBN](https://vbn.aau.dk/en/publications/deep-digital-joint-source-channel-based-wireless-speech-transmiss/)
  - **Closeness: related.** These are speech JSCC on fading channels. No packet erasures or UEP are reported on the abstract pages; the datasets and metrics were not stated there.
- **wav2vec-based audio semantic communication.** Tong et al., Frontiers in Communications and Networks, 2021 (DOI 10.3389/frcmn.2021.734402). A wav2vec-style CNN autoencoder with federated training; the authors claim about 100x lower MSE than traditional coding. — [Frontiers](https://frontiersin.org/journals/communications-and-networks/articles/10.3389/frcmn.2021.734402/text)
  - **Closeness: distant.**
- I found **no** paper that transmits HuBERT discrete units over a wireless or erasure channel (see Gaps). The closest "foundation model" speech semantic-communication work is LargeSC (below).

**(b) Speech semantic communication / JSCC under packet loss or with UEP**

- **Glaris (closest threat).** Z. Han, J. Dai, S. Yao, J. Wang, Y. Li, K. Niu, W. Xu, P. Zhang, "Error-Resilient Semantic Communication for Speech Transmission over Packet-Loss Networks," arXiv 2512.08203, Dec 2025, BUPT + GUET. No venue stated. [verified] — [arXiv abs](https://arxiv.org/abs/2512.08203); [HTML](https://arxiv.org/html/2512.08203)
  - Architecture: a VQ-VAE generative latent (Stage I at 24 kbps with RVQ), then a transform coder with a hyperprior. The **hyperprior side information z is reused as in-band FEC**: N backup copies at frame offsets K (non-consecutive, e.g., {1, 13}, for bursts). Redundancy costs 0.5·Q·N kbps. The authors assume z has a lower loss probability than the main stream. Side-information-based PLC uses learned mask tokens. — [HTML](https://arxiv.org/html/2512.08203)
  - Loss models: i.i.d. loss at 5/10/20/30%; a three-state Markov burst model (10% average); real traces from the INTERSPEECH 2022 PLC Challenge; COST2100 + Sionna 5G HARQ-derived loss traces. — [HTML](https://arxiv.org/html/2512.08203)
  - Data and metrics: LibriSpeech at 16 kHz (train-clean-100 + 360; test-clean). Metrics are PESQ, STOI, WER (wav2vec2 XLSR ASR), DNSMOS, NISQA, PLCMOS, PESQ-Q10, BD-rate, RTF and MUSHRA. Baselines are Opus 1.5 (FARGAN PLC, DRED), SoundStream (zero-fill and with entropy coding), FD-PLC, SoundSpring and DeepSC-S. **No Reed-Solomon** (it appears only in related work), and no importance-ranked UEP beyond protecting z. — [HTML](https://arxiv.org/html/2512.08203)
  - Example result: at 18 kbps with 30% loss, PESQ is 2.63 (Glaris + 3 kbps FEC) vs 2.14 (Opus DRED) vs 2.06 (SoundSpring). FEC overheads at 12 kbps were 4.2–25%. — [HTML](https://arxiv.org/html/2512.08203)
  - **Closeness: close / partial overlap.** It uses the same dataset, sampling rate and channel abstraction (packet erasures) and the same idea of protecting part of the latent more strongly (z gets backup copies). It differs in having no RS, no importance ranking within the main latent, no Deep JSCC (it is a separated digital codec + FEC), and a higher bitrate (12–18 kbps vs our 8 kbps). **Novelty threat: moderate.** We must cite it and position against it. Its baseline set (Opus 1.5 with DRED, SoundStream, DeepSC-S) is what a reviewer from this community will expect.
- **SoundSpring.** S. Yao et al., "SoundSpring: Loss-Resilient Audio Transceiver with Dual-Functional Masked Language Modeling," arXiv 2501.12696. A library catalog record and Glaris's reference list give IEEE JSAC, vol. 43, no. 4, 2025. RVQ tokens are modeled by a masked LM that serves both entropy coding (transmitter) and lost-token prediction (receiver). It explicitly separates compression from digital transmission instead of using deep JSCC. — [arXiv 2501.12696](https://arxiv.org/pdf/2501.12696); [catalog record](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c318t-e2b5c7daa897c5d51025d6f4f2fdedc1a5e9ab1519d9a8512d5828b39586da5d3?lng=en)
  - A search summary suggested that SoundSpring gives coarse RVQ layers more FEC protection. **I could not confirm this from the paper text**; treat it as unverified. — [search lead](https://arxiv.org/pdf/2501.12696)
  - **Closeness: close (if the layer-wise protection claim holds), otherwise related.**
- **LargeSC.** "Large Speech Model Enabled Semantic Communication," arXiv 2512.04711. Moshi fine-tuned with LoRA regenerates lost speech tokens. The authors report 550 bps–2.06 kbps, better quality at high packet loss, and about 460 ms latency (from a search snippet; full text not read). — [arXiv 2512.04711](https://arxiv.org/abs/2512.04711)
  - **Closeness: related.**
- **Ross-S2T.** "Robust Semantic Communications for Speech Transmission" (speech-to-text translation), arXiv 2403.05187. A GAN-based semantic compensator estimates lost semantics. — [arXiv 2403.05187](https://arxiv.org/abs/2403.05187)
  - **Closeness: distant** (S2T task).
- **Feature partitioning (not speech).** "Feature Partitioning and Semantic Equalization for Intrinsic Robustness in Semantic Communication under Packet Loss," arXiv 2511.16983. It reports that a CNN with semantic equalization keeps about 85% of lossless PSNR at 40% packet loss (image-oriented). — [arXiv 2511.16983](https://arxiv.org/abs/2511.16983)
- **Generic semantic UEP papers (not speech).** These belong to the image/text UEP literature; I did not verify their modalities. Each argues that semantic features have non-uniform importance and so deserve UEP. — [UEP for Digital Semantic Communication with Channel Coding, arXiv 2508.03381](https://arxiv.org/html/2508.03381); [Reinforcement-Learned UEP for Quantized Semantic Embeddings, arXiv 2601.00186](https://arxiv.org/html/2601.00186); [Proactive Importance-Ordered Restructuring for Enhanced UEP, arXiv 2604.00595](https://arxiv.org/html/2604.00595)
- **ResiGLC (visual, not speech).** S. Yao, J. Dai, S. Wang, G. Lu, K. Niu, W. Xu, W. Zhang, P. Zhang, "Loss-Resilient Semantic Communication over Packet-Loss Networks at Extreme-Low Bandwidth," arXiv 2608.19590 (Aug 2026), with a comment saying it was accepted to IEEE TMC. A masked in-context latent predictor gives packet-loss resilience; no FEC/UEP is mentioned. It is listed under eess.IV and targets visual communication. — [arXiv 2608.19590](https://arxiv.org/abs/2608.19590)

### Inferences
- The specific claim "importance-ranked RS-UEP over a byte-quantized Deep JSCC speech latent on a packet-erasure channel" was not found in any source examined. Precedents exist for each component: importance-aware rate (DSST), SE attention (DeepSC-S), latent in-band FEC (Glaris, NESC, DRED), and packet-loss JSCC for speech (Glaris). The novelty argument should be framed as the *combination plus a systematic UEP study*, not as a first in any one component.
- The BUPT group (Dai, Niu, Zhang, Yao) has three closely related papers (DSST, SoundSpring, Glaris) and is the most likely reviewer pool.

### Gaps
- I could not confirm whether SoundSpring applies layer-wise unequal FEC to RVQ levels; the full text was not read.
- I did not read the full Bokaei et al. papers (datasets, metrics, any erasure experiments).
- I did not do an exhaustive search of IEEE Xplore for 2026 conference papers (ICC/GLOBECOM 2026) on speech + UEP. A narrow 2026 paper could exist that I did not find.

---

## Q2. What datasets, bitrates and metrics are standard in speech semantic communication papers?

### Takeaway
The datasets in use are LibriSpeech at 16 kHz (Glaris, the most comparable packet-loss work), TIMIT at 16 kHz (DSST) and Edinburgh DataShare at 8 kHz (DeepSC-S). PESQ is universal. STOI and WER are common, MUSHRA is used for subjective tests, and the packet-loss/codec community adds DNSMOS, NISQA and PLCMOS. SDR is used by DeepSC-S. ViSQOL and segmental SNR are rare in this sub-community. Bitrates for packet-loss speech work cluster at 6–24 kbps (Glaris 12–18 kbps), so our 8 kbit/s is in range. JSCC-on-AWGN papers report channel bandwidth (kHz or symbols) rather than bit/s.

### Cited Findings
- DeepSC-S uses Edinburgh DataShare (16 kHz down-sampled to 8 kHz), with SDR and PESQ as metrics. — [ar5iv 2012.05369](https://ar5iv.labs.arxiv.org/html/2012.05369)
- DSST uses TIMIT at 16 kHz with 512-sample frames. It reports PESQ and MUSHRA, and measures bandwidth in kHz. — [ar5iv 2211.02283](https://ar5iv.labs.arxiv.org/html/2211.02283)
- Glaris uses LibriSpeech at 16 kHz (train-clean-100/360, test-clean). Metrics are PESQ, STOI, WER, DNSMOS, NISQA, PLCMOS, PESQ-Q10 (10th-percentile PESQ), BD-rate, RTF and MUSHRA. Bitrates are 12–18 kbps, with packet loss at 5–30%, Markov bursts and real PLC-challenge traces. — [arXiv 2512.08203](https://arxiv.org/html/2512.08203)
- INTERSPEECH 2022 PLC Challenge (L. Diener et al., arXiv 2204.05222; ISCA Interspeech 2022): built on **real-world packet loss traces**, evaluated by crowd-sourced ITU-T P.808 CCR MOS plus a speech-recognition rate. It introduced **PLCMOS**, a neural estimator of human ratings for concealed audio. — [arXiv 2204.05222](https://arxiv.org/pdf/2204.05222); [ISCA archive](https://www.isca-archive.org/interspeech_2022/diener22_interspeech.html); [Microsoft challenge page](https://www.microsoft.com/en-us/research/academic-program/audio-deep-packet-loss-concealment-challenge-interspeech-2022/)
- Standard metric references [prior knowledge, not re-fetched]: PESQ is ITU-T P.862, with the wideband extension P.862.2 (https://www.itu.int/rec/T-REC-P.862.2). STOI is C. H. Taal, R. C. Hendriks, R. Heusdens, J. Jensen, "An Algorithm for Intelligibility Prediction of Time–Frequency Weighted Noisy Speech," IEEE TASLP 19(7), 2011. SI-SDR is J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, "SDR – Half-baked or Well Done?," ICASSP 2019 (https://arxiv.org/abs/1811.02508). ViSQOL is A. Hines et al., EURASIP JASMP 2015, and ViSQOL v3 is M. Chinen et al., QoMEX 2020 (https://arxiv.org/abs/2004.09584). Crowdsourced MOS is ITU-T P.808.

### Inferences
- Our LibriSpeech 16 kHz setup with PESQ-WB and STOI matches Glaris, the closest packet-loss work. Adding **WER** (an ASR on the reconstruction) and a **non-intrusive MOS** (DNSMOS or PLCMOS) would align us with Glaris and the PLC-challenge community. Reporting a **low-percentile PESQ** (e.g., Q10) suits erasure studies, where tail behavior matters.
- Per-segment SNR and SI-SDR are fine as fidelity metrics, but reviewers in this area weight PESQ/STOI/WER/MOS more. SNR is weak for perceptual claims.
- A realistic loss model beyond i.i.d. is expected: Gilbert–Elliott/Markov bursts or the public INTERSPEECH 2022 PLC traces. Glaris uses all three kinds.
- Reviewers may question 1.024 s segments (a 1 s algorithmic delay) for conversational speech. Glaris uses 20 ms frames with a causal streaming design. We should state the latency explicitly and frame the work as streaming/broadcast rather than conversational, or discuss shorter frames.

### Gaps
- I did not verify the exact dataset and metrics used by DeepSC-ST, Han et al., or Bokaei et al.
- I did not verify the ICASSP 2024 PLC Grand Challenge paper (I believe a second edition exists; not confirmed this session).

---

## Q3. What is the closest neural-codec + FEC/PLC work, and would reviewers expect Opus+FEC or an RVQ neural-codec comparison?

### Takeaway
The closest codec-side works are Glaris (latent in-band FEC + PLC, LibriSpeech), NESC with in-band FEC + latent index prediction (Fraunhofer, Interspeech 2024), Opus DRED (RDO-VAE redundancy, shipped in Opus 1.5), SoundSpring (RVQ + masked-LM concealment), and FD-PLC (feature-domain PLC for a neural codec). Every recent packet-loss speech paper compares against **Opus 1.5 with its loss tools (LBRR in-band FEC, DRED, neural PLC)** and at least one **RVQ neural codec (SoundStream or EnCodec/DAC) with PLC**. Reviewers will very likely expect both. A plain "Opus + RS" baseline that switches off Opus's own FEC/DRED/PLC would look like a straw man.

### Cited Findings
- **DRED.** J.-M. Valin, J. Büthe, A. Mustafa, M. Klingbeil, "DRED: Deep REDundancy Coding of Speech Using a Rate-Distortion-Optimized Variational Autoencoder," IEEE JSTSP, 2024 (arXiv 2212.04453). An RDO-VAE with a Laplace-distributed quantized latent sends low-bitrate redundancy. It covers burst losses up to 1 s (50x redundancy) for under 32 kb/s, a significant improvement over Opus LBRR, and ships in Opus 1.5. [verified] — [arXiv 2212.04453](https://arxiv.org/html/2212.04453v3); [author PDF](https://jmvalin.ca/papers/valin_dred_journal.pdf)
  - **Closeness: close (codec side).** It is a learned latent redundancy for packet loss, and it is the strongest "Opus + FEC" baseline.
- **NESC error resilience.** K. Gupta, N. Pia, S. Korse, A. Brendel, G. Fuchs, M. Multrus, "On Improving Error Resilience of Neural End-to-End Speech Coders," Interspeech 2024 (arXiv 2406.08900). A low-complexity network predicts lost codebook indices in latent space (PLC), and an **in-band FEC adds about 0.8 kbps**. [verified] — [arXiv 2406.08900](https://arxiv.org/abs/2406.08900); [ISCA PDF](https://www.isca-archive.org/interspeech_2024/gupta24c_interspeech.pdf)
  - **Closeness: close.**
- **FD-PLC.** H. Xue, X. Peng, X. Jiang, Y. Lu, "Towards Error-Resilient Neural Speech Coding," Interspeech 2022, pp. 4217–4221, DOI 10.21437/Interspeech.2022-779 (arXiv 2207.00993). Self-attention feature-domain concealment of lost frames before decoding. The authors note that neural codecs trained for source coding only are extremely sensitive to channel errors. [verified] — [ISCA](https://www.isca-archive.org/interspeech_2022/xue22b_interspeech.html)
  - **Closeness: related.**
- **SoundSpring** (RVQ + masked LM for coding and concealment; IEEE JSAC 2025). See Q1. — [arXiv 2501.12696](https://arxiv.org/pdf/2501.12696)
- **Glaris** baselines: Opus 1.5 (FARGAN PLC, DRED), SoundStream, FD-PLC, SoundSpring, DeepSC-S. — [arXiv 2512.08203](https://arxiv.org/html/2512.08203)
- **SoundStream.** N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, M. Tagliasacchi, "SoundStream: An End-to-End Neural Audio Codec," IEEE/ACM TASLP 2021/2022 (arXiv 2107.03312). The RVQ codec with quantizer dropout for scalable bitrate. [verified existence via search/Glaris refs] — [arXiv 2107.03312](https://arxiv.org/pdf/2107.03312)
- **EnCodec.** A. Défossez, J. Copet, G. Synnaeve, Y. Adi, "High Fidelity Neural Audio Compression," TMLR 2023 (arXiv 2210.13438). Cited in Glaris's references. — [arXiv 2210.13438](https://arxiv.org/abs/2210.13438)
- **DAC** [prior knowledge, not re-fetched]: R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, K. Kumar, "High-Fidelity Audio Compression with Improved RVQGAN," NeurIPS 2023 (https://arxiv.org/abs/2306.06546). **Lyra** [prior knowledge, not re-fetched]: W. B. Kleijn et al., "Generative Speech Coding with Predictive Variance Regularization," ICASSP 2021 (https://arxiv.org/abs/2102.09660); Lyra v2 is SoundStream-based.
- **Opus neural PLC** [prior knowledge, not re-fetched]: J.-M. Valin et al., "Real-Time Packet Loss Concealment With Mixed Generative and Predictive Model," Interspeech 2022 (https://arxiv.org/abs/2205.05785). Glaris cites the FARGAN vocoder (Valin et al., IEEE SPL 2024) as the Opus PLC baseline. — [arXiv 2512.08203](https://arxiv.org/html/2512.08203)
- **Neural PLC lineage** (cited by Glaris): Shi et al. APSIPA 2019 (GAN loss compensation); Pascual et al. WASPAA 2021 (adversarial auto-encoding PLC); Wang et al. JASA 2021 (temporal-spectral GAN PLC); Li et al. Interspeech 2022 (end-to-end multi-loss PLC). — [arXiv 2512.08203](https://arxiv.org/html/2512.08203)
- Minje Kim's IEEE SPM 2024 tutorial ("Neural Speech and Audio Coding") notes that the very low bitrates of neural codecs leave room for redundant packets. — [PDF](https://minjekim.com/wp-content/uploads/spm2024_mkim.pdf)
- **Robust Residual FSQ** (arXiv 2508.15860) concerns quantizer robustness, not loss robustness. — [arXiv 2508.15860](https://arxiv.org/pdf/2508.15860)

### Inferences
- Minimum baseline set reviewers in this space are likely to expect, at matched bitrate (8 kbit/s) and matched total byte budget:
  1. **Opus with its native loss tools**: LBRR in-band FEC and/or DRED (Opus 1.5) plus the neural PLC.
  2. **Opus + RS**, our current plan. Keep it, but as one baseline among several.
  3. A **neural RVQ codec with packet-level RS or simple PLC**: EnCodec at 6 kbps (or DAC/SoundStream) + RS with equal and unequal protection of RVQ layers. RVQ layers are naturally importance-ordered, so this is the *obvious* UEP competitor to our importance ranking. A reviewer is likely to ask "why not just UEP over RVQ levels?"
  4. A **JSCC baseline** such as DeepSC-S or DSST. These are designed for AWGN, so this mainly serves positioning.
- At 8 kbit/s, Opus is in its SILK/hybrid speech range, so it is a strong competitor. A learned JSCC at 8 kbit/s may beat it on PESQ only at moderate to high loss. The claim should be framed around graceful degradation and the cliff effect.

### Gaps
- I did not verify Opus LBRR internals from RFC 6716 this session (see Q4). Exact LBRR overhead and its interaction with packet size are unconfirmed here.
- I did not confirm whether EnCodec/DAC packet-loss + RS evaluations exist in the literature as a standard benchmark. I found none.

---

## Q4. Does classic AMR/EVS bit-class UEP provide a direct precedent for importance-ranked protection in speech?

### Takeaway
Yes, and it is a strong and citable precedent. 3GPP AMR (TS 26.101) reorders each frame's speech bits by **subjective importance** into Class A, B and C. Class A gets an 8-bit CRC and the strongest protection (EP1); B and C get progressively weaker schemes. That is importance-ranked UEP of a speech codec's bitstream, standardized around 1999. EVS channel-aware mode and Opus LBRR are the packet-loss counterparts: in-band partial redundancy at an offset. Our contribution is to *learn* the importance ranking of a JSCC latent and apply RS-based UEP over erasures. This is a learned analogue of AMR bit classes, and it should be framed as such rather than as a new idea.

### Cited Findings
- **AMR frame structure.** In 3GPP TS 26.101, speech bits are rearranged and divided into three classes by subjective importance. Class A is protected by the internal CRC and Error Protection scheme 1 (EP1), Class B by EP2, and Class C by EP3. — [3GPP SP-99253 (TS 26.101 draft)](https://3gpp.org/FTP/tsg_sa/TSG_SA/TSGS_04/Docs/PDF/SP-99253.pdf); [TS 26.101 Rel-19 text](https://whatthespec.net/friendlyspec/spec/26.101/19.0.0)
- The 8-bit CRC uses G(x) = D^8 + D^6 + D^5 + D^4 + 1 over the Class A bits. Release 19 keeps the Codec CRC over Class A bits. — [SP-99253](https://3gpp.org/FTP/tsg_sa/TSG_SA/TSGS_04/Docs/PDF/SP-99253.pdf); [TS 26.101 Rel-19](https://whatthespec.net/friendlyspec/spec/26.101/19.0.0)
- Example class sizes from the 1999 v1.0.0 draft. At 12.2 kbit/s: 244 bits = 81 A + 103 B + 60 C. At 4.75 kbit/s: 95 bits = 39 A + 56 B, with no C. These are from a draft; check the current release before quoting. — [SP-99253](https://3gpp.org/FTP/tsg_sa/TSG_SA/TSGS_04/Docs/PDF/SP-99253.pdf)
- RAN1 liaison (1999) proposed moving 3 bits from B to A for AMR 6.7 (58 A / 76 B). This shows that class boundaries were tuned jointly with the channel coding. — [R1-99e42](https://3gpp.org/ftp/tsg_ran/wg1_rl1/TSGR1_07/Docs/Pdfs/R1-99e42.pdf)
- RTP payload drafts for AMR propagate class-based UEP and FEC to IP transport. — [draft-ietf-avt-rtp-amr-03](https://www.potaroo.net/ietf/all-ids/draft-ietf-avt-rtp-amr-03.txt)
- **EVS channel-aware mode** (3GPP TS 26.445, §5.8.1–5.8.2) sends a partial redundant copy of a frame in-band at a 2/3/5/7-frame offset (40–140 ms) at a constant 13.2 kbps (WB/SWB). It is receiver-driven via CMR. Short offsets fail in long bursts, and long offsets add delay. — [EUSIPCO 2018, "Application-Layer Redundancy for the EVS Codec"](https://new.eurasip.org/Proceedings/Eusipco/Eusipco2018/papers/1570437225.pdf); [HAL record](https://hal-imt-atlantique.archives-ouvertes.fr/hal-01971759v1)
- **Opus LBRR**: partial redundancy coding is used in Opus's LBRR (RFC 6716 §2.1.7), per the EUSIPCO 2018 paper. — [EUSIPCO 2018](https://new.eurasip.org/Proceedings/Eusipco/Eusipco2018/papers/1570437225.pdf). RFC 6716 itself [prior knowledge, not re-fetched]: J.-M. Valin, K. Vos, T. Terriberry, "Definition of the Opus Audio Codec," IETF RFC 6716, 2012, https://datatracker.ietf.org/doc/html/rfc6716.
- Xue et al. (FD-PLC) explicitly name UEP as a known tool from traditional speech coding. — [search lead; ISCA paper](https://www.isca-archive.org/interspeech_2022/xue22b_interspeech.pdf)

### Inferences
- AMR Class A/B/C is the most direct precedent and should be cited in the introduction and related work. It establishes that importance-ranked protection of speech parameters is standard practice. Our novelty is (i) a learned latent instead of hand-designed codec parameters, (ii) data-driven importance (e.g., gradient or ablation-based ranking) instead of subjective listening tests, and (iii) packet-erasure RS at the application layer instead of convolutional coding at the radio layer.
- AMR's classes protect against *bit errors* on a circuit-switched radio channel. EVS-CA, Opus LBRR, DRED and Glaris protect against *packet erasures*. We should state which lineage our erasure model follows (packet erasures), and note that AMR UEP in its original form addresses bit errors.
- A natural extra experiment, and a likely reviewer request, is a baseline that applies AMR-style class UEP to Opus or an RVQ codec's bitstream. For example, protect the first RVQ level or Opus's SILK core more strongly.

### Gaps
- The current-release AMR-WB (TS 26.201) class tables were not retrieved; only AMR-NB draft tables were seen.
- I did not fetch the RFC 6716 LBRR text or the TS 26.445 EVS spec directly; the EVS-CA details come from the EUSIPCO 2018 paper.
