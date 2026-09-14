"""
ADJSCC-Q: models/adjscc.py's ADJSCCEncoder/ADJSCCDecoder wrapped around
the learnable soft-to-hard scalar quantizer in models/quantizer.py.
This is the Phase 3 (DeepJSCC-Q integration) deliverable.

Built on top of ADJSCC directly, not the plain DeepJSCC baseline,
because the final RS-tiering pipeline (Phase 6) needs ADJSCC's
per-channel importance signal (af5_bottleneck) regardless, and doing
the quantization integration against the baseline first would mean
repeating this same wiring a second time later for no benefit.

Kept as a separate file rather than editing models/adjscc.py in place,
for the same reason models/adjscc.py itself gave for not editing
models/deepjscc.py in place: models/adjscc.py stays a stable,
already-tested reference to fall back to / compare against.

=====================================================================
Phase-3-scoped decisions made here (full reasoning in the Week 3
planning conversation; summarized in PROJECT_CONTEXT.md):
=====================================================================

1. CHANNEL MODE: IDENTITY.
   quantize -> dequantize -> decoder. No AWGN noise and no simulated
   erasure are injected in this phase -- ADJSCCQEncoder's output z_q is
   fed directly into the unmodified ADJSCCDecoder.

   Why: ADJSCC already has one documented channel-mismatch caveat
   (trained under AWGN, but the project's actual target is packet
   erasure). Training the quantizer under AWGN or under a simulated
   erasure right now would stack a SECOND unresolved channel-mismatch
   question on top of the first one, and the erasure option specifically
   requires deciding the still-open fallback/imputation strategy
   (zero-fill / mean-fill / learned) before it can even be implemented.
   Identity-channel training isolates pure quantization distortion as
   its own clean, reportable number, independent of both of those open
   questions.

   IMPORTANT: a checkpoint trained under identity-channel mode encodes
   quantizer centers and encoder/decoder weights that were never
   exposed to corruption, and is NOT expected to be robust if later
   evaluated under AWGN or erasure without retraining. Swapping channel
   modes after the fact is not a config change -- it requires a new
   training run. See PROJECT_CONTEXT.md Phase 3 section for the
   AWGN/erasure retraining plan once coding/ exists.

2. CODEBOOK: single shared, learnable, 256-level 1D scalar codebook.
   See models/quantizer.py's docstring for the full reasoning (256
   levels = 1 byte = 1 future GF(2^8) RS symbol per latent scalar).

3. AF-MODULE SNR CONDITIONING UNDER IDENTITY CHANNEL:
   AFModule.forward() requires an snr_db argument architecturally,
   even though no noise is actually injected in this phase. Rather than
   sampling a random SNR per batch (which would train the AF gates to
   respond to an "SNR" value that has no causal relationship to any
   real corruption -- pure training noise with no benefit), callers
   should pass one fixed nominal value for the whole identity-channel
   phase. NOMINAL_SNR_DB below defaults to 10.0 dB purely to match Week
   1's overfit-test convention, so PSNR numbers from this phase are at
   least nominally comparable to earlier logged runs; it does NOT
   represent an actual channel condition and should not be interpreted
   as one. Flag this as a Phase-3-only placeholder in the Week 4
   meeting notes once a real channel sits between quantizer and
   decoder.
"""
import torch
import torch.nn as nn

from models.adjscc import ADJSCCEncoder, ADJSCCDecoder
from models.quantizer import ScalarSoftToHardQuantizer

# See point 3 above. Experiment scripts should import and use this
# constant (or override it explicitly) rather than re-deriving their
# own "what SNR do I pass under identity channel" convention.
NOMINAL_SNR_DB_IDENTITY = 10.0


class ADJSCCQEncoder(nn.Module):
    """ADJSCCEncoder followed by ScalarSoftToHardQuantizer. Composition,
    not inheritance or reimplementation: the underlying ADJSCC conv/AF
    stack is untouched and imported directly from models/adjscc.py, so
    this file cannot silently drift from that already-tested backbone."""

    def __init__(self, k_over_n: float = 1 / 6, image_size: int = 32,
                 num_levels: int = 256, init_range: tuple = (-3.0, 3.0),
                 init_sigma: float = 1.0):
        super().__init__()
        self.encoder = ADJSCCEncoder(k_over_n=k_over_n, image_size=image_size)
        self.k = self.encoder.k
        self.quantizer = ScalarSoftToHardQuantizer(
            num_levels=num_levels, init_range=init_range, init_sigma=init_sigma
        )

    @torch.no_grad()
    def calibrate_quantizer(self, x: torch.Tensor, snr_db=NOMINAL_SNR_DB_IDENTITY,
                             low_pct: float = 1.0, high_pct: float = 99.0):
        """Run one calibration batch of real images through the
        (unquantized) ADJSCC encoder and re-center the codebook on the
        empirical range of z. Call this once, right after construction
        and before training starts -- see models/quantizer.py:calibrate
        for why this matters more here than a generic fixed-range guess
        would."""
        z = self.encoder(x, snr_db, return_attn=False)
        self.quantizer.calibrate(z, low_pct=low_pct, high_pct=high_pct)

    def forward(self, x: torch.Tensor, snr_db, return_attn: bool = False,
                return_indices: bool = False):
        if return_attn:
            z, attn = self.encoder(x, snr_db, return_attn=True)
        else:
            z = self.encoder(x, snr_db, return_attn=False)
            attn = None

        z_q, idx = self.quantizer(z)

        if return_attn and return_indices:
            return z_q, attn, idx
        if return_attn:
            return z_q, attn
        if return_indices:
            return z_q, idx
        return z_q


# No separate decoder class. ADJSCCDecoder is reused completely
# unchanged: under identity-channel mode, z_q has exactly the same
# shape and role as the z that ADJSCCDecoder already consumes (a
# (B, 2k) real tensor), so there is nothing for a wrapper decoder class
# to do. Re-evaluate this alias once Phase 5 puts a real channel
# between the quantizer and the decoder -- at that point the decoder
# may need to consume RS-decoded/imputed symbols rather than z_q
# directly, and this alias should become a real class again.
ADJSCCQDecoder = ADJSCCDecoder
