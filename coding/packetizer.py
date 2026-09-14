"""
Packetization: chunks a quantized ADJSCC-Q latent (the byte-valued `idx`
tensor from models/quantizer.py's ScalarSoftToHardQuantizer.forward(), or
equivalently ADJSCCQEncoder(..., return_indices=True)) into fixed-size
"packets" -- the unit a packet-erasure channel drops or delivers whole,
per Problem_Statement.pdf's packet-erasure-channel description (Packet 1
.. Packet 8, some received, some erased).



NOT yet implemented here (explicitly out of scope for this file):
  - Reed-Solomon encoding/decoding (GF(2^8) arithmetic, parity symbols,
    erasure recovery). That needs the `reedsolo` library; this file has
    no dependency on it and no RS-specific logic.
  - The calibrated Bernoulli/Gilbert-Elliott packet-erasure channel
    simulator described in the 12-week plan's Week 7 deliverable.
    erase_packets() below is a minimal, seedable erasure applier --
    enough to test packetize/depacketize round-trips and to eventually
    replace experiments/toy_demo.py's ad hoc erasure placeholder -- not
    the calibrated channel model Week 7 asks for.
  - Any fallback-imputation strategy for symbols an erased packet takes
    with it (zero-fill / mean-fill / learned). depacketize() returns an
    explicit erased-symbol mask precisely so that decision stays visible
    and overridable rather than silently baked in -- see its docstring.
    That decision is still open per the PHASE3 report's Phase 3 section.

OPEN DESIGN QUESTION, flagged rather than silently decided:
  How a "packet" (this file's unit of erasure) relates to an RS(n,k)
  codeword (Phase 5's unit of correction) is not yet decided. Two
  extremes: (a) one packet == one full RS codeword's worth of data
  symbols, so losing a packet means losing an entire codeword's data at
  once and RS's erasure budget must absorb that many erased symbols in
  one shot; (b) one RS codeword's symbols are interleaved across many
  packets, so a single packet loss only costs one symbol per codeword,
  spread thin -- closer to how real systems survive bursty/packet losses,
  but a packet then has no meaning on its own without the interleaving
  pattern. This file implements neither: it only does fixed-size,
  non-interleaved chunking (option (a)'s naive form, with no RS framing
  at all yet). Revisit before wiring in real RS in Phase 5.

PACKET_SIZE_SYMBOLS below is likewise an arbitrary, documented placeholder
-- it happens to divide the current architecture's latent length exactly
(k_over_n=1/6, image_size=32 -> k=512 -> 2k=1024 symbols per image;
1024 / 32 = 32 packets, no padding needed today), but nothing here
requires that; packetize() pads correctly either way.
"""
from dataclasses import dataclass

import torch

PACKET_SIZE_SYMBOLS = 32


@dataclass
class PacketizedLatent:
    """Batched packet representation of one quantized latent tensor.

    packets:    (B, num_packets, packet_size) torch.uint8 -- the actual
                symbol bytes, one RS-symbol-to-be per element.
    erased:     (B, num_packets) torch.bool -- True where the whole
                packet has been erased (all-False immediately after
                packetize(); set by erase_packets()).
    n_symbols:  true per-sample symbol count *before* padding -- what
                depacketize() trims back to.
    packet_size: symbols per packet, kept alongside the data rather than
                re-derived, so depacketize() doesn't need it passed again.
    orig_shape: idx's shape before packetization (e.g. (B, 2k)) -- what
                depacketize() reshapes its output back to.
    """
    packets: torch.Tensor
    erased: torch.Tensor
    n_symbols: int
    packet_size: int
    orig_shape: torch.Size


def packetize(idx: torch.Tensor, packet_size: int = PACKET_SIZE_SYMBOLS) -> PacketizedLatent:
    """
    idx: (B, L) tensor of byte-valued symbols in [0, 256) -- the `idx`
    output of ScalarSoftToHardQuantizer.forward() / ADJSCCQEncoder's
    return_indices=True. Must be 2D: this is the shape ADJSCCQEncoder
    actually produces (a flat power-normalized vector per image), and
    packetization has no reason to special-case any other layout.

    Pads the last packet with zero-bytes if L isn't a multiple of
    packet_size; padding never leaks into depacketize()'s output because
    n_symbols records the true length and depacketize() trims to it.
    """
    assert idx.dim() == 2, f"packetize expects a (B, L) idx tensor, got shape {tuple(idx.shape)}"
    assert int(idx.min()) >= 0 and int(idx.max()) <= 255, \
        "idx values must be valid byte indices in [0, 256) -- got a value outside that range"

    B, L = idx.shape
    num_packets = -(-L // packet_size)  # ceil division
    pad_len = num_packets * packet_size - L

    idx_u8 = idx.to(torch.uint8)
    if pad_len > 0:
        pad = torch.zeros(B, pad_len, dtype=torch.uint8, device=idx.device)
        idx_u8 = torch.cat([idx_u8, pad], dim=1)

    packets = idx_u8.view(B, num_packets, packet_size)
    erased = torch.zeros(B, num_packets, dtype=torch.bool, device=idx.device)

    return PacketizedLatent(packets=packets, erased=erased, n_symbols=L,
                             packet_size=packet_size, orig_shape=idx.shape)


def erase_packets(packetized: PacketizedLatent, erasure_rate: float,
                   generator: torch.Generator = None) -> PacketizedLatent:
    """
    Marks whole packets erased independently at random, each with
    probability `erasure_rate` -- NOT a calibrated channel model (see
    module docstring), just enough to exercise the packet-erasure
    concept and test depacketize()'s masking. Erased packets' bytes are
    zeroed (not merely flagged) so nothing downstream can accidentally
    read ground-truth data for a packet that was never "received."

    Returns a new PacketizedLatent; does not modify `packetized` in
    place. Composes: calling this again on an already-erased
    PacketizedLatent only adds erasures, it never un-erases a packet.
    """
    if erasure_rate <= 0.0:
        return packetized

    B, num_packets, _ = packetized.packets.shape
    draw = torch.rand(B, num_packets, generator=generator).to(packetized.packets.device)
    new_erased = packetized.erased | (draw < erasure_rate)

    new_packets = packetized.packets.clone()
    new_packets[new_erased] = 0

    return PacketizedLatent(packets=new_packets, erased=new_erased, n_symbols=packetized.n_symbols,
                             packet_size=packetized.packet_size, orig_shape=packetized.orig_shape)


def depacketize(packetized: PacketizedLatent, fill_value: int = 0):
    """
    Inverse of packetize(): flattens packets back into an idx-shaped
    tensor and trims off packetize()'s padding.

    Returns (idx_recovered, erased_symbol_mask):
      idx_recovered: torch.long, shape == orig_shape. Positions whose
        packet was erased are filled with `fill_value` (default 0) --
        an ARBITRARY placeholder, not a considered imputation strategy.
        The real fallback-imputation decision (zero-fill / mean-fill /
        learned) is still open -- see module docstring. Prefer using
        erased_symbol_mask to apply your own strategy over trusting this
        default.
      erased_symbol_mask: torch.bool, same shape as idx_recovered. True
        at every symbol position whose packet was erased -- i.e. "this
        value is not real data, decide what to do with it."
    """
    B, num_packets, packet_size = packetized.packets.shape
    flat_bytes = packetized.packets.reshape(B, num_packets * packet_size)
    flat_erased = (packetized.erased.unsqueeze(-1)
                   .expand(B, num_packets, packet_size)
                   .reshape(B, num_packets * packet_size))

    idx_recovered = flat_bytes[:, :packetized.n_symbols].to(torch.long)
    erased_symbol_mask = flat_erased[:, :packetized.n_symbols]

    if fill_value != 0:
        idx_recovered = idx_recovered.clone()
        idx_recovered[erased_symbol_mask] = fill_value

    return idx_recovered.reshape(packetized.orig_shape), erased_symbol_mask.reshape(packetized.orig_shape)
