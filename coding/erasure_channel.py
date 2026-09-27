"""Seeded packet-erasure channel for packetized RS codewords.

Packet storage stays in ``coding/packetizer.py``; this module only decides
which packets are delivered. A whole packet is either received intact or
erased, and the receiver knows which packets were erased.

The first model is independent (Bernoulli) packet loss. Burst models such as
Gilbert-Elliott are a later robustness experiment and are not mixed in here.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from coding.packetizer import PacketizedLatent


@dataclass
class ChannelOutput:
    """Received packets plus the erasure realization that produced them."""

    received: PacketizedLatent
    erasure_mask: torch.Tensor
    realized_rate: float


def _validate_rate(erasure_rate: float) -> None:
    if not 0.0 <= erasure_rate <= 1.0:
        raise ValueError("erasure_rate must satisfy 0 <= erasure_rate <= 1")


def sample_erasure_mask(
    batch_size: int,
    num_packets: int,
    erasure_rate: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw one independent loss decision per packet; ``True`` means erased.

    Sampling is done on CPU so the same generator seed yields the same mask
    regardless of the device the packets live on.
    """
    _validate_rate(erasure_rate)
    draw = torch.rand(batch_size, num_packets, generator=generator)
    return draw < erasure_rate


def apply_erasure_mask(packetized: PacketizedLatent, erasure_mask: torch.Tensor) -> ChannelOutput:
    """Erase the packets selected by ``erasure_mask`` without mutating the input.

    Erased packet bytes are zeroed so no downstream code can read transmitted
    data that was never received. Existing erasures are preserved.
    """
    expected = tuple(packetized.erased.shape)
    if tuple(erasure_mask.shape) != expected:
        raise ValueError(f"erasure_mask must have shape {expected}, got {tuple(erasure_mask.shape)}")
    mask = erasure_mask.to(device=packetized.erased.device, dtype=torch.bool)
    erased = packetized.erased | mask
    packets = packetized.packets.clone()
    packets[erased] = 0
    received = PacketizedLatent(
        packets=packets,
        erased=erased,
        n_symbols=packetized.n_symbols,
        packet_size=packetized.packet_size,
        orig_shape=packetized.orig_shape,
    )
    return ChannelOutput(received=received, erasure_mask=mask, realized_rate=mask.float().mean().item())


class BernoulliPacketErasureChannel:
    """Independent packet erasures with a fixed per-packet loss probability."""

    def __init__(self, erasure_rate: float):
        _validate_rate(erasure_rate)
        self.erasure_rate = erasure_rate

    def __call__(
        self,
        packetized: PacketizedLatent,
        generator: Optional[torch.Generator] = None,
        erasure_mask: Optional[torch.Tensor] = None,
    ) -> ChannelOutput:
        """Transmit packets through the channel.

        Pass ``erasure_mask`` to replay a previously sampled realization, so
        methods with the same packet count are compared on identical losses.
        """
        if erasure_mask is None:
            batch_size, num_packets = packetized.erased.shape
            erasure_mask = sample_erasure_mask(batch_size, num_packets, self.erasure_rate, generator)
        return apply_erasure_mask(packetized, erasure_mask)
