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


# Largest packet budget of any scheme compared in one sweep (the 96-packet
# fixed-overhead schemes). The unprotected baseline sends fewer packets.
MASTER_PACKET_COUNT = 96


def sample_master_mask(
    batch_size: int,
    erasure_rate: float,
    generator: Optional[torch.Generator] = None,
    num_packets: int = MASTER_PACKET_COUNT,
) -> torch.Tensor:
    """Draw the single loss realization every compared method must face.

    Methods with fewer packets (e.g. ``no_rs``, 64 packets) use a prefix of it
    via :func:`slice_master_mask`, so packet ``i`` has the same fate in every
    method. Drawing a separate mask per packet count lets a method win by luck.
    """
    return sample_erasure_mask(batch_size, num_packets, erasure_rate, generator)


def slice_master_mask(master_mask: torch.Tensor, num_packets: int) -> torch.Tensor:
    """First ``num_packets`` columns of the master mask."""
    if num_packets > master_mask.shape[1]:
        raise ValueError(
            f"scheme needs {num_packets} packets but the master mask only has {master_mask.shape[1]}"
        )
    return master_mask[:, :num_packets]


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


class GilbertElliottChannel:
    """Two-state Markov burst-loss channel (Track D, Step D.1).

    State Good loses a packet with probability ``p_g`` (~1%), state Bad with
    ``p_b`` (~85%). ``p_gb`` / ``p_bg`` are the Good->Bad / Bad->Good transition
    probabilities, so the mean number of consecutive packets spent in Bad is
    ``1 / p_bg`` (the "burst length"). The chain starts in its stationary
    distribution, so the average loss rate is the same at every packet slot.
    """

    def __init__(self, p_gb: float, p_bg: float, p_g: float = 0.01, p_b: float = 0.85):
        for name, value in (("p_gb", p_gb), ("p_bg", p_bg)):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must satisfy 0 < {name} <= 1")
        for name, value in (("p_g", p_g), ("p_b", p_b)):
            _validate_rate(value)
        self.p_gb, self.p_bg, self.p_g, self.p_b = p_gb, p_bg, p_g, p_b

    @classmethod
    def from_average_loss(cls, average_loss: float, burst_length: float,
                          p_g: float = 0.01, p_b: float = 0.85) -> "GilbertElliottChannel":
        """Channel with the requested long-run loss rate and mean Bad-state burst length."""
        if burst_length < 1.0:
            raise ValueError("burst_length must be >= 1 packet")
        if not p_g <= average_loss < p_b:
            raise ValueError(f"average_loss must lie in [p_g, p_b) = [{p_g}, {p_b})")
        pi_bad = (average_loss - p_g) / (p_b - p_g)
        p_bg = 1.0 / burst_length
        p_gb = pi_bad * p_bg / (1.0 - pi_bad)
        if p_gb > 1.0:
            raise ValueError(
                f"average_loss={average_loss} with burst_length={burst_length} needs p_gb={p_gb:.3f} > 1"
            )
        return cls(p_gb=max(p_gb, 1e-12), p_bg=p_bg, p_g=p_g, p_b=p_b)

    @property
    def stationary_bad(self) -> float:
        return self.p_gb / (self.p_gb + self.p_bg)

    @property
    def average_loss(self) -> float:
        pi_bad = self.stationary_bad
        return (1.0 - pi_bad) * self.p_g + pi_bad * self.p_b

    @property
    def burst_length(self) -> float:
        return 1.0 / self.p_bg

    def sample_mask(self, batch_size: int, total_packets: int,
                    generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """``(B, total_packets)`` boolean timeline mask; ``True`` means erased."""
        start = torch.rand(batch_size, generator=generator)
        u_state = torch.rand(batch_size, total_packets, generator=generator)
        u_loss = torch.rand(batch_size, total_packets, generator=generator)
        bad = start < self.stationary_bad
        mask = torch.empty(batch_size, total_packets, dtype=torch.bool)
        for t in range(total_packets):
            if t > 0:
                bad = torch.where(bad, u_state[:, t] >= self.p_bg, u_state[:, t] < self.p_gb)
            loss_probability = torch.where(bad, torch.tensor(self.p_b), torch.tensor(self.p_g))
            mask[:, t] = u_loss[:, t] < loss_probability
        return mask


def mean_loss_run_length(mask: torch.Tensor) -> float:
    """Average length of runs of consecutive erased packets (0.0 if nothing was lost)."""
    runs, lost = 0, 0
    for row in mask.cpu().numpy():
        previous = False
        for value in row:
            if value:
                lost += 1
                if not previous:
                    runs += 1
            previous = bool(value)
    return lost / runs if runs else 0.0
