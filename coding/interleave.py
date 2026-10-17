"""Transmission order of a scheme's packets (for burst-loss studies).

``encode_blocks`` lays packets out tier after tier (``sequential``): all packets
of the most important tier, then the next tier, and so on. Under independent
losses the order is irrelevant. Under a burst channel it is not: a burst that
lands on the first tier destroys the most important block, while the same burst
spread over the whole timeline costs every tier only its proportional share.

``round_robin`` therefore spreads the packets of every coding block evenly over
the transmission timeline. The loss process is always sampled on the *timeline*
(one master mask shared by all methods), and each method reads the slots its
own packets occupy, so all methods face the same physical losses.
"""
from functools import lru_cache

import torch

from coding.rs_pipeline import CodingScheme

MODES = ("sequential", "round_robin")


@lru_cache(maxsize=None)
def transmission_slots(scheme: CodingScheme, mode: str = "sequential") -> torch.Tensor:
    """``slots[p]`` = timeline slot at which packet ``p`` of ``scheme`` is sent."""
    total = scheme.total_packets
    if mode == "sequential":
        return torch.arange(total)
    if mode != "round_robin":
        raise ValueError(f"unknown interleave mode {mode!r}; expected one of {MODES}")

    items, start = [], 0
    for block_index, (n, _) in enumerate(scheme.tiers):
        items.extend(((j + 0.5) / n, block_index, start + j) for j in range(n))
        start += n
    items.sort()
    time_order = torch.tensor([packet for _, _, packet in items], dtype=torch.long)
    slots = torch.empty(total, dtype=torch.long)
    slots[time_order] = torch.arange(total)
    return slots


def packet_mask_from_timeline(timeline_mask: torch.Tensor, scheme: CodingScheme,
                              mode: str = "sequential") -> torch.Tensor:
    """Map a ``(B, >=N)`` timeline loss mask onto the packets of ``scheme``."""
    slots = transmission_slots(scheme, mode)
    if timeline_mask.shape[1] < scheme.total_packets:
        raise ValueError(
            f"timeline has {timeline_mask.shape[1]} slots but the scheme sends {scheme.total_packets} packets"
        )
    return timeline_mask[:, : scheme.total_packets][:, slots]
