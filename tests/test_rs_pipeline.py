import torch

from coding.packetizer import erase_packets
from coding.rs_pipeline import decode_uniform, encode_uniform


def test_uniform_rs_pipeline_round_trip_without_erasure():
    indices = torch.arange(1024, dtype=torch.long).reshape(1, 1024) % 256
    batch = encode_uniform(indices)
    recovered, failed = decode_uniform(batch)
    assert torch.equal(recovered, indices)
    assert not failed.any()
    assert batch.packetized.packets.shape == (1, 48, 32)


def test_uniform_rs_pipeline_recovers_one_lost_packet_per_codeword():
    indices = torch.arange(1024, dtype=torch.long).reshape(1, 1024) % 256
    batch = encode_uniform(indices)
    erased = erase_packets(batch.packetized, erasure_rate=0.0)
    erased.erased[0, 1] = True
    erased.packets[0, 1] = 0
    recovered, failed = decode_uniform(batch.__class__(
        erased, batch.original_length, batch.codeword_n, batch.codeword_k, batch.packet_size
    ))
    assert torch.equal(recovered, indices)
    assert not failed.any()