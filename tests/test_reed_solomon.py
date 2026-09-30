import numpy as np
import pytest

from coding.reed_solomon import decode, decode_block, encode, encode_block


def test_rs_round_trip_without_erasures():
    data = bytes(range(64))
    codeword = encode(data, n=96, k=64)
    recovered, success = decode(codeword, [], n=96, k=64)
    assert success
    assert recovered == data


def test_rs_recovers_one_lost_32_symbol_packet():
    data = bytes((index * 17) % 256 for index in range(64))
    codeword = bytearray(encode(data, n=96, k=64))
    codeword[32:64] = b"\x00" * 32
    recovered, success = decode(codeword, list(range(32, 64)), n=96, k=64)
    assert success
    assert recovered == data


def test_rs_reports_failure_above_erasure_capacity():
    data = bytes(range(64))
    codeword = encode(data, n=96, k=64)
    recovered, success = decode(codeword, list(range(33)), n=96, k=64)
    assert not success
    assert len(recovered) == 64


def test_rs_handles_full_byte_range():
    data = bytes(range(256))[:64]
    codeword = encode(data, n=96, k=64)
    codeword = bytearray(codeword)
    erased = list(range(0, 32))
    codeword[:32] = b"\x00" * 32
    recovered, success = decode(codeword, erased, n=96, k=64)
    assert success
    assert recovered == data


def test_rs_rejects_wrong_block_parameters():
    try:
        encode(b"abc", n=3, k=3)
        assert False, "expected RS parameter validation"
    except ValueError:
        pass

def _block(k=64, columns=16, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (k, columns), dtype=np.uint8)


def test_block_columns_match_single_codeword_encoding():
    data = _block()
    block = encode_block(data, n=96)
    assert block.shape == (96, 16)
    assert np.array_equal(block[:64], data)
    for column in range(16):
        assert bytes(block[:, column]) == encode(data[:, column].tolist(), n=96, k=64)


@pytest.mark.parametrize("n,k", [(96, 64), (32, 16), (24, 16), (40, 32)])
def test_block_recovers_up_to_n_minus_k_erased_rows(n, k):
    rng = np.random.default_rng(n)
    data = _block(k=k)
    block = encode_block(data, n=n)
    for _ in range(10):
        erased = rng.choice(n, size=n - k, replace=False)
        received = block.copy()
        received[erased] = 0
        recovered, success = decode_block(received, erased, n=n, k=k)
        assert success
        assert np.array_equal(recovered, data)


def test_block_reports_failure_above_capacity_and_keeps_received_rows():
    data = _block()
    received = encode_block(data, n=96)
    erased = list(range(10, 43))  # 33 rows > 32 parity rows
    received[erased] = 0
    recovered, success = decode_block(received, erased, n=96, k=64)
    assert not success
    keep = [row for row in range(64) if row not in erased]
    assert np.array_equal(recovered[keep], data[keep])


def test_block_with_only_parity_erased_needs_no_solve():
    data = _block()
    received = encode_block(data, n=96)
    received[64:] = 0
    recovered, success = decode_block(received, list(range(64, 96)), n=96, k=64)
    assert success
    assert np.array_equal(recovered, data)
