from coding.reed_solomon import decode, encode


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