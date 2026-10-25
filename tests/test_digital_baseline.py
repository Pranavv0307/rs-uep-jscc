"""Digital-baseline plumbing: packing, bit budget and graceful failure."""
import numpy as np
import pytest

from experiments.eval_digital_baseline import decode_stream, fit_to_budget, pack, unpack


def smooth_image(size=32):
    ramp = np.linspace(0, 1, size, dtype=np.float32)
    return np.stack([np.outer(ramp, ramp), np.outer(ramp, 1 - ramp), np.outer(1 - ramp, ramp)])


def test_pack_unpack_roundtrip_and_padding():
    packed = pack(b"hello", 32)
    assert len(packed) == 32
    assert unpack(packed, np.zeros(32, dtype=bool)) == b"hello"


def test_failed_symbols_are_zeroed_before_decoding():
    packed = pack(b"abcdef", 32)
    failed = np.zeros(32, dtype=bool)
    failed[4] = True  # bytes 0-1 are the length header, so this is the third payload byte
    assert unpack(packed, failed) == b"ab\x00def"


def test_header_cannot_claim_more_than_the_buffer():
    broken = np.frombuffer((60000).to_bytes(2, "big") + b"x" * 30, dtype=np.uint8)
    assert len(unpack(broken, np.zeros(32, dtype=bool))) <= 30


@pytest.mark.parametrize("codec", ["jpeg", "webp", "jpeg2000"])
def test_stream_fits_budget_and_decodes(codec):
    image = smooth_image()
    stream = fit_to_budget(image, codec, 1022)
    assert stream is not None and len(stream) <= 1022
    decoded = decode_stream(stream, codec, image.shape)
    assert decoded is not None and decoded.shape == image.shape
    assert float(np.abs(decoded - image).mean()) < 0.2


def test_garbage_stream_returns_none_not_an_exception():
    assert decode_stream(b"\x00" * 64, "jpeg", (3, 32, 32)) is None
