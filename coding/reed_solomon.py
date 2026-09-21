"""Small GF(256) Reed-Solomon erasure codec used by the pipeline.

The public API works with bytes so the coding boundary is explicit: one
quantizer index is one GF(256) symbol. The decoder handles known erasures,
which is the packet-loss case in this project. Unknown symbol errors are not
silently treated as erasures.
"""

from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple


_FIELD_SIZE = 256
_PRIMITIVE_POLYNOMIAL = 0x11D


def _build_field_tables() -> Tuple[List[int], List[int]]:
    exp = [0] * 512
    log = [-1] * _FIELD_SIZE
    value = 1
    for index in range(255):
        exp[index] = value
        log[value] = index
        value <<= 1
        if value & 0x100:
            value ^= _PRIMITIVE_POLYNOMIAL
    for index in range(255, 512):
        exp[index] = exp[index - 255]
    return exp, log


_EXP, _LOG = _build_field_tables()


def gf_mul(left: int, right: int) -> int:
    """Multiply two GF(256) elements."""
    if left == 0 or right == 0:
        return 0
    return _EXP[_LOG[left] + _LOG[right]]


def gf_pow(value: int, exponent: int) -> int:
    """Raise a GF(256) element to an integer power."""
    if exponent == 0:
        return 1
    if value == 0:
        return 0
    return _EXP[(_LOG[value] * exponent) % 255]


def _validate_parameters(n: int, k: int) -> int:
    if not 0 < k < n <= 255:
        raise ValueError("RS parameters must satisfy 0 < k < n <= 255")
    return n - k


def _validate_bytes(values: Iterable[int]) -> bytes:
    try:
        result = bytes(values)
    except (TypeError, ValueError):
        raise ValueError("RS data must contain integer byte values in [0, 255]")
    return result


def _multiply_polynomials(left: Sequence[int], right: Sequence[int]) -> List[int]:
    result = [0] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] ^= gf_mul(left_value, right_value)
    return result


@lru_cache(maxsize=None)
def _generator_polynomial(nsym: int) -> Tuple[int, ...]:
    generator = [1]
    for index in range(nsym):
        generator = _multiply_polynomials(generator, [1, gf_pow(2, index)])
    return tuple(generator)


def _encode_systematic(data: bytes, nsym: int) -> bytes:
    generator = _generator_polynomial(nsym)
    parity = [0] * nsym
    for data_value in data:
        feedback = data_value ^ parity[0]
        parity = parity[1:] + [0]
        for index in range(nsym):
            parity[index] ^= gf_mul(generator[index + 1], feedback)
    return data + bytes(parity)


@lru_cache(maxsize=None)
def _parity_matrix(k: int, nsym: int) -> Tuple[Tuple[int, ...], ...]:
    rows = []
    for data_index in range(k):
        basis = bytearray(k)
        basis[data_index] = 1
        encoded = _encode_systematic(bytes(basis), nsym)
        rows.append(encoded[k:])
    return tuple(tuple(rows[data_index][parity_index] for data_index in range(k))
                 for parity_index in range(nsym))


def encode(data: Iterable[int], n: int, k: int) -> bytes:
    """Return one systematic RS codeword of exactly ``n`` symbols."""
    nsym = _validate_parameters(n, k)
    data_bytes = _validate_bytes(data)
    if len(data_bytes) != k:
        raise ValueError("data length must equal k; pad blocks before encoding")
    return _encode_systematic(data_bytes, nsym)


def _solve_erasure_system(
    codeword: List[int], erasures: List[int], k: int, nsym: int
) -> Tuple[List[int], bool]:
    unknown = sorted(set(erasures))
    unknown_index = {position: index for index, position in enumerate(unknown)}
    parity_matrix = _parity_matrix(k, nsym)
    rows = []

    for parity_index in range(nsym):
        coefficients = [0] * len(unknown)
        for position in unknown:
            if position < k:
                coefficients[unknown_index[position]] = parity_matrix[parity_index][position]
            elif position == k + parity_index:
                coefficients[unknown_index[position]] = 1

        rhs = codeword[k + parity_index] if k + parity_index not in unknown_index else 0
        for data_index in range(k):
            if data_index not in unknown_index:
                rhs ^= gf_mul(parity_matrix[parity_index][data_index], codeword[data_index])
        rows.append(coefficients + [rhs])

    row = 0
    pivots = {}
    for column in range(len(unknown)):
        pivot = next((candidate for candidate in range(row, len(rows))
                      if rows[candidate][column] != 0), None)
        if pivot is None:
            continue
        rows[row], rows[pivot] = rows[pivot], rows[row]
        inverse = gf_pow(rows[row][column], 254)
        rows[row] = [gf_mul(value, inverse) for value in rows[row]]
        for candidate in range(len(rows)):
            if candidate == row or rows[candidate][column] == 0:
                continue
            factor = rows[candidate][column]
            rows[candidate] = [value ^ gf_mul(factor, pivot_value)
                               for value, pivot_value in zip(rows[candidate], rows[row])]
        pivots[column] = row
        row += 1

    if len(pivots) != len(unknown):
        return codeword, False

    recovered = list(codeword)
    for column, pivot_row in pivots.items():
        recovered[unknown[column]] = rows[pivot_row][-1]
    return recovered, True


def decode(
    codeword: Iterable[int],
    erasure_positions: Sequence[int],
    n: int,
    k: int,
) -> Tuple[bytes, bool]:
    """Recover a codeword with known erasures and return ``(data, success)``."""
    nsym = _validate_parameters(n, k)
    values = list(_validate_bytes(codeword))
    if len(values) != n:
        raise ValueError("codeword length must equal n")
    erasures = sorted(set(erasure_positions))
    if any(position < 0 or position >= n for position in erasures):
        raise ValueError("erasure positions must be valid codeword positions")
    if len(erasures) > nsym:
        return bytes(values[:k]), False
    recovered, success = _solve_erasure_system(values, erasures, k, nsym)
    if not success:
        return bytes(values[:k]), False
    expected = _encode_systematic(bytes(recovered[:k]), nsym)
    if bytes(recovered) != expected:
        return bytes(values[:k]), False
    return bytes(recovered[:k]), True