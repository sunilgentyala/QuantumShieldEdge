"""One-time-pad encryption + Wegman-Carter authentication, and the sparse-update wire format.

Encrypt-then-MAC. The MAC is a polynomial hash over GF(p), p = 2^61 - 1, with a long-lived hash key whose
output is one-time-padded with 64 fresh key bits per message (Wegman-Carter). Forgery probability per
attempt is <= (L+1)/p with L the number of 8-byte message blocks, so the security of the channel is
information-theoretic given uniformly random, secret, never-reused key bits.
"""

from __future__ import annotations

import hmac
import struct

import numpy as np

P61 = (1 << 61) - 1
TAG_BYTES = 8
HEADER_BYTES = 12  # round (u32), entry count (u32), flags (u32)


def poly_hash(data: bytes, key: int) -> int:
    """Polynomial evaluation over GF(2^61-1) on 7-byte blocks (each block < 2^56 < p, so no folding)."""
    pad = (-len(data)) % 7
    buf = np.frombuffer(data + b"\x00" * pad, dtype=np.uint8).reshape(-1, 7)
    wide = np.zeros((len(buf), 8), dtype=np.uint8)
    wide[:, :7] = buf
    h = 0
    for b in wide.view("<u8").reshape(-1).tolist():
        h = (h * key + b) % P61
    return (h * key + len(data)) % P61  # length block prevents zero-padding ambiguity


class AuthError(Exception):
    pass


class FormatError(ValueError):
    """A (decrypted) payload is malformed: wrong length, out-of-range header or index."""


def xor_bytes(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):  # explicit check: an assert would be removed under python -O
        raise ValueError("xor_bytes: length mismatch")
    return (np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)).tobytes()


def _check_mac_key(mac_key: int) -> None:
    if not (1 <= mac_key < P61):
        raise ValueError("mac_key must be in [1, 2^61 - 2]")


def seal(payload: bytes, key_pad: bytes, tag_pad: bytes, mac_key: int) -> bytes:
    """Return ciphertext || tag. key_pad has len(payload) bytes, tag_pad has TAG_BYTES bytes."""
    _check_mac_key(mac_key)
    if len(tag_pad) != TAG_BYTES:
        raise ValueError("tag_pad must be exactly TAG_BYTES")
    ct = xor_bytes(payload, key_pad)
    t = poly_hash(ct, mac_key) ^ int.from_bytes(tag_pad, "little")
    return ct + t.to_bytes(TAG_BYTES, "little")


def open_sealed(msg: bytes, key_pad: bytes, tag_pad: bytes, mac_key: int) -> bytes:
    _check_mac_key(mac_key)
    if len(msg) < TAG_BYTES or len(tag_pad) != TAG_BYTES:
        raise AuthError("message too short")
    ct, tag = msg[:-TAG_BYTES], msg[-TAG_BYTES:]
    if len(key_pad) != len(ct):
        raise AuthError("pad length mismatch")
    expect = (poly_hash(ct, mac_key) ^ int.from_bytes(tag_pad, "little")).to_bytes(TAG_BYTES, "little")
    if not hmac.compare_digest(expect, tag):  # constant-time comparison; decrypt only after verification
        raise AuthError("tag mismatch")
    return xor_bytes(ct, key_pad)


def index_bits(n_params: int) -> int:
    return max(1, int(np.ceil(np.log2(n_params))))


def pack_sparse(round_id: int, idx: np.ndarray, val: np.ndarray, n_params: int) -> bytes:
    """Pack (index, fp16 value) pairs. Indices use ceil(log2 P) bits each, little-endian bit packing."""
    ib = index_bits(n_params)
    idx = idx.astype(np.uint64)
    bits = ((idx[:, None] >> np.arange(ib, dtype=np.uint64)) & 1).astype(np.uint8).reshape(-1)
    idx_bytes = np.packbits(bits, bitorder="little").tobytes()
    val_bytes = val.astype("<f2").tobytes()
    return struct.pack("<III", round_id, len(idx), 0) + idx_bytes + val_bytes


def unpack_sparse(data: bytes, n_params: int):
    ib = index_bits(n_params)
    if len(data) < HEADER_BYTES:
        raise FormatError("truncated header")
    round_id, n, _ = struct.unpack("<III", data[:HEADER_BYTES])
    if n > n_params:
        raise FormatError("entry count exceeds the parameter count")
    n_idx_bytes = (n * ib + 7) // 8
    if len(data) != HEADER_BYTES + n_idx_bytes + 2 * n:
        raise FormatError("payload length does not match the header")
    ib_data = np.frombuffer(data[HEADER_BYTES:HEADER_BYTES + n_idx_bytes], np.uint8)
    bits = np.unpackbits(ib_data, bitorder="little")[: n * ib].reshape(n, ib).astype(np.uint64)
    idx = (bits << np.arange(ib, dtype=np.uint64)).sum(1).astype(np.int64)
    val = np.frombuffer(data[HEADER_BYTES + n_idx_bytes:HEADER_BYTES + n_idx_bytes + 2 * n], dtype="<f2")
    if n and int(idx.max()) >= n_params:
        raise FormatError("index out of range")
    return round_id, idx, val.astype(np.float32)


def pack_dense(round_id: int, val: np.ndarray) -> bytes:
    return struct.pack("<III", round_id, len(val), 1) + val.astype("<f2").tobytes()


def unpack_dense(data: bytes, n_params: int | None = None):
    if len(data) < HEADER_BYTES:
        raise FormatError("truncated header")
    round_id, n, _ = struct.unpack("<III", data[:HEADER_BYTES])
    if len(data) != HEADER_BYTES + 2 * n or (n_params is not None and n != n_params):
        raise FormatError("payload length does not match the header")
    return round_id, np.frombuffer(data[HEADER_BYTES:HEADER_BYTES + 2 * n], dtype="<f2").astype(np.float32)


def sparse_payload_bits(k: int, n_params: int) -> int:
    ib = index_bits(n_params)
    return 8 * (HEADER_BYTES + (k * ib + 7) // 8 + 2 * k)


def dense_payload_bits(n_params: int) -> int:
    return 8 * (HEADER_BYTES + 2 * n_params)


def message_key_bits(payload_bits: int) -> int:
    """Key consumed by one sealed message: payload pad plus the 64-bit MAC pad."""
    return payload_bits + 8 * TAG_BYTES
