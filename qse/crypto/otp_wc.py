"""One-time-pad encryption + Wegman-Carter authentication, and the sparse-update wire format.

Encrypt-then-MAC. The MAC is a polynomial hash over GF(p), p = 2^61 - 1, with a long-lived hash key whose
output is one-time-padded with 64 fresh key bits per message (Wegman-Carter). Forgery probability per
attempt is <= (L+1)/p with L the number of 8-byte message blocks, so the security of the channel is
information-theoretic given uniformly random, secret, never-reused key bits.
"""

from __future__ import annotations

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


def xor_bytes(a: bytes, b: bytes) -> bytes:
    assert len(a) == len(b)
    return (np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)).tobytes()


def seal(payload: bytes, key_pad: bytes, tag_pad: bytes, mac_key: int) -> bytes:
    """Return ciphertext || tag. key_pad has len(payload) bytes, tag_pad has TAG_BYTES bytes."""
    ct = xor_bytes(payload, key_pad)
    t = poly_hash(ct, mac_key) ^ int.from_bytes(tag_pad, "little")
    return ct + t.to_bytes(TAG_BYTES, "little")


def open_sealed(msg: bytes, key_pad: bytes, tag_pad: bytes, mac_key: int) -> bytes:
    ct, tag = msg[:-TAG_BYTES], msg[-TAG_BYTES:]
    expect = poly_hash(ct, mac_key) ^ int.from_bytes(tag_pad, "little")
    if expect != int.from_bytes(tag, "little"):
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
    round_id, n, _ = struct.unpack("<III", data[:HEADER_BYTES])
    n_idx_bytes = (n * ib + 7) // 8
    ib_data = np.frombuffer(data[HEADER_BYTES:HEADER_BYTES + n_idx_bytes], np.uint8)
    bits = np.unpackbits(ib_data, bitorder="little")[: n * ib].reshape(n, ib).astype(np.uint64)
    idx = (bits << np.arange(ib, dtype=np.uint64)).sum(1).astype(np.int64)
    val = np.frombuffer(data[HEADER_BYTES + n_idx_bytes:HEADER_BYTES + n_idx_bytes + 2 * n], dtype="<f2")
    return round_id, idx, val.astype(np.float32)


def pack_dense(round_id: int, val: np.ndarray) -> bytes:
    return struct.pack("<III", round_id, len(val), 1) + val.astype("<f2").tobytes()


def unpack_dense(data: bytes):
    round_id, n, _ = struct.unpack("<III", data[:HEADER_BYTES])
    return round_id, np.frombuffer(data[HEADER_BYTES:HEADER_BYTES + 2 * n], dtype="<f2").astype(np.float32)


def sparse_payload_bits(k: int, n_params: int) -> int:
    ib = index_bits(n_params)
    return 8 * (HEADER_BYTES + (k * ib + 7) // 8 + 2 * k)


def dense_payload_bits(n_params: int) -> int:
    return 8 * (HEADER_BYTES + 2 * n_params)


def message_key_bits(payload_bits: int) -> int:
    """Key consumed by one sealed message: payload pad plus the 64-bit MAC pad."""
    return payload_bits + 8 * TAG_BYTES
