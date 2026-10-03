"""Key pools and QKD link model.

A QKD link produces secret-key bits at rate R(t) = sifted_rate(t) * fraction(Q(t), n). Both endpoints hold a
KeyPool; they are fed the *same* bytes. The bytes stand in for QKD output (drawn from a seeded PRNG
here, because no physical link exists); everything downstream (OTP, MAC) uses them as real key material.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .finite_key import asymptotic_fraction, finite_fraction


class KeyPool:
    def __init__(self, capacity_bits: int) -> None:
        self.capacity_bits = int(capacity_bits)
        self._q: deque[bytes] = deque()
        self._bits = 0

    def level_bits(self) -> int:
        return self._bits

    def level(self) -> float:
        return self._bits / self.capacity_bits

    def deposit(self, data: bytes) -> int:
        """Append key bytes, truncated to remaining capacity. Returns bits actually stored."""
        room = (self.capacity_bits - self._bits) // 8
        data = data[:room]
        if data:
            self._q.append(data)
            self._bits += 8 * len(data)
        return 8 * len(data)

    def take(self, n_bytes: int) -> bytes | None:
        """Atomically dequeue n_bytes or return None (never partial)."""
        if 8 * n_bytes > self._bits:
            return None
        out = bytearray()
        need = n_bytes
        while need:
            chunk = self._q[0]
            if len(chunk) <= need:
                out += chunk
                need -= len(chunk)
                self._q.popleft()
            else:
                out += chunk[:need]
                self._q[0] = chunk[need:]
                need = 0
        self._bits -= 8 * n_bytes
        return bytes(out)


@dataclass
class LinkConfig:
    sifted_kbps_mean: float = 60.0      # sifted-key rate before error correction / privacy amplification
    sifted_kbps_std: float = 6.0
    qber_mean: float = 0.03
    qber_std: float = 0.01
    block_bits: int | None = None       # None -> asymptotic; else finite-key block length n
    f_ec: float = 1.16
    qber_override: float | None = None  # fixed QBER (stress tests)
    rate_scale: float = 1.0             # multiplies the secret rate (key-rate sweeps)


class QKDLink:
    """One EAN <-> cloud QKD link. `advance(dt)` generates dt seconds of secret key into both pools."""

    def __init__(self, cfg: LinkConfig, capacity_bits: int, seed: int) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.key_rng = np.random.default_rng(seed + 7919)
        self.ean_pool = KeyPool(capacity_bits)
        self.cloud_pool = KeyPool(capacity_bits)
        self._carry_bits = 0.0
        self.last_qber = 0.0
        self.last_rate_bps = 0.0

    def sample_state(self) -> tuple[float, float]:
        c = self.cfg
        if c.qber_override is not None:
            q = c.qber_override
        else:
            q = float(np.clip(self.rng.normal(c.qber_mean, c.qber_std), 0.002, 0.2))
        sifted = max(0.0, float(self.rng.normal(c.sifted_kbps_mean, c.sifted_kbps_std))) * 1000.0
        frac = (asymptotic_fraction(q, c.f_ec) if c.block_bits is None
                else finite_fraction(q, c.block_bits, f_ec=c.f_ec))
        return q, sifted * float(frac) * c.rate_scale

    def advance(self, dt: float) -> int:
        q, rate = self.sample_state()
        self.last_qber, self.last_rate_bps = q, rate
        self._carry_bits += rate * dt
        n_bytes = int(self._carry_bits // 8)
        if n_bytes == 0:
            return 0
        self._carry_bits -= 8 * n_bytes
        data = self.key_rng.bytes(n_bytes)  # stand-in for QKD output, identical at both ends
        stored = self.ean_pool.deposit(data)
        self.cloud_pool.deposit(data[: stored // 8])
        return stored
