"""
QKD key buffer with proactive re-keying.

Implements REQUEST_KEY(size) and KEY_LEVEL() operations described in Section IV-B.
Re-keying is triggered when buffer occupancy falls below REKEY_THRESHOLD (40%).
"""

import threading
from collections import deque
import numpy as np

REKEY_THRESHOLD = 0.40  # trigger re-keying below this occupancy


class KeyBuffer:
    """
    Thread-safe key buffer backed by a BB84Emulator.

    Pre-loads at least `prefill_rounds` rounds of keying material on
    construction and replenishes proactively whenever occupancy drops
    below REKEY_THRESHOLD, decoupling QKD timing from FL round timing.
    """

    def __init__(
        self,
        emulator,
        capacity_bits: int,
        round_key_bits: int,
        prefill_rounds: int = 3,
        seed: int | None = None,
    ) -> None:
        self._emulator = emulator
        self._capacity_bits = capacity_bits
        self._round_key_bits = round_key_bits
        self._buffer: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._total_bits_stored = 0

        # Pre-fill with at least prefill_rounds of material
        for _ in range(prefill_rounds):
            self._fill_buffer()

    # ------------------------------------------------------------------
    # Public interface (paper Section IV-B)
    # ------------------------------------------------------------------

    def request_key(self, n_bits: int) -> bytes:
        """
        REQUEST_KEY(size) — atomically dequeue n_bits of key material.

        Raises BufferError if insufficient key material is available.
        After dequeue, triggers proactive re-keying if occupancy < threshold.
        """
        n_bytes = (n_bits + 7) // 8
        with self._lock:
            available = self._available_bytes()
            if available < n_bytes:
                raise BufferError(
                    f"Key buffer underrun: need {n_bytes} B, have {available} B."
                )
            key = self._dequeue_bytes(n_bytes)
            self._total_bits_stored -= n_bytes * 8

        if self.key_level() < REKEY_THRESHOLD:
            self._fill_buffer()

        return key

    def key_level(self) -> float:
        """KEY_LEVEL() — return current buffer occupancy in [0, 1]."""
        with self._lock:
            return self._total_bits_stored / self._capacity_bits

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fill_buffer(self) -> None:
        try:
            rate_bps, qber = self._emulator.sample_channel_state()
            key_bytes = self._emulator.generate_key_bits(
                self._round_key_bits, rate_bps, qber
            )
            with self._lock:
                self._buffer.append(key_bytes)
                self._total_bits_stored += len(key_bytes) * 8
        except RuntimeError:
            # High QBER: skip this fill cycle; buffer retains existing material
            pass

    def _available_bytes(self) -> int:
        return sum(len(chunk) for chunk in self._buffer)

    def _dequeue_bytes(self, n_bytes: int) -> bytes:
        result = bytearray()
        remaining = n_bytes
        while remaining > 0 and self._buffer:
            chunk = bytearray(self._buffer[0])
            take = min(remaining, len(chunk))
            result.extend(chunk[:take])
            remaining -= take
            if take == len(chunk):
                self._buffer.popleft()
            else:
                self._buffer[0] = bytes(chunk[take:])
        return bytes(result)
