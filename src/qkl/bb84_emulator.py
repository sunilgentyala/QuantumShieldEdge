"""
BB84 decoy-state QKD emulator.

Simulates key generation with realistic QBER and key rate statistics
matching metropolitan QKD infrastructure characteristics used in the paper.
"""

import numpy as np
from typing import Tuple


class BB84Emulator:
    """
    Decoy-state BB84 protocol emulator for the QKD Key Management Layer.

    Key rates sampled from N(mean_kbps, std_kbps^2); QBER sampled uniformly
    from [qber_min, qber_max]. Abort threshold follows standard BB84 security
    analysis (QBER <= 11% for information-theoretic security).
    """

    ABORT_QBER = 0.11  # standard BB84 abort threshold

    def __init__(
        self,
        mean_kbps: float = 40.0,
        std_kbps: float = 8.0,
        qber_min: float = 0.01,
        qber_max: float = 0.09,
        seed: int | None = None,
    ) -> None:
        self.mean_kbps = mean_kbps
        self.std_kbps = std_kbps
        self.qber_min = qber_min
        self.qber_max = qber_max
        self.rng = np.random.default_rng(seed)

    def sample_channel_state(self) -> Tuple[float, float]:
        """Return (key_rate_bps, qber) for one FL round."""
        kbps = float(np.clip(
            self.rng.normal(self.mean_kbps, self.std_kbps), 1.0, 200.0
        ))
        qber = float(self.rng.uniform(self.qber_min, self.qber_max))
        return kbps * 1000.0, qber  # convert to bps

    def generate_key_bits(self, n_bits: int, key_rate_bps: float, qber: float) -> bytes:
        """
        Simulate sifted + error-corrected + privacy-amplified key output.

        Returns n_bits of secure key material, or raises RuntimeError if
        QBER exceeds abort threshold (information-theoretic security boundary).
        """
        if qber > self.ABORT_QBER:
            raise RuntimeError(
                f"QBER {qber:.3f} exceeds abort threshold {self.ABORT_QBER}. "
                "Key generation aborted."
            )
        # Privacy amplification: secure key fraction ~ 1 - 2*h(QBER)
        # where h is binary entropy. We simulate the net output directly.
        h = self._binary_entropy(qber)
        secure_fraction = max(0.0, 1.0 - 2.0 * h)
        if secure_fraction < 0.1:
            raise RuntimeError(
                f"Secure key fraction {secure_fraction:.3f} too low at QBER {qber:.3f}."
            )
        n_bytes = (n_bits + 7) // 8
        return bytes(self.rng.integers(0, 256, size=n_bytes, dtype=np.uint8))

    @staticmethod
    def _binary_entropy(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
