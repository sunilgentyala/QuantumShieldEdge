"""Protocol-level BB84 Monte Carlo (prepare-and-measure, no photon loss, no decoy states).

Purpose: validate the sifting / QBER-estimation / abort logic and show that an intercept-resend eavesdropper is
detected through QBER. Secret-key *rates* used by the FL study come from finite_key.py, not from this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BB84Result:
    n_pulses: int
    n_sifted: int
    n_sample: int
    qber_est: float
    qber_true_key: float
    aborted: bool
    key_alice: np.ndarray  # sifted bits not used for estimation (Alice)
    key_bob: np.ndarray


def run_bb84(n_pulses: int, channel_flip: float, rng: np.random.Generator, eve_fraction: float = 0.0,
             sample_frac: float = 0.1, abort_qber: float = 0.11) -> BB84Result:
    """One BB84 session.

    channel_flip : probability that the channel flips a bit when Alice and Bob use the same basis.
    eve_fraction : fraction of pulses subjected to an intercept-resend attack in a uniformly random basis.
    """
    a_bits = rng.integers(0, 2, n_pulses, dtype=np.uint8)
    a_bas = rng.integers(0, 2, n_pulses, dtype=np.uint8)
    b_bas = rng.integers(0, 2, n_pulses, dtype=np.uint8)

    # Qubit as (bit, basis). Eve measures in a random basis and resends what she measured.
    bit_on_wire = a_bits.copy()
    basis_on_wire = a_bas.copy()
    attacked = rng.random(n_pulses) < eve_fraction
    e_bas = rng.integers(0, 2, n_pulses, dtype=np.uint8)
    e_meas = np.where(e_bas == basis_on_wire, bit_on_wire, rng.integers(0, 2, n_pulses, dtype=np.uint8))
    bit_on_wire = np.where(attacked, e_meas, bit_on_wire)
    basis_on_wire = np.where(attacked, e_bas, basis_on_wire)

    # Channel noise (basis-independent flip applied to the bit as measured in the sender's frame).
    flipped = rng.random(n_pulses) < channel_flip
    bit_on_wire = bit_on_wire ^ flipped.astype(np.uint8)

    # Bob measures: same basis as the wire state gives the bit, otherwise a random outcome.
    b_bits = np.where(b_bas == basis_on_wire, bit_on_wire, rng.integers(0, 2, n_pulses, dtype=np.uint8))

    sift = a_bas == b_bas
    ka, kb = a_bits[sift], b_bits[sift]
    n_sift = len(ka)
    n_s = max(1, int(sample_frac * n_sift))
    perm = rng.permutation(n_sift)
    s_idx, k_idx = perm[:n_s], perm[n_s:]
    q_est = float(np.mean(ka[s_idx] != kb[s_idx]))
    q_key = float(np.mean(ka[k_idx] != kb[k_idx])) if len(k_idx) else 0.0
    return BB84Result(n_pulses, n_sift, n_s, q_est, q_key, q_est > abort_qber, ka[k_idx], kb[k_idx])
