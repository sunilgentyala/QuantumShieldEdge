"""Secret-key fraction models for BB84-type QKD.

asymptotic : r = max(0, 1 - h(Q) - f_ec h(Q))                 (Shor-Preskill form with EC inefficiency f_ec)
finite     : Tomamichel-et-al.-style statistical penalty. With n key bits and k parameter-estimation bits,
             Q is replaced by Q + mu,  mu = sqrt( (n+k)/(n k) * (k+1)/k * ln(1/eps_sec) ),
             and a fixed privacy-amplification/correctness cost is subtracted:
                 ell = n (1 - h(Q+mu)) - leak_ec - log2(2 / (eps_sec^2 eps_cor)),   leak_ec = f_ec n h(Q)
This is a *modelling choice* following the structure of the tight finite-key bound; it is not a security
proof for a specific implementation (decoy-state parameter estimation is not modelled).
"""

from __future__ import annotations

import numpy as np


def h2(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def asymptotic_fraction(q, f_ec: float = 1.16):
    return np.maximum(0.0, 1.0 - h2(q) - f_ec * h2(q))


def asymptotic_threshold(f_ec: float = 1.16) -> float:
    """QBER at which the asymptotic fraction reaches zero (f_ec=1 gives the classical 11.0% bound)."""
    lo, hi = 1e-4, 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 1.0 - (1.0 + f_ec) * float(h2(mid)) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def finite_fraction(q, n: int, k_frac: float = 0.1, f_ec: float = 1.16, eps_sec: float = 1e-10,
                    eps_cor: float = 1e-15):
    """Secret fraction ell/n for a block of N=n/(1-k_frac) sifted bits (k_frac used for estimation)."""
    q = np.asarray(q, dtype=float)
    k = max(1.0, n * k_frac / (1.0 - k_frac))
    mu = np.sqrt((n + k) / (n * k) * (k + 1) / k * np.log(1.0 / eps_sec))
    ell = n * (1.0 - h2(np.minimum(q + mu, 0.5))) - f_ec * n * h2(q) - np.log2(2.0 / (eps_sec ** 2 * eps_cor))
    return np.maximum(0.0, ell / n)
