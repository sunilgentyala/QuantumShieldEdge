"""Byzantine attacks. Each acts on the update matrix of the round; honest rows are visible to the adversary
(omniscient model, as in Shejwalkar & Houmansadr 2021), which is the strong setting for evaluating defenses.

  sign_flip : u -> -u
  scale     : u -> 10 u   (gradient boosting; a scalar multiple, easy to detect by norm)
  label_flip: data poisoning (labels y -> (y+1) mod 10); handled in the client trainer, update is untouched
  alie      : mean - z*std per coordinate (Baruch et al. 2019)
  min_max   : mean + gamma * dir, largest gamma keeping max distance to any honest update within the honest
              diameter (Shejwalkar & Houmansadr 2021, AGR-agnostic min-max)
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm

ATTACKS = ("sign_flip", "scale", "label_flip", "alie", "min_max")


def alie_z(n: int, f: int) -> float:
    s = int(np.floor(n / 2 + 1)) - f
    p = (n - s) / n
    return float(norm.ppf(min(max(p, 1e-6), 1 - 1e-6)))


def craft(attack: str, honest: torch.Tensor, own: torch.Tensor | None, n_total: int, n_byz: int) -> torch.Tensor:
    """Return one malicious update (same for all colluding attackers, except sign_flip/scale which use `own`)."""
    if attack == "sign_flip":
        return -own
    if attack == "scale":
        return 10.0 * own
    if attack == "label_flip":
        return own
    mu = honest.mean(0)
    if attack == "alie":
        z = alie_z(n_total, n_byz)
        return mu - z * honest.std(0, unbiased=False)
    if attack == "min_max":
        d = -mu / (mu.norm() + 1e-12)
        h = honest
        if len(h) > 40:  # diameter on a subsample keeps cost bounded; exact for <=40 honest rows
            h = h[torch.randperm(len(h))[:40]]
        diam = torch.cdist(h, h).max()
        lo, hi = 0.0, 1.0
        while (mu + hi * d - h).norm(dim=1).max() <= diam and hi < 1e6:
            lo, hi = hi, hi * 2
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if (mu + mid * d - h).norm(dim=1).max() <= diam:
                lo = mid
            else:
                hi = mid
        return mu + lo * d
    raise ValueError(attack)
