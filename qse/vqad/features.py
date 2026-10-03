"""Four-dimensional, scale-free fingerprint of each leaf update, computed relative to its own cluster.

All features lie in (-1, 1) so they can be angle-encoded (x -> pi x).
  f0  tanh(log(||u_i|| / median_j ||u_j||))            relative magnitude
  f1  cos(u_i, m)                                      alignment with the cluster coordinate-wise median m
  f2  tanh(0.5 * robust z of excess kurtosis of u_i)   tail heaviness relative to the cluster
  f3  tanh(10 * (pos_i - median_j pos_j))              positive-fraction relative to the cluster
The EAN sees the raw update (it must, to aggregate it); the detector, and any QPU behind it, sees only these
four scalars per update.
"""

from __future__ import annotations

import torch


def _excess_kurtosis(U: torch.Tensor) -> torch.Tensor:
    c = U - U.mean(1, keepdim=True)
    m2 = (c ** 2).mean(1)
    m4 = (c ** 4).mean(1)
    return m4 / (m2 ** 2 + 1e-30) - 3.0


def fingerprints(U: torch.Tensor) -> torch.Tensor:
    """U: (n, P) cluster update matrix -> (n, 4) float64 in (-1, 1)."""
    U = U.double()
    n = U.shape[0]
    norms = U.norm(dim=1) + 1e-12
    ref = U.median(dim=0).values
    f0 = torch.tanh(torch.log(norms / norms.median()))
    f1 = (U @ ref) / (norms * (ref.norm() + 1e-12))
    k = _excess_kurtosis(U)
    mad = (k - k.median()).abs().median() * 1.4826 + 1e-6
    f2 = torch.tanh(0.5 * (k - k.median()) / mad)
    pos = (U > 0).double().mean(1)
    f3 = torch.tanh(10.0 * (pos - pos.median()))
    return torch.stack([f0, f1, f2, f3], 1).clamp(-1, 1)
