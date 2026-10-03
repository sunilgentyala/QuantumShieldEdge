"""EAN-side aggregation rules. Each takes the (n, P) matrix of leaf updates in one cluster, per-leaf sample
counts, and returns (aggregate (P,), effective weight scalar, flags (n,) bool or None)."""

from __future__ import annotations

import torch

from .vqad.features import fingerprints


def fedavg(U, n_samples, **_):
    w = n_samples / n_samples.sum()
    return (w[:, None] * U).sum(0), float(n_samples.sum()), None


def _krum_scores(U, f):
    n = len(U)
    d = torch.cdist(U, U) ** 2
    m = max(1, n - f - 2)
    return torch.stack([d[i][torch.arange(n) != i].topk(m, largest=False).values.sum() for i in range(n)])


def krum(U, n_samples, f=2, **_):
    i = int(_krum_scores(U, f).argmin())
    flags = torch.ones(len(U), dtype=torch.bool)
    flags[i] = False
    return U[i], float(n_samples.sum()), flags.numpy()


def multi_krum(U, n_samples, f=2, **_):
    n = len(U)
    sel = _krum_scores(U, f).argsort()[: max(1, n - f)]
    flags = torch.ones(n, dtype=torch.bool)
    flags[sel] = False
    w = n_samples[sel] / n_samples[sel].sum()
    return (w[:, None] * U[sel]).sum(0), float(n_samples.sum()), flags.numpy()


def coord_median(U, n_samples, **_):
    return U.median(0).values, float(n_samples.sum()), None


def trimmed_mean(U, n_samples, beta=0.2, **_):
    n = len(U)
    k = int(beta * n)
    s = U.sort(0).values
    return s[k:n - k].mean(0), float(n_samples.sum()), None


def detector_weighted(U, n_samples, detector=None, flagged_weight=0.1, **_):
    X = fingerprints(U).numpy()
    flags = detector.flag(X)
    w = n_samples * torch.where(torch.tensor(flags), flagged_weight, 1.0).double()
    w = w / w.sum()
    return (w[:, None] * U.double()).sum(0).float(), float(n_samples.sum()), flags


AGGREGATORS = {
    "fedavg": fedavg,
    "krum": krum,
    "multi_krum": multi_krum,
    "median": coord_median,
    "trimmed_mean": trimmed_mean,
    "vqad": detector_weighted,
    "mlp": detector_weighted,
    "linear": detector_weighted,
}
