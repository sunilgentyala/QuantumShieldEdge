"""CIFAR-10 loading and Dirichlet non-IID partitioning (actually implemented and tested)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def _default_root() -> Path:
    return Path(__file__).resolve().parent.parent / "data_cache" / "cifar-10-batches-py"


def load_cifar10(root: str | Path | None = None):
    """Return (xtr, ytr, xte, yte) as normalised float32 NCHW tensors and int64 labels."""
    root = Path(root) if root else _default_root()

    def read(name):
        with open(root / name, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        x = d[b"data"].reshape(-1, 3, 32, 32)
        return x, np.array(d[b"labels"], dtype=np.int64)

    xs, ys = zip(*[read(f"data_batch_{i}") for i in range(1, 6)])
    xtr, ytr = np.concatenate(xs), np.concatenate(ys)
    xte, yte = read("test_batch")

    def prep(x):
        t = torch.from_numpy(x).float() / 255.0
        return ((t - _MEAN) / _STD).contiguous()

    return prep(xtr), torch.from_numpy(ytr), prep(xte), torch.from_numpy(yte)


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float, rng: np.random.Generator,
                        min_size: int = 50) -> list[np.ndarray]:
    """Label-skew Dirichlet partition (Hsu et al. 2019 style). Resamples until every client has >= min_size."""
    labels = np.asarray(labels)
    n_classes = int(labels.max()) + 1
    while True:
        parts: list[list[int]] = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(sorted(p), dtype=np.int64) for p in parts]


def assign_clusters(n_clients: int, cluster_sizes: list[int], rng: np.random.Generator) -> list[np.ndarray]:
    """Randomly assign client ids to EAN clusters with the given sizes."""
    assert sum(cluster_sizes) == n_clients
    perm = rng.permutation(n_clients)
    out, s = [], 0
    for sz in cluster_sizes:
        out.append(np.sort(perm[s:s + sz]))
        s += sz
    return out
