"""CIFAR-10 loading and Dirichlet non-IID partitioning (actually implemented and tested)."""

from __future__ import annotations

import hashlib
import io
import pickle
from pathlib import Path

import numpy as np
import torch

_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


# SHA-256 of the official CIFAR-10 python batches; a file that does not match is refused before it is parsed.
_SHA256 = {
    "data_batch_1": "54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd",
    "data_batch_2": "766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8",
    "data_batch_3": "0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9",
    "data_batch_4": "3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b",
    "data_batch_5": "d91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154",
    "test_batch": "f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831",
}

_SAFE_GLOBALS = {
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    """The CIFAR-10 batches are pickles of dicts of bytes, lists and numpy arrays. Anything else (a callable such
    as os.system, for instance) is refused, so a tampered or substituted cache file cannot run code on load."""

    def find_class(self, module, name):
        if (module, name) in _SAFE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a data file")


def _default_root() -> Path:
    return Path(__file__).resolve().parent.parent / "data_cache" / "cifar-10-batches-py"


def load_cifar10(root: str | Path | None = None):
    """Return (xtr, ytr, xte, yte) as normalised float32 NCHW tensors and int64 labels."""
    root = Path(root) if root else _default_root()

    def read(name):
        with open(root / name, "rb") as f:
            raw = f.read()
        if hashlib.sha256(raw).hexdigest() != _SHA256[name]:
            raise ValueError(f"{name}: SHA-256 does not match the official CIFAR-10 file; refusing to load it")
        d = _RestrictedUnpickler(io.BytesIO(raw), encoding="bytes").load()
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
    if sum(cluster_sizes) != n_clients:
        raise ValueError("cluster sizes must sum to the number of clients")
    perm = rng.permutation(n_clients)
    out, s = [], 0
    for sz in cluster_sizes:
        out.append(np.sort(perm[s:s + sz]))
        s += sz
    return out
