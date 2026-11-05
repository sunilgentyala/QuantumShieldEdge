"""Test fixtures. The simulator tests only need CIFAR-shaped tensors, so when the CIFAR-10 cache is absent (CI, a
fresh clone) or QSE_SYNTHETIC_DATA=1 is set, random tensors of the same shape are used instead. This is for tests
only: experiments always read the real data."""

import os

import torch


def pytest_configure(config):
    from qse import data, sim

    if os.environ.get("QSE_SYNTHETIC_DATA") == "1" or not data._default_root().exists():
        g = torch.Generator().manual_seed(0)
        sim._DATA["d"] = (torch.randn(6000, 3, 32, 32, generator=g), torch.randint(0, 10, (6000,), generator=g),
                          torch.randn(1000, 3, 32, 32, generator=g), torch.randint(0, 10, (1000,), generator=g))
