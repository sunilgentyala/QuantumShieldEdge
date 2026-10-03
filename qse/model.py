"""Compact CIFAR-10 CNN. GroupNorm (no running statistics) so the full model state is the parameter vector."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """Three stride-2 conv blocks (3x3, GroupNorm, ReLU), global average pooling, linear head.
    Default widths (16, 32, 64) give 24,458 parameters."""

    def __init__(self, widths=(16, 32, 64)) -> None:
        super().__init__()
        c1, c2, c3 = widths
        self.conv1 = nn.Conv2d(3, c1, 3, 2, 1)
        self.gn1 = nn.GroupNorm(4, c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, 2, 1)
        self.gn2 = nn.GroupNorm(4, c2)
        self.conv3 = nn.Conv2d(c2, c3, 3, 2, 1)
        self.gn3 = nn.GroupNorm(4, c3)
        self.fc = nn.Linear(c3, 10)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        return self.fc(x.mean(dim=(2, 3)))


def n_params(model: nn.Module | None = None) -> int:
    model = model or SmallCNN()
    return sum(p.numel() for p in model.parameters())


def get_flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def set_flat(model: nn.Module, flat: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, bs: int = 1000) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        correct += (model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(x)


def layer_breakdown(model: nn.Module | None = None) -> dict[str, int]:
    model = model or SmallCNN()
    return {n: p.numel() for n, p in model.named_parameters()}
