"""
Lightweight CNN for CIFAR-10 (~180K parameters) used in Section VI-A.

Architecture: two conv blocks + global average pooling + linear classifier.
Compatible with PyTorch and Flower (flwr) parameter serialisation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCNN(nn.Module):
    """
    ~180K parameter CNN for CIFAR-10 (32x32 RGB, 10 classes).

    Conv1: 3->32 (3x3) -> BN -> ReLU -> MaxPool
    Conv2: 32->64 (3x3) -> BN -> ReLU -> MaxPool
    Global avg pool -> FC(64, 10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_parameters(self) -> list[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_parameters(self, params: list[np.ndarray]) -> None:
        for p, v in zip(self.parameters(), params):
            p.data = torch.tensor(v, dtype=p.dtype)

    @staticmethod
    def parameter_count() -> int:
        m = LightweightCNN()
        return sum(p.numel() for p in m.parameters())
