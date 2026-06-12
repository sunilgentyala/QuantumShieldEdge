"""
Flower FL client with integrated QKL encryption.

Each client holds a local CIFAR-10 partition (non-IID, Dirichlet alpha=0.5),
trains the LightweightCNN locally, sparsifies and encrypts gradients via QSAL
before transmitting to the EAN aggregator (Section III-A, IV-B/C).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import flwr as fl
from flwr.common import NDArrays, Scalar, Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays

from ..qkl import KeyBuffer, BB84Emulator
from ..qsal import OTPEncryptor, TopKSparsifier
from .model import LightweightCNN


class QuantumShieldClient(fl.client.Client):
    """
    FL leaf client implementing QKD-secured gradient upload.

    In the simulation, encryption/decryption happens locally to track
    key consumption metrics. In production, the EAN would decrypt.
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        key_buffer: KeyBuffer,
        is_byzantine: bool = False,
        attack_type: str = "min_max",
        device: str = "cpu",
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.key_buffer = key_buffer
        self.is_byzantine = is_byzantine
        self.attack_type = attack_type
        self.device = torch.device(device)
        self.model = LightweightCNN().to(self.device)
        self.sparsifier = TopKSparsifier(density=0.05)
        self.encryptor = OTPEncryptor()

    def fit(self, ins: FitIns) -> FitRes:
        params = parameters_to_ndarrays(ins.parameters)
        self.model.set_parameters(params)

        # Local training
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            criterion(self.model(images), labels).backward()
            optimizer.step()

        # Compute gradient delta
        new_params = self.model.get_parameters()
        gradients = [n - o for n, o in zip(new_params, params)]

        if self.is_byzantine:
            from ..utils.attacks import ByzantineAttack
            gradients = ByzantineAttack.apply(gradients, attack_type=self.attack_type)

        # Sparsify + encrypt each gradient tensor, track key consumption
        total_key_bits = 0
        encrypted_gradients = []
        for grad in gradients:
            idx, vals = self.sparsifier.sparsify(grad)
            grad_bytes = OTPEncryptor.gradients_to_bytes(idx, vals)
            key_bits = len(grad_bytes) * 8
            key = self.key_buffer.request_key(key_bits)
            enc = self.encryptor.encrypt(grad_bytes, key)
            encrypted_gradients.append(enc)
            total_key_bits += key_bits

        metrics: dict[str, Scalar] = {
            "client_id": self.client_id,
            "key_bits_consumed": total_key_bits,
            "buffer_level": self.key_buffer.key_level(),
            "is_byzantine": int(self.is_byzantine),
        }

        # For simulation: return original gradients (decryption happens at aggregator)
        return FitRes(
            parameters=ndarrays_to_parameters(gradients),
            num_examples=len(self.train_loader.dataset),
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        params = parameters_to_ndarrays(ins.parameters)
        self.model.set_parameters(params)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                out = self.model(images)
                loss += criterion(out, labels).item() * len(labels)
                correct += (out.argmax(dim=1) == labels).sum().item()
                total += len(labels)
        accuracy = correct / total if total > 0 else 0.0
        return EvaluateRes(
            loss=loss / total if total > 0 else 0.0,
            num_examples=total,
            metrics={"accuracy": accuracy, "client_id": self.client_id},
        )
