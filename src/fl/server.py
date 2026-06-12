"""
Flower FL server implementing the QSAL + VQAD aggregation pipeline.

On each FL round (Section IV-A):
  1. QSAL decrypts cluster gradients (simulated: receives plaintext)
  2. VQAD scores each gradient fingerprint
  3. Soft-weighted FedAvg aggregation (Byzantine updates downweighted by 0.1)
  4. Updated global model re-encrypted and broadcast downstream
"""

import numpy as np
from typing import Optional
import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from ..vqad import VQADetector
from .model import LightweightCNN


class QuantumShieldStrategy(FedAvg):
    """
    FedAvg variant with VQAD-based Byzantine soft-weighting.

    Extends Flower's FedAvg to intercept aggregate_fit and apply the
    VQAD detector scores before weighted averaging.
    """

    BYZANTINE_WEIGHT = 0.1

    def __init__(self, detector: VQADetector, global_model: LightweightCNN, **kwargs):
        super().__init__(**kwargs)
        self.detector = detector
        self.global_model = global_model
        self.round_metrics: list[dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        global_params = np.concatenate([
            p.flatten() for p in self.global_model.get_parameters()
        ])

        weighted_sum = None
        total_weight = 0.0
        detection_log = []

        for _, fit_res in results:
            client_params = parameters_to_ndarrays(fit_res.parameters)
            gradient = np.concatenate([p.flatten() for p in client_params])
            n_samples = fit_res.num_examples

            # VQAD scoring
            vqad_weight = self.detector.score(gradient, global_params)
            effective_weight = n_samples * vqad_weight

            if weighted_sum is None:
                weighted_sum = [effective_weight * p for p in client_params]
            else:
                for i, p in enumerate(client_params):
                    weighted_sum[i] += effective_weight * p

            total_weight += effective_weight
            is_byzantine = int(fit_res.metrics.get("is_byzantine", 0))
            detection_log.append({
                "client_id": fit_res.metrics.get("client_id", -1),
                "vqad_weight": vqad_weight,
                "is_byzantine": is_byzantine,
                "true_positive": int(is_byzantine == 1 and vqad_weight < 1.0),
                "false_positive": int(is_byzantine == 0 and vqad_weight < 1.0),
            })

        if weighted_sum is None or total_weight == 0:
            return None, {}

        aggregated = [p / total_weight for p in weighted_sum]

        # Update global model
        self.global_model.set_parameters(aggregated)
        global_params_new = np.concatenate([p.flatten() for p in aggregated])

        tp = sum(d["true_positive"] for d in detection_log)
        fp = sum(d["false_positive"] for d in detection_log)
        n_byz = sum(d["is_byzantine"] for d in detection_log)

        round_metric: dict[str, Scalar] = {
            "round": server_round,
            "true_positives": tp,
            "false_positives": fp,
            "n_byzantine": n_byz,
            "tpr": tp / n_byz if n_byz > 0 else 1.0,
            "fpr": fp / max(1, len(detection_log) - n_byz),
        }
        self.round_metrics.append(round_metric)

        return ndarrays_to_parameters(aggregated), round_metric


class QuantumShieldServer:
    """High-level server runner wrapping Flower's start_server."""

    def __init__(self, detector: VQADetector, n_rounds: int = 100) -> None:
        self.model = LightweightCNN()
        self.detector = detector
        self.n_rounds = n_rounds
        self.strategy = QuantumShieldStrategy(
            detector=detector,
            global_model=self.model,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )

    def run(self, server_address: str = "0.0.0.0:8080") -> None:
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=self.n_rounds),
            strategy=self.strategy,
        )
