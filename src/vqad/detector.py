"""
Variational Quantum Anomaly Detector (VQAD).

Wraps the VQC circuit with training (parameter-shift + Adam) and inference.
Byzantine participants are flagged when <Z> < 0 and down-weighted by 0.1
in the FedAvg aggregation step (Section IV-D).
"""

import numpy as np
from pathlib import Path
from .circuit import make_vqc, random_params, N_LAYERS, N_QUBITS
from .fingerprint import GradientFingerprint

BYZANTINE_WEIGHT = 0.1
DETECTION_THRESHOLD = 0.0  # <Z> < threshold => Byzantine


class VQADetector:
    """
    Trains and runs the 4-qubit VQC-based Byzantine anomaly detector.

    Usage
    -----
    detector = VQADetector()
    detector.train(benign_gradients, global_model, byzantine_gradients)
    weight = detector.score(gradient, global_model)  # returns 1.0 or BYZANTINE_WEIGHT
    """

    def __init__(self, seed: int | None = 42) -> None:
        self._params = random_params(seed)
        self._circuit = make_vqc()
        self._fingerprint = GradientFingerprint()

    def train(
        self,
        benign_gradients: list[np.ndarray],
        byzantine_gradients: list[np.ndarray],
        global_model: np.ndarray,
        n_epochs: int = 100,
        lr: float = 0.01,
    ) -> list[float]:
        """
        Train the VQAD offline using the parameter-shift rule and Adam optimizer.

        Labels: benign -> +1, Byzantine -> -1 (matching the <Z> output sign).
        Returns training loss history.
        """
        X, y = self._build_dataset(benign_gradients, byzantine_gradients, global_model)

        # Adam state
        m = np.zeros_like(self._params)
        v = np.zeros_like(self._params)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        losses = []

        for epoch in range(n_epochs):
            grad = np.zeros_like(self._params)
            loss = 0.0
            for feat, label in zip(X, y):
                pred = self._circuit(feat, self._params)
                loss += (pred - label) ** 2
                # Parameter-shift rule: df/dp = (f(p+pi/2) - f(p-pi/2)) / 2
                for i in range(len(self._params)):
                    p_plus = self._params.copy()
                    p_plus[i] += np.pi / 2
                    p_minus = self._params.copy()
                    p_minus[i] -= np.pi / 2
                    grad[i] += 2 * (pred - label) * (
                        self._circuit(feat, p_plus) - self._circuit(feat, p_minus)
                    ) / 2

            loss /= len(X)
            grad /= len(X)
            losses.append(float(loss))

            # Adam update
            t = epoch + 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            self._params -= lr * m_hat / (np.sqrt(v_hat) + eps)

        return losses

    def score(self, gradient: np.ndarray, global_model: np.ndarray) -> float:
        """
        Return aggregation weight for this gradient update.

        1.0  => benign  (<Z> >= 0)
        0.1  => Byzantine flag (<Z> < 0)
        """
        feat = self._fingerprint.extract(gradient, global_model)
        z_exp = float(self._circuit(feat, self._params))
        return 1.0 if z_exp >= DETECTION_THRESHOLD else BYZANTINE_WEIGHT

    def save_weights(self, path: str | Path) -> None:
        np.save(str(path), self._params)

    def load_weights(self, path: str | Path) -> None:
        self._params = np.load(str(path))

    def _build_dataset(
        self,
        benign: list[np.ndarray],
        byzantine: list[np.ndarray],
        global_model: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        fp = GradientFingerprint()
        X, y = [], []
        for g in benign:
            X.append(fp.extract(g, global_model))
            y.append(1.0)
        for g in byzantine:
            X.append(fp.extract(g, global_model))
            y.append(-1.0)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
