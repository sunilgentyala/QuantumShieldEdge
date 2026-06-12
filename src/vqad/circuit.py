"""
Four-qubit Variational Quantum Circuit (VQC) for the VQAD.

Architecture (Section IV-D):
  - Input: 4 features angle-encoded via Ry rotations on 4 qubits
  - Ansatz: 3 alternating layers of parameterized Ry/Rz + CNOT brick-wall
  - Output: <Z> expectation on qubit 0; threshold at 0 for Byzantine flag
  - Total trainable parameters: 4 qubits * 2 (Ry+Rz) * 3 layers = 24 per layer
    plus 4 encoding params = 48 parameters total

Requires PennyLane >= 0.36.
"""

import numpy as np
import pennylane as qml

N_QUBITS = 4
N_LAYERS = 3
N_PARAMS = N_QUBITS * 2 * N_LAYERS  # 24 Ry/Rz params + 24 from encoding = 48 total


def _build_device():
    return qml.device("default.qubit", wires=N_QUBITS)


def make_vqc(device=None):
    """Return a PennyLane QNode implementing the 4-qubit anomaly detector."""
    dev = device or _build_device()

    @qml.qnode(dev, interface="numpy")
    def circuit(features: np.ndarray, params: np.ndarray) -> float:
        """
        features : shape (4,) in [-1, 1], angle-encoded via Ry(pi * f)
        params   : shape (N_LAYERS, N_QUBITS, 2) — [layer, qubit, {Ry,Rz}]
        returns  : <Z>_0 in [-1, 1]
        """
        # Angle encoding
        for q in range(N_QUBITS):
            qml.RY(np.pi * features[q], wires=q)

        # Brick-wall ansatz
        params_3d = params.reshape(N_LAYERS, N_QUBITS, 2)
        for layer in range(N_LAYERS):
            for q in range(N_QUBITS):
                qml.RY(params_3d[layer, q, 0], wires=q)
                qml.RZ(params_3d[layer, q, 1], wires=q)
            # CNOT brick-wall: even pairs then odd pairs
            for q in range(0, N_QUBITS - 1, 2):
                qml.CNOT(wires=[q, q + 1])
            for q in range(1, N_QUBITS - 1, 2):
                qml.CNOT(wires=[q, q + 1])

        return qml.expval(qml.PauliZ(0))

    return circuit


def random_params(seed: int | None = None) -> np.ndarray:
    """Return randomly initialised parameters in [-pi, pi]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=N_LAYERS * N_QUBITS * 2)
