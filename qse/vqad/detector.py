"""Variational quantum anomaly detector (VQAD) and classical baselines on the same 4-d fingerprint.

The 4-qubit circuit is simulated exactly as a 16x16 unitary acting on a product input state, in torch
(complex128), so gradients are exact. tests/test_vqad.py checks it against PennyLane and checks autograd
gradients against the parameter-shift rule.

Circuit: Ry(pi x_i) on qubit i; then for each of L layers: Ry(theta) then Rz(phi) on every qubit,
followed by a brick-wall of CNOTs (layers alternate (0,1),(2,3) and (1,2),(3,0)). Output <Z_0> in [-1,1].
Trainable parameters: L * 4 * 2 = 24 for L = 3.
"""

from __future__ import annotations

import numpy as np
import torch

N_Q = 4
DIM = 16


def _cnot(control: int, target: int) -> torch.Tensor:
    m = torch.zeros(DIM, DIM, dtype=torch.complex128)
    for s in range(DIM):
        bits = [(s >> (N_Q - 1 - q)) & 1 for q in range(N_Q)]
        if bits[control]:
            bits[target] ^= 1
        t = sum(b << (N_Q - 1 - q) for q, b in enumerate(bits))
        m[t, s] = 1.0
    return m


_CNOT_EVEN = _cnot(2, 3) @ _cnot(0, 1)
_CNOT_ODD = _cnot(3, 0) @ _cnot(1, 2)
_Z0 = torch.tensor([1.0 if ((s >> (N_Q - 1)) & 1) == 0 else -1.0 for s in range(DIM)], dtype=torch.float64)


def _ry(a):
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex128)


def _rz(b):
    e = torch.exp(-0.5j * b.to(torch.complex128))
    z = torch.zeros((), dtype=torch.complex128)
    return torch.stack([torch.stack([e, z]), torch.stack([z, e.conj()])])


def circuit_unitary(theta: torch.Tensor) -> torch.Tensor:
    """theta shape (L, 4, 2) -> 16x16 unitary of the trainable part."""
    U = torch.eye(DIM, dtype=torch.complex128)
    for l in range(theta.shape[0]):
        ops = [_rz(theta[l, q, 1]) @ _ry(theta[l, q, 0]) for q in range(N_Q)]
        layer = ops[0]
        for o in ops[1:]:
            layer = torch.kron(layer, o)
        ent = _CNOT_EVEN if l % 2 == 0 else _CNOT_ODD
        U = ent @ layer @ U
    return U


def encode(x: torch.Tensor) -> torch.Tensor:
    """x (B,4) in [-1,1] -> product states (B,16) complex via Ry(pi x)."""
    a = np.pi * x.to(torch.float64)
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    psi = torch.stack([c[:, 0], s[:, 0]], 1)
    for q in range(1, N_Q):
        q2 = torch.stack([c[:, q], s[:, q]], 1)
        psi = (psi[:, :, None] * q2[:, None, :]).reshape(len(x), -1)
    return psi.to(torch.complex128)


def vqc_expval(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    psi = encode(x) @ circuit_unitary(theta).T
    return (psi.real ** 2 + psi.imag ** 2) @ _Z0


class _Base:
    name = "base"

    def _forward(self, x):  # pragma: no cover
        raise NotImplementedError

    def _params(self):  # pragma: no cover
        raise NotImplementedError

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self._params()))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.05, restarts: int = 4,
            seed: int = 0) -> float:
        """y in {+1 benign, -1 byzantine}. Balanced MSE on the +-1 target; returns best final loss."""
        X_t = torch.tensor(X, dtype=torch.float64)
        y_t = torch.tensor(y, dtype=torch.float64)
        w = torch.where(y_t > 0, 0.5 / max(1, int((y_t > 0).sum())), 0.5 / max(1, int((y_t < 0).sum())))
        best, best_state = float("inf"), None
        for r in range(restarts):
            self._init(seed * 1000 + r)
            opt = torch.optim.Adam(self._params(), lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                loss = (w * (self._forward(X_t) - y_t) ** 2).sum()
                loss.backward()
                opt.step()
            with torch.no_grad():
                final = float((w * (self._forward(X_t) - y_t) ** 2).sum())
            if final < best:
                best, best_state = final, [p.detach().clone() for p in self._params()]
        with torch.no_grad():
            for p, b in zip(self._params(), best_state):
                p.copy_(b)
        return best

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        return self._forward(torch.tensor(X, dtype=torch.float64)).numpy()

    def flag(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) < 0.0


class VQAD(_Base):
    name = "VQAD"

    def __init__(self, layers: int = 3) -> None:
        self.layers = layers
        self.theta = torch.zeros(layers, N_Q, 2, dtype=torch.float64, requires_grad=True)

    def _init(self, seed):
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.theta.copy_(torch.randn(self.theta.shape, generator=g, dtype=torch.float64) * 0.5)

    def _params(self):
        return [self.theta]

    def _forward(self, x):
        return vqc_expval(x, self.theta)

    def get_weights(self) -> np.ndarray:
        return self.theta.detach().numpy().copy()

    def set_weights(self, w: np.ndarray) -> None:
        with torch.no_grad():
            self.theta.copy_(torch.tensor(w))


class MLPDetector(_Base):
    """Classical 4-h-1 tanh MLP. h=4 gives 25 parameters, matched to the 24-parameter VQC."""
    name = "MLP"

    def __init__(self, hidden: int = 4) -> None:
        self.h = hidden
        self.W1 = torch.zeros(hidden, 4, dtype=torch.float64, requires_grad=True)
        self.b1 = torch.zeros(hidden, dtype=torch.float64, requires_grad=True)
        self.W2 = torch.zeros(1, hidden, dtype=torch.float64, requires_grad=True)
        self.b2 = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    def _init(self, seed):
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self._params():
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64) * 0.5)

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def _forward(self, x):
        return torch.tanh(torch.tanh(x @ self.W1.T + self.b1) @ self.W2.T + self.b2).squeeze(-1)


class LogisticDetector(_Base):
    name = "Linear"

    def __init__(self) -> None:
        self.w = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        self.b = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    def _init(self, seed):
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.w.copy_(torch.randn(4, generator=g, dtype=torch.float64) * 0.5)
            self.b.zero_()

    def _params(self):
        return [self.w, self.b]

    def _forward(self, x):
        return torch.tanh(x @ self.w + self.b)


class FourierMLPDetector(_Base):
    """Classical control for the VQC's data encoding: inputs are mapped to (cos(pi x), sin(pi x)) per feature
    (8 dims, the same trigonometric encoding the circuit applies), then a small tanh MLP. hidden=2 gives
    8*2+2+2+1 = 21 parameters, hidden=3 gives 31."""
    name = "FourierMLP"

    def __init__(self, hidden: int = 3) -> None:
        self.h = hidden
        self.W1 = torch.zeros(hidden, 8, dtype=torch.float64, requires_grad=True)
        self.b1 = torch.zeros(hidden, dtype=torch.float64, requires_grad=True)
        self.W2 = torch.zeros(1, hidden, dtype=torch.float64, requires_grad=True)
        self.b2 = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    def _init(self, seed):
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self._params():
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64) * 0.5)

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def _forward(self, x):
        a = np.pi * x
        phi = torch.cat([torch.cos(a), torch.sin(a)], dim=1)
        return torch.tanh(torch.tanh(phi @ self.W1.T + self.b1) @ self.W2.T + self.b2).squeeze(-1)


class FourierLinearDetector(_Base):
    """Linear model on the trigonometric encoding (9 parameters)."""
    name = "FourierLinear"

    def __init__(self) -> None:
        self.w = torch.zeros(8, dtype=torch.float64, requires_grad=True)
        self.b = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    def _init(self, seed):
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.w.copy_(torch.randn(8, generator=g, dtype=torch.float64) * 0.5)
            self.b.zero_()

    def _params(self):
        return [self.w, self.b]

    def _forward(self, x):
        a = np.pi * x
        phi = torch.cat([torch.cos(a), torch.sin(a)], dim=1)
        return torch.tanh(phi @ self.w + self.b)
