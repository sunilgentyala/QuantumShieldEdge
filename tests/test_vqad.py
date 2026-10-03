import numpy as np
import pytest
import torch

from qse.vqad.detector import VQAD, MLPDetector, circuit_unitary, vqc_expval
from qse.vqad.features import fingerprints


def test_unitary_is_unitary():
    th = torch.randn(3, 4, 2, dtype=torch.float64)
    U = circuit_unitary(th)
    assert torch.allclose(U @ U.conj().T, torch.eye(16, dtype=torch.complex128), atol=1e-10)


def test_matches_pennylane():
    qml = pytest.importorskip("pennylane")
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circ(x, th):
        for q in range(4):
            qml.RY(np.pi * x[q], wires=q)
        for l in range(th.shape[0]):
            for q in range(4):
                qml.RY(th[l, q, 0], wires=q)
                qml.RZ(th[l, q, 1], wires=q)
            if l % 2 == 0:
                qml.CNOT(wires=[0, 1]); qml.CNOT(wires=[2, 3])
            else:
                qml.CNOT(wires=[1, 2]); qml.CNOT(wires=[3, 0])
        return qml.expval(qml.PauliZ(0))

    rng = np.random.default_rng(0)
    th = rng.normal(size=(3, 4, 2))
    for _ in range(5):
        x = rng.uniform(-1, 1, 4)
        ref = float(circ(x, th))
        mine = float(vqc_expval(torch.tensor(x[None]), torch.tensor(th)))
        assert abs(ref - mine) < 1e-9


def test_autograd_equals_parameter_shift():
    rng = np.random.default_rng(1)
    th = torch.tensor(rng.normal(size=(3, 4, 2)), requires_grad=True)
    x = torch.tensor(rng.uniform(-1, 1, (1, 4)))
    vqc_expval(x, th).sum().backward()
    g_auto = th.grad.clone().numpy().reshape(-1)
    base = th.detach().numpy().reshape(-1)
    g_ps = np.zeros_like(base)
    for i in range(len(base)):
        up, dn = base.copy(), base.copy()
        up[i] += np.pi / 2
        dn[i] -= np.pi / 2
        f = lambda v: float(vqc_expval(x, torch.tensor(v.reshape(3, 4, 2))))
        g_ps[i] = 0.5 * (f(up) - f(dn))
    assert np.allclose(g_auto, g_ps, atol=1e-9)


def test_parameter_counts():
    assert VQAD(layers=3).n_params() == 24
    assert MLPDetector(hidden=4).n_params() == 25


def test_detectors_learn_separable_data():
    rng = np.random.default_rng(2)
    Xb = rng.normal(0, 0.2, (200, 4)).clip(-1, 1)
    Xz = rng.normal(0, 0.2, (50, 4)).clip(-1, 1) + np.array([0.6, -0.6, 0.0, 0.0])
    X = np.vstack([Xb, Xz.clip(-1, 1)])
    y = np.r_[np.ones(200), -np.ones(50)]
    for det in (VQAD(), MLPDetector()):
        det.fit(X, y, epochs=200, restarts=3)
        f = det.flag(X)
        assert f[y < 0].mean() > 0.8 and f[y > 0].mean() < 0.2


def test_fingerprint_shape_range_and_scale_attack_visible():
    torch.manual_seed(0)
    U = torch.randn(12, 500) * 0.1 + 0.02
    U[0] = U[0] * 10
    F = fingerprints(U)
    assert F.shape == (12, 4) and F.abs().max() <= 1.0
    assert F[0, 0] > F[1:, 0].max()  # magnitude feature separates the boosted update
