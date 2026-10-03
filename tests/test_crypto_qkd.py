import numpy as np
import pytest

from qse.crypto import otp_wc as ow
from qse.qkd.bb84 import run_bb84
from qse.qkd.finite_key import asymptotic_fraction, asymptotic_threshold, finite_fraction
from qse.qkd.keypool import KeyPool, LinkConfig, QKDLink


def test_sparse_roundtrip_and_size():
    rng = np.random.default_rng(1)
    P = 24458
    k = 1200
    idx = np.sort(rng.choice(P, k, replace=False))
    val = rng.normal(size=k).astype(np.float16)
    pl = ow.pack_sparse(7, idx, val, P)
    assert len(pl) * 8 == ow.sparse_payload_bits(k, P)
    r, i2, v2 = ow.unpack_sparse(pl, P)
    assert r == 7 and np.array_equal(i2, idx) and np.array_equal(v2, val.astype(np.float32))


def test_seal_open_and_tamper_detection():
    rng = np.random.default_rng(2)
    pl = rng.bytes(5000)
    kp, tp = rng.bytes(5000), rng.bytes(8)
    mk = int(rng.integers(1, ow.P61))
    msg = ow.seal(pl, kp, tp, mk)
    assert ow.open_sealed(msg, kp, tp, mk) == pl
    assert msg[:-8] != pl  # ciphertext differs from plaintext
    for pos in (0, 100, 4999, len(msg) - 1):
        bad = bytearray(msg)
        bad[pos] ^= 0x01
        with pytest.raises(ow.AuthError):
            ow.open_sealed(bytes(bad), kp, tp, mk)


def test_wrong_pad_yields_garbage_and_wrong_mac_key_is_rejected():
    rng = np.random.default_rng(3)
    pl = rng.bytes(2000)
    kp, tp = rng.bytes(2000), b"\x01" * 8
    msg = ow.seal(pl, kp, tp, 12345)
    # MAC is over the ciphertext, so a wrong *pad* opens (tag ok) but yields unrelated plaintext
    assert ow.open_sealed(msg, rng.bytes(2000), tp, 12345) != pl
    # a wrong MAC key or tag pad is rejected
    with pytest.raises(ow.AuthError):
        ow.open_sealed(msg, kp, tp, 54321)
    with pytest.raises(ow.AuthError):
        ow.open_sealed(msg, kp, b"\x02" * 8, 12345)


def test_keypool_atomic_and_capacity():
    p = KeyPool(capacity_bits=800)
    assert p.deposit(bytes(range(100))) == 800
    assert p.deposit(b"x") == 0  # full
    assert p.take(101) is None and p.level_bits() == 800  # never partial
    assert p.take(40) == bytes(range(40))
    assert p.take(60) == bytes(range(40, 100))
    assert p.level_bits() == 0


def test_link_pools_stay_synchronised():
    link = QKDLink(LinkConfig(), capacity_bits=2_000_000, seed=5)
    for _ in range(5):
        link.advance(60.0)
    a = link.ean_pool.take(1000)
    b = link.cloud_pool.take(1000)
    assert a == b and a is not None


def test_thresholds():
    assert abs(asymptotic_threshold(1.0) - 0.110) < 1e-3
    t = asymptotic_threshold(1.16)
    assert 0.09 < t < 0.10
    assert asymptotic_fraction(t + 0.002) == 0.0
    assert asymptotic_fraction(0.01) > asymptotic_fraction(0.05) > asymptotic_fraction(0.09)


def test_finite_key_monotone_in_block_length():
    for q in (0.01, 0.03, 0.06):
        vals = [float(finite_fraction(q, n)) for n in (1e4, 1e5, 1e6, 1e7)]
        assert vals == sorted(vals)
        assert vals[-1] <= float(asymptotic_fraction(q)) + 1e-9


def test_bb84_detects_intercept_resend():
    rng = np.random.default_rng(4)
    clean = run_bb84(100_000, 0.02, rng)
    assert 0.01 < clean.qber_est < 0.035 and not clean.aborted
    full = run_bb84(100_000, 0.02, rng, eve_fraction=1.0)
    assert 0.22 < full.qber_est < 0.30 and full.aborted  # theoretical 25% + channel noise
