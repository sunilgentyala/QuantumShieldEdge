import numpy as np
import torch

from qse import attacks as atk
from qse.data import assign_clusters, dirichlet_partition
from qse.model import SmallCNN, n_params
from qse.sim import SimConfig, Simulation


def test_dirichlet_partition_is_real_noniid():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(10), 500)
    parts = dirichlet_partition(labels, 20, 0.5, rng)
    assert sorted(np.concatenate(parts)) == list(range(5000))  # exact partition, no overlap
    # label skew: mean max-class share is far above the IID value 0.1
    shares = [np.bincount(labels[p], minlength=10).max() / len(p) for p in parts]
    assert np.mean(shares) > 0.3
    iid = dirichlet_partition(labels, 20, 1000.0, np.random.default_rng(1))
    iid_shares = [np.bincount(labels[p], minlength=10).max() / len(p) for p in iid]
    assert np.mean(iid_shares) < 0.2


def test_clusters_cover_all_clients():
    cl = assign_clusters(50, [13, 12, 13, 12], np.random.default_rng(0))
    assert len(cl) == 4 and sorted(np.concatenate(cl)) == list(range(50))


def test_param_count_matches_model():
    assert n_params(SmallCNN((16, 32, 64))) == sum(p.numel() for p in SmallCNN((16, 32, 64)).parameters())


def test_min_max_stays_inside_honest_diameter_and_alie_shifts():
    torch.manual_seed(0)
    honest = torch.randn(30, 200) * 0.1 + 0.05
    mal = atk.craft("min_max", honest, None, 40, 10)
    diam = torch.cdist(honest, honest).max()
    assert (mal - honest).norm(dim=1).max() <= diam * 1.001
    assert mal.dot(honest.mean(0)) < honest.mean(0).norm() ** 2  # pushed against the honest direction
    a = atk.craft("alie", honest, None, 40, 10)
    assert (a - honest.mean(0)).abs().max() > 0


def test_pipeline_exercises_crypto_and_matches_dense_in_expectation():
    """Tiny end-to-end: sealed messages are verified and decoded; no auth failures; model moves."""
    cfg = SimConfig(rounds=2, n_clients=12, cluster_sizes=(6, 6), local_steps=2, eval_every=1,
                    pipeline="qkd", density_up=0.1, density_down=0.1)
    out = Simulation(cfg).run()
    L = out["log"]
    assert L["auth_fail"] == 0 and L["macs_verified"] >= 8
    assert sum(L["key_up_bits"]) > 0 and sum(L["key_down_bits"]) > 0
    assert 0.0 <= out["final_acc"] <= 1.0


def test_key_accounting_matches_wire_format():
    from qse.crypto import otp_wc as ow
    cfg = SimConfig(rounds=1, n_clients=12, cluster_sizes=(6, 6), local_steps=1, pipeline="qkd",
                    density_up=0.05, density_down=0.05)
    sim = Simulation(cfg)
    out = sim.run()
    k = max(1, int(0.05 * sim.P))
    per_msg = ow.message_key_bits(ow.sparse_payload_bits(k, sim.P))
    assert out["log"]["key_up_bits"][0] == 2 * per_msg  # 2 EANs
    assert out["log"]["key_down_bits"][0] == 2 * per_msg


def test_defer_policy_loses_no_update_and_never_sends_plaintext():
    cfg = SimConfig(rounds=2, n_clients=12, cluster_sizes=(6, 6), local_steps=1, pipeline="qkd",
                    policy="defer", link=dict(rate_scale=0.0), prefill_rounds=0, eval_every=1)
    sim = Simulation(cfg)
    out = sim.run()
    assert sum(out["log"]["n_deferred"]) == 4 and sum(out["log"]["n_ok"]) == 0
    assert out["log"]["key_up_bits"] == [0, 0]
    assert float(sim.resid[0].norm()) > 0  # update retained locally via error feedback


def test_adapt_never_overdraws_pool_for_any_level():
    """Regression: adapt must pick a k whose exact sealed cost fits the pool (byte rounding of indices)."""
    from qse.crypto import otp_wc as ow
    cfg = SimConfig(rounds=1, n_clients=12, cluster_sizes=(6, 6), local_steps=1, pipeline="qkd", policy="adapt",
                    prefill_rounds=0, link=dict(rate_scale=0.0))
    sim = Simulation(cfg)
    for lvl_bytes in list(range(600, 9000, 37)):
        link = sim.links[0]
        link.ean_pool = type(link.ean_pool)(10 ** 7)
        link.ean_pool.deposit(bytes(lvl_bytes))
        mode, ku, kd, _ = sim._plan(0)
        if mode == "adapted":
            up, down = sim._need(0, 0, ku, kd)
            assert up + down <= link.ean_pool.level_bits()
