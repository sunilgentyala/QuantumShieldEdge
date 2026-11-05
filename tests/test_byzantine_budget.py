import numpy as np
import torch

from qse import attacks as atk
from qse.sim import SimConfig, Simulation, byz_budget


def test_budget_matches_documented_counts():
    assert [byz_budget(0.2, n) for n in (13, 12, 13, 12)] == [3, 2, 3, 2]
    assert [byz_budget(0.1, n) for n in (13, 12)] == [1, 1]
    assert [byz_budget(0.3, n) for n in (13, 12)] == [4, 4]


def test_attackers_and_krum_f_use_the_same_budget():
    sim = Simulation(SimConfig(rounds=0, attack="sign_flip", aggregator="krum"))
    got = [int(sim.byz[c].sum()) for c in sim.clusters]
    assert [len(c) for c in sim.clusters] == [13, 12, 13, 12]
    assert got == sim.budget == [3, 2, 3, 2]
    assert int(sim.byz.sum()) == 10


def test_local_alie_round_runs_and_sets_one_vector_per_cluster():
    cfg = SimConfig(rounds=1, n_clients=12, cluster_sizes=(6, 6), local_steps=1, attack="alie_local",
                    aggregator="fedavg", pipeline="dense", byz_frac=0.34)
    sim = Simulation(cfg)
    assert sim.budget == [2, 2]
    out = sim.run()
    assert 0.0 <= out["final_acc"] <= 1.0


def test_no_attack_has_no_byzantine_leaves_but_keeps_the_budget():
    sim = Simulation(SimConfig(rounds=0, attack="none"))
    assert int(sim.byz.sum()) == 0 and sum(sim.budget) == 10


def test_alie_z_depends_on_n_and_f():
    z_pooled = atk.alie_z(50, 10)
    assert abs(z_pooled - 0.47) < 0.01
    # cluster-local parameters give different shifts, which is why the local variant is a separate attack
    assert atk.alie_z(12, 2) != z_pooled and atk.alie_z(13, 3) != z_pooled


def test_local_alie_uses_cluster_statistics():
    torch.manual_seed(0)
    honest = torch.randn(10, 50) * 0.1
    mal = atk.craft("alie", honest, None, 13, 3)
    z = atk.alie_z(13, 3)
    assert torch.allclose(mal, honest.mean(0) - z * honest.std(0, unbiased=False), atol=1e-6)
