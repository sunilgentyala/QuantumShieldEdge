"""Define and launch all experiment suites. Usage: python experiments/run_experiments.py <suite> [workers]

Suites (all results are written to results/raw/<suite>/ and are resumable):
  main       6 aggregators x 6 attack settings x 3 seeds on the QKD pipeline, + dense-FedAvg reference
  keyrate    key-supply sweep x 4 shortage policies (no attack)
  qber       fixed-QBER stress sweep + transient QBER event (no attack)
  finitekey  finite block-length effect at marginal key supply
  downlink   sparse-OTP / dense-OTP / MAC-only downlink under key shortage
  sparsity   density sensitivity (accuracy vs key demand)
  ablation   flagged-update weight for the VQAD under adaptive attacks
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import ROOT, run_all  # noqa: E402

import json

HP = json.loads((ROOT / "results" / "hparams.json").read_text()) if (ROOT / "results" / "hparams.json").exists() \
    else dict(lr=0.1, local_steps=10, rounds=60)
DET = {k: str(ROOT / "results" / "detectors" / f"{k}.npz") for k in ("vqad", "mlp", "linear")}
ATTACKS = ["none", "sign_flip", "scale", "label_flip", "alie", "min_max"]
AGGS = ["fedavg", "krum", "median", "trimmed_mean", "mlp", "vqad"]


def base(**kw):
    c = dict(lr=HP["lr"], local_steps=HP["local_steps"], rounds=HP["rounds"], eval_every=5)
    c.update(kw)
    if c.get("aggregator") in DET:
        c["detector_path"] = DET[c["aggregator"]]
    return c


def suite_main():
    jobs = []
    for seed in range(3):
        for atk in ATTACKS:
            jobs.append((f"ref_dense_fedavg_{atk}_s{seed}",
                         base(seed=seed, attack=atk, aggregator="fedavg", pipeline="dense")))
            for agg in AGGS:
                jobs.append((f"qkd_{agg}_{atk}_s{seed}",
                             base(seed=seed, attack=atk, aggregator=agg, pipeline="qkd",
                                  record_fingerprints=True)))
    return jobs


def suite_keyrate():
    jobs = []
    for seed in range(2):
        for scale in (0.005, 0.01, 0.02, 0.05, 0.1, 1.0):
            for pol in ("adapt", "wait", "defer", "pq"):
                jobs.append((f"kr_{pol}_x{scale}_s{seed}",
                             base(rounds=30, seed=seed, attack="none", aggregator="fedavg", pipeline="qkd", policy=pol,
                                  link=dict(rate_scale=scale))))
    return jobs


def suite_qber():
    jobs = []
    for seed in range(2):
        for q in (0.03, 0.08, 0.09, 0.095, 0.10, 0.12):
            for pol in ("adapt", "defer", "pq"):
                jobs.append((f"qb_{pol}_q{q}_s{seed}",
                             base(rounds=30, seed=seed, attack="none", aggregator="fedavg", pipeline="qkd", policy=pol,
                                  link=dict(qber_override=q))))
        for pol in ("adapt", "wait", "defer", "pq"):
            jobs.append((f"qev_{pol}_s{seed}",
                         base(rounds=60, seed=seed, attack="none", aggregator="fedavg", pipeline="qkd", policy=pol,
                              qber_events=[(20, 40, 0.12)], eval_every=2)))
        jobs.append((f"qev_nominal_s{seed}",
                     base(rounds=60, seed=seed, attack="none", aggregator="fedavg", pipeline="qkd", policy="adapt",
                          eval_every=2)))
    return jobs


def suite_finitekey():
    jobs = []
    for seed in range(2):
        for blk in (10_000, 100_000, 1_000_000, None):
            for scale in (0.05, 0.1):
                jobs.append((f"fk_{blk}_x{scale}_s{seed}",
                             base(rounds=30, seed=seed, attack="none", aggregator="fedavg", pipeline="qkd", policy="adapt",
                                  link=dict(block_bits=blk, rate_scale=scale))))
    return jobs


def suite_downlink():
    jobs = []
    for seed in range(2):
        for mode in ("sparse_otp", "dense_otp", "mac_only"):
            for scale in (0.05, 0.1, 1.0):
                jobs.append((f"dl_{mode}_x{scale}_s{seed}",
                             base(rounds=30, seed=seed, attack="none", aggregator="fedavg", pipeline="qkd", policy="adapt",
                                  down_mode=mode, link=dict(rate_scale=scale))))
    return jobs


def suite_sparsity():
    jobs = []
    for seed in range(2):
        for d in (0.01, 0.02, 0.05, 0.1, 0.25, 1.0):
            jobs.append((f"sp_{d}_s{seed}",
                         base(rounds=30, seed=seed, attack="none", aggregator="fedavg", pipeline="qkd",
                              density_up=d, density_down=d, capacity_bits=40_000_000)))
    return jobs


def suite_ablation():
    jobs = []
    for seed in range(2):
        for atk in ("alie", "min_max", "sign_flip"):
            for fw in (0.0, 0.1, 0.3):
                jobs.append((f"ab_vqad_fw{fw}_{atk}_s{seed}",
                             base(rounds=30, seed=seed, attack=atk, aggregator="vqad", pipeline="qkd", flagged_weight=fw)))
    return jobs


def suite_qevsmall():
    """Transient QBER outage (rounds 20-40, QBER 12%) with a small pool (about 5 rounds of demand)."""
    jobs = []
    for seed in range(2):
        for pol in ("adapt", "wait", "defer", "pq"):
            jobs.append((f"qevs_{pol}_s{seed}",
                         base(rounds=60, seed=seed, attack="none", aggregator="fedavg", pipeline="qkd", policy=pol,
                              qber_events=[(20, 40, 0.12)], eval_every=2, capacity_bits=400_000)))
        jobs.append((f"qevs_nominal_s{seed}",
                     base(rounds=60, seed=seed, attack="none", aggregator="fedavg", pipeline="qkd", policy="adapt",
                          eval_every=2, capacity_bits=400_000)))
    return jobs


def suite_sens():
    """Sensitivity of the aggregation rules to the corrupted fraction and to label-skew concentration."""
    jobs = []
    settings = [("byz0.1", dict(byz_frac=0.1)), ("byz0.3", dict(byz_frac=0.3)),
                ("alpha0.1", dict(alpha=0.1)), ("alpha1.0", dict(alpha=1.0))]
    for seed in range(3):
        for tag, kw in settings:
            for atk in ("sign_flip", "min_max"):
                for agg in ("fedavg", "median", "vqad"):
                    jobs.append((f"sens_{tag}_{agg}_{atk}_s{seed}",
                                 base(seed=seed, attack=atk, aggregator=agg, pipeline="qkd", rounds=40, **kw)))
    return jobs


SUITES = dict(sens=suite_sens, main=suite_main, keyrate=suite_keyrate, qber=suite_qber, finitekey=suite_finitekey,
              downlink=suite_downlink, qevsmall=suite_qevsmall, sparsity=suite_sparsity, ablation=suite_ablation)

if __name__ == "__main__":
    name = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = SUITES[name]()
    print(f"suite {name}: {len(jobs)} runs, hparams {HP}", flush=True)
    run_all(jobs, name, workers=workers)
