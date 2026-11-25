"""Held-out seed replication (Section 7.11): seeds 10-14, FedAvg / MLP / VQAD, sign_flip / min_max / scale.
Usage: python experiments/holdout_seeds.py [workers]. Writes results/raw/holdout/ (resumable)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import base  # noqa: E402
from runner import run_all  # noqa: E402


def suite_holdout():
    jobs = []
    for seed in range(10, 15):
        for atk in ("sign_flip", "min_max", "scale"):
            for agg in ("fedavg", "mlp", "vqad"):
                jobs.append((f"qkd_{agg}_{atk}_s{seed}", base(seed=seed, attack=atk, aggregator=agg, pipeline="qkd")))
    return jobs


if __name__ == "__main__":
    run_all(suite_holdout(), "holdout", int(sys.argv[1]) if len(sys.argv) > 1 else 4)
