"""Independent replications of the leave-one-attack-out detector study.

Each replication regenerates the whole fingerprint dataset from scratch: new Dirichlet partitions, new benign and
attacker assignments and new FL trajectories (disjoint seeds), for all five attacks. Detectors are then trained
and tested with the leave-one-attack-out protocol inside each replication. The unit of replication is therefore a
full, independent dataset, not a detector initialisation, so the replications support a valid paired test.

Usage: python experiments/independent_replications.py [n_reps] [workers]
Output: results/tables/independent_replications.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import ROOT, load_suite, run_all  # noqa: E402

ATTACKS = ["sign_flip", "scale", "label_flip", "alie", "min_max"]
ROUNDS = 20


def generate(n_reps: int, workers: int) -> None:
    jobs = []
    for rep in range(n_reps):
        for k, a in enumerate(ATTACKS):
            jobs.append((f"rep{rep}_{a}", dict(seed=20000 + 100 * rep + k, attack=a, aggregator="median",
                                               pipeline="dense", rounds=ROUNDS, eval_every=10,
                                               record_fingerprints=True, lr=0.1, local_steps=8)))
    run_all(jobs, "indep", workers=workers)


def collect(rep: int):
    res = load_suite("indep")
    X, y, tag = [], [], []
    for name, r in res.items():
        if not name.startswith(f"rep{rep}_"):
            continue
        a = name.split("_", 1)[1]
        for fp in r["fingerprints"]:
            x = np.array(fp["X"]); t = np.array(fp["y"])
            X.append(x); y.append(np.where(t, -1.0, 1.0)); tag += [a] * len(x)
    return np.vstack(X), np.concatenate(y), np.array(tag)


def evaluate(n_reps: int) -> dict:
    from qse.vqad.detector import (FourierLinearDetector, FourierMLPDetector, LogisticDetector, MLPDetector, VQAD)
    out = {}
    for rep in range(n_reps):
        X, y, atk = collect(rep)
        for held in ATTACKS:
            tr, te = np.where(atk != held)[0], np.where(atk == held)[0]
            for mk in (VQAD, MLPDetector, LogisticDetector, FourierMLPDetector, FourierLinearDetector):
                det = mk()
                det.fit(X[tr], y[tr], epochs=300, restarts=4, seed=rep + 1)
                fl = det.flag(X[te]); yt = y[te]
                tpr = float(fl[yt < 0].mean()) if (yt < 0).any() else float("nan")
                fpr = float(fl[yt > 0].mean())
                out.setdefault(f"{det.name}|{held}", []).append(dict(rep=rep, tpr=tpr, fpr=fpr))
        print("rep", rep, "evaluated", flush=True)
    return out


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    generate(n_reps, workers)
    res = evaluate(n_reps)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "tables" / "independent_replications.json").write_text(json.dumps(res))
    print("finished")
