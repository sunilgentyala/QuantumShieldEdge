"""Collect labelled fingerprints from dedicated pre-training runs (seeds 9000+, disjoint from evaluation seeds
0-4, hence different data partitions and different benign/attacker assignments) and train the VQAD and the two
classical baselines. Pre-training runs use a median aggregator so the global model keeps learning under attack.

Output: results/detectors/{vqad,mlp,linear}.npz and results/pretrain_fingerprints.npz
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


def main():
    jobs = []
    for k, a in enumerate(ATTACKS):
        jobs.append((f"pre_{a}", dict(seed=9000 + k, attack=a, aggregator="median", pipeline="dense",
                                      rounds=ROUNDS, eval_every=10, record_fingerprints=True, lr=0.1, local_steps=8)))
    run_all(jobs, "pretrain", workers=5)
    res = load_suite("pretrain")
    X, y, tag, rnd = [], [], [], []
    for name, r in res.items():
        a = name[4:]
        for fp in r["fingerprints"]:
            x = np.array(fp["X"]); t = np.array(fp["y"])
            X.append(x); y.append(np.where(t, -1.0, 1.0)); tag += [a] * len(x); rnd += [fp["round"]] * len(x)
    X, y = np.vstack(X), np.concatenate(y)
    np.savez(ROOT / "results" / "pretrain_fingerprints.npz", X=X, y=y, attack=np.array(tag), round=np.array(rnd))
    print("samples", len(X), "byzantine", int((y < 0).sum()))

    from qse.vqad.detector import VQAD, MLPDetector, LogisticDetector
    out = ROOT / "results" / "detectors"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for det in (VQAD(), MLPDetector(), LogisticDetector()):
        loss = det.fit(X, y, epochs=300, restarts=4)
        fl = det.flag(X)
        summary[det.name] = dict(params=det.n_params(), train_loss=loss,
                                 train_tpr=float(fl[y < 0].mean()), train_fpr=float(fl[y > 0].mean()))
        np.savez(out / f"{det.name.lower() if det.name != 'Linear' else 'linear'}.npz",
                 **{f"p{i}": p.detach().numpy() for i, p in enumerate(det._params())})
        print(det.name, summary[det.name], flush=True)
    (ROOT / "results" / "detector_training_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
