"""Additional detector studies on the fixed fingerprint data (leave-one-attack-out, balanced accuracy).

Usage: python experiments/detector_studies.py <study> [n_seeds]
  noise      shot noise and rotation-angle noise applied to a trained 4-qubit circuit at inference
  capacity   circuit depth (1-5 layers) and MLP width (1-8 hidden units)
  features   drop one of the four fingerprint features (set to zero for train and test)
  datasize   fraction of the training fingerprints used
Results go to results/tables/detector_studies_<study>.json (resumable per seed).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import ROOT  # noqa: E402
from qse.vqad.detector import VQAD, MLPDetector, vqc_expval  # noqa: E402

torch.set_num_threads(1)
Z = np.load(ROOT / "results" / "pretrain_fingerprints.npz")
X, Y, ATK = Z["X"], Z["y"], Z["attack"]
ATTACKS = sorted(set(ATK))


def ba(flags, y):
    tpr = float(flags[y < 0].mean())
    fpr = float(flags[y > 0].mean())
    return 0.5 * (tpr + 1 - fpr), tpr, fpr


def fit(mk, Xtr, ytr, seed):
    d = mk()
    d.fit(Xtr, ytr, epochs=300, restarts=4, seed=seed + 1)
    return d


def out_path(study):
    return ROOT / "results" / "tables" / f"detector_studies_{study}.json"


def load(study):
    p = out_path(study)
    return json.loads(p.read_text()) if p.exists() else {}


def save(study, res):
    out_path(study).write_text(json.dumps(res))


def study_noise(n_seeds):
    res = load("noise")
    rng = np.random.default_rng(0)
    shots = [32, 128, 1024, 8192, 0]            # 0 = exact expectation
    sigmas = [0.0, 0.02, 0.05, 0.1, 0.2]
    for seed in range(n_seeds):
        if f"s{seed}" in res:
            continue
        rec = {}
        for held in ATTACKS:
            tr, te = np.where(ATK != held)[0], np.where(ATK == held)[0]
            det = fit(VQAD, X[tr], Y[tr], seed)
            Xt = torch.tensor(X[te], dtype=torch.float64)
            th = det.theta.detach()
            with torch.no_grad():
                z = vqc_expval(Xt, th).numpy()
            for n in shots:
                vals = []
                for _ in range(10):
                    if n == 0:
                        est = z
                    else:
                        k = rng.binomial(n, np.clip((1 + z) / 2, 0, 1))
                        est = 2 * k / n - 1
                    vals.append(ba(est < 0, Y[te])[0])
                rec[f"shots{n}|{held}"] = float(np.mean(vals))
            for s in sigmas:
                vals = []
                for _ in range(10):
                    thn = th + torch.tensor(rng.normal(0, s, th.shape), dtype=torch.float64)
                    with torch.no_grad():
                        zz = vqc_expval(Xt, thn).numpy()
                    vals.append(ba(zz < 0, Y[te])[0])
                rec[f"sigma{s}|{held}"] = float(np.mean(vals))
        res[f"s{seed}"] = rec
        save("noise", res)
        print("noise seed", seed, "done", flush=True)


def study_capacity(n_seeds):
    res = load("capacity")
    for seed in range(n_seeds):
        for held in ATTACKS:
            tr, te = np.where(ATK != held)[0], np.where(ATK == held)[0]
            for L in (1, 2, 3, 4, 5):
                key = f"VQAD-L{L}|{held}|{seed}"
                if key in res:
                    continue
                d = fit(lambda L=L: VQAD(layers=L), X[tr], Y[tr], seed)
                res[key] = dict(zip(("ba", "tpr", "fpr"), ba(d.flag(X[te]), Y[te])), params=d.n_params())
            for h in (1, 2, 4, 8):
                key = f"MLP-h{h}|{held}|{seed}"
                if key in res:
                    continue
                d = fit(lambda h=h: MLPDetector(hidden=h), X[tr], Y[tr], seed)
                res[key] = dict(zip(("ba", "tpr", "fpr"), ba(d.flag(X[te]), Y[te])), params=d.n_params())
        save("capacity", res)
        print("capacity seed", seed, "done", flush=True)


def study_features(n_seeds):
    res = load("features")
    for seed in range(n_seeds):
        for drop in (None, 0, 1, 2, 3):
            Xm = X.copy()
            if drop is not None:
                Xm[:, drop] = 0.0
            for held in ATTACKS:
                tr, te = np.where(ATK != held)[0], np.where(ATK == held)[0]
                for name, mk in (("VQAD", VQAD), ("MLP", MLPDetector)):
                    key = f"{name}|drop{drop}|{held}|{seed}"
                    if key in res:
                        continue
                    d = fit(mk, Xm[tr], Y[tr], seed)
                    res[key] = dict(zip(("ba", "tpr", "fpr"), ba(d.flag(Xm[te]), Y[te])))
        save("features", res)
        print("features seed", seed, "done", flush=True)


def study_datasize(n_seeds):
    res = load("datasize")
    for seed in range(n_seeds):
        rng = np.random.default_rng(100 + seed)
        for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
            for held in ATTACKS:
                tr, te = np.where(ATK != held)[0], np.where(ATK == held)[0]
                sub = rng.choice(tr, size=max(20, int(frac * len(tr))), replace=False)
                for name, mk in (("VQAD", VQAD), ("MLP", MLPDetector)):
                    key = f"{name}|f{frac}|{held}|{seed}"
                    if key in res:
                        continue
                    d = fit(mk, X[sub], Y[sub], seed)
                    res[key] = dict(zip(("ba", "tpr", "fpr"), ba(d.flag(X[te]), Y[te])))
        save("datasize", res)
        print("datasize seed", seed, "done", flush=True)


if __name__ == "__main__":
    study, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
    dict(noise=study_noise, capacity=study_capacity, features=study_features, datasize=study_datasize)[study](n)
    print("finished", study)
