"""Leave-one-attack-out detector study repeated over training seeds (detector initialisation and restart
selection are the only sources of randomness; the fingerprint data are fixed)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import ROOT  # noqa: E402
from qse.vqad.detector import VQAD, MLPDetector, LogisticDetector, FourierMLPDetector, FourierLinearDetector  # noqa: E402

if __name__ == "__main__":
    z = np.load(ROOT / "results" / "pretrain_fingerprints.npz")
    X, y, atk = z["X"], z["y"], z["attack"]
    attacks = sorted(set(atk))
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    res = {}
    for seed in range(n_seeds):
        for held in attacks:
            tr, te = np.where(atk != held)[0], np.where(atk == held)[0]
            for mk in (VQAD, MLPDetector, LogisticDetector, FourierMLPDetector, FourierLinearDetector):
                det = mk()
                det.fit(X[tr], y[tr], epochs=300, restarts=4, seed=seed + 1)
                fl = det.flag(X[te]); yt = y[te]
                res.setdefault(f"{det.name}|{held}", []).append(dict(seed=seed, tpr=float(fl[yt < 0].mean()), fpr=float(fl[yt > 0].mean())))
        print("seed", seed, "done", flush=True)
        (ROOT / "results" / "tables" / "loo_seeds_v2.json").write_text(json.dumps(res))
    print("finished")
