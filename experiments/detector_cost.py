"""Wall-clock cost of scoring one cluster of updates (13 fingerprints), and of computing the fingerprints."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qse.vqad.detector import VQAD, MLPDetector, LogisticDetector  # noqa: E402
from qse.vqad.features import fingerprints  # noqa: E402

torch.set_num_threads(1)
rng = np.random.default_rng(0)
P = 24458
U = torch.tensor(rng.normal(size=(13, P)).astype(np.float32))
res = {}
t0 = time.perf_counter()
for _ in range(200):
    F = fingerprints(U)
res["fingerprint_ms_per_cluster"] = 1000 * (time.perf_counter() - t0) / 200
X = np.clip(rng.normal(0, 0.4, size=(13, 4)), -1, 1)
for name, mk in (("VQAD", VQAD), ("MLP", MLPDetector), ("Linear", LogisticDetector)):
    d = mk()
    d._init(1)
    d.score(X)
    t0 = time.perf_counter()
    for _ in range(500):
        d.score(X)
    res[f"{name}_ms_per_cluster"] = 1000 * (time.perf_counter() - t0) / 500
(ROOT / "results" / "tables" / "detector_cost.json").write_text(json.dumps(res))
print(res)
