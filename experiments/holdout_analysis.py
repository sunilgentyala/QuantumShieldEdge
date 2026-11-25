"""Table 13 of the manuscript: held-out seeds 10-14 plus main-matrix seeds 0-2, paired against FedAvg."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(__file__).resolve().parent.parent / "results" / "raw"
AT = ("sign_flip", "min_max", "scale")


def load(suite, a, k, seeds):
    return np.array([json.loads((R / suite / f"qkd_{a}_{k}_s{s}.json").read_text())["final_acc"] * 100 for s in seeds])


def pooled(a, k):
    return np.concatenate([load("main", a, k, range(3)), load("holdout", a, k, range(10, 15))])


for k in AT:
    h = {a: load("holdout", a, k, range(10, 15)) for a in ("fedavg", "mlp", "vqad")}
    print(k, {a: f"{v.mean():.1f} +- {stats.t.ppf(.975, 4) * v.std(ddof=1) / 5 ** .5:.1f}" for a, v in h.items()})
    for a in ("mlp", "vqad"):
        d = pooled(a, k) - pooled("fedavg", k)
        ci = stats.t.ppf(.975, 7) * d.std(ddof=1) / 8 ** .5
        w = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
        print(f"  {a}-fedavg n=8: {d.mean():+.2f} +- {ci:.2f}, above FedAvg {(d > 0).sum()}/8, "
              f"t-p={stats.ttest_1samp(d, 0).pvalue:.4f}, wilcoxon-p={w:.4f}")
    d = pooled("vqad", k) - pooled("mlp", k)
    print(f"  vqad-mlp n=8: {d.mean():+.2f}, p={stats.ttest_1samp(d, 0).pvalue:.2f}")
