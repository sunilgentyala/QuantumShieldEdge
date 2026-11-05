"""Summaries for the studies added in response to review: cluster-local ALIE, long-horizon runs, and independent
replications of the leave-one-attack-out detector study. Writes results/tables/revision_studies.json and prints it.

The replication unit for the detector comparison is a whole regenerated fingerprint dataset, so the paired tests
below are over independent replications (n = number of replications), not over initialisation seeds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import ROOT, load_suite  # noqa: E402

TAB = ROOT / "results" / "tables"


def mean_ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def det_rates(r):
    L = r["log"]
    tp, fp, fn, tn = (sum(L[k]) for k in ("tp", "fp", "fn", "tn"))
    return (tp / (tp + fn) if tp + fn else float("nan")), (fp / (fp + tn) if fp + tn else float("nan"))


def alielocal():
    loc = load_suite("alielocal")
    main = load_suite("main")
    out = {}
    for agg in ("fedavg", "median", "vqad"):
        a_loc = [loc[f"alielocal_{agg}_s{s}"]["final_acc"] * 100 for s in range(3) if f"alielocal_{agg}_s{s}" in loc]
        a_pool = [main[f"qkd_{agg}_alie_s{s}"]["final_acc"] * 100 for s in range(3)]
        row = dict(local=mean_ci(a_loc), pooled=mean_ci(a_pool), n_local=len(a_loc))
        if agg == "vqad":
            row["local_tpr_fpr"] = np.nanmean([det_rates(loc[f"alielocal_{agg}_s{s}"]) for s in range(3)
                                               if f"alielocal_{agg}_s{s}" in loc], axis=0).tolist()
            row["pooled_tpr_fpr"] = np.nanmean([det_rates(main[f"qkd_{agg}_alie_s{s}"]) for s in range(3)],
                                               axis=0).tolist()
        out[agg] = row
    return out


def long_runs():
    res = load_suite("long")
    out = {}
    for name, r in res.items():
        _, agg, atk, seed = name.split("_", 3)
        out.setdefault(f"{agg}|{atk}", []).append(r["final_acc"] * 100)
    return {k: dict(zip(("mean", "ci95"), mean_ci(v)), n=len(v), values=v) for k, v in sorted(out.items())}


def independent():
    f = TAB / "independent_replications.json"
    if not f.exists():
        return None
    raw = json.loads(f.read_text())
    dets = sorted({k.split("|")[0] for k in raw})
    attacks = sorted({k.split("|")[1] for k in raw})
    n_reps = max(e["rep"] for v in raw.values() for e in v) + 1
    ba = {d: np.zeros(n_reps) for d in dets}
    per_attack = {d: {} for d in dets}
    for d in dets:
        for rep in range(n_reps):
            vals = []
            for a in attacks:
                e = next(x for x in raw[f"{d}|{a}"] if x["rep"] == rep)
                b = 0.5 * (e["tpr"] + 1 - e["fpr"])
                if not np.isnan(b):
                    vals.append(b)
                per_attack[d].setdefault(a, []).append(b)
            ba[d][rep] = np.mean(vals)
    out = {"n_replications": n_reps, "balanced_accuracy": {}, "paired_vs_VQAD": {}, "per_attack": {}}
    for d in dets:
        m, c = mean_ci(ba[d])
        out["balanced_accuracy"][d] = dict(mean=m, ci95=c, per_rep=ba[d].tolist())
        out["per_attack"][d] = {a: float(np.nanmean(v)) for a, v in per_attack[d].items()}
    for d in dets:
        if d == "VQAD":
            continue
        diff = ba["VQAD"] - ba[d]
        t = stats.ttest_rel(ba["VQAD"], ba[d]) if n_reps > 1 else None
        try:
            w = stats.wilcoxon(ba["VQAD"], ba[d], method="exact").pvalue if n_reps > 1 else None
        except ValueError:
            w = None
        out["paired_vs_VQAD"][d] = dict(mean_diff=float(diff.mean()), diffs=diff.tolist(),
                                        t_p=None if t is None else float(t.pvalue),
                                        wilcoxon_exact_p=None if w is None else float(w),
                                        vqad_wins=int((diff > 0).sum()), n=n_reps)
    return out


if __name__ == "__main__":
    out = dict(alie_local=alielocal(), long=long_runs(), independent=independent())
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "revision_studies.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
