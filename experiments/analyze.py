"""Aggregate raw runs into tables (results/tables/*.json|csv) and figures (results/figures/*.png).
Usage: python experiments/analyze.py [part ...]   parts: main detect keyrate qber finitekey downlink sparsity ablation keys loo
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import ROOT, load_suite  # noqa: E402

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300})
AGG_LABEL = {"fedavg": "FedAvg", "krum": "Krum", "multi_krum": "Multi-Krum", "median": "Coord. median",
             "trimmed_mean": "Trimmed mean", "linear": "Linear det.", "mlp": "MLP det.", "vqad": "VQAD (VQC)"}
ATK_LABEL = {"none": "No attack", "sign_flip": "Sign flip", "scale": "Scale x10", "label_flip": "Label flip",
             "alie": "ALIE", "min_max": "Min-max"}
AGG_COLOR = {"fedavg": "#6c757d", "krum": "#8d99ae", "multi_krum": "#457b9d", "median": "#2a9d8f",
             "trimmed_mean": "#e9c46a", "linear": "#f4a261", "mlp": "#e76f51", "vqad": "#7b2cbf"}


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name)
    plt.close()
    print("figure", name)


def parse(name):
    return {k: v for k, v in re.findall(r"_([a-z]+)([0-9.]+|None)(?=_|$)", name)}


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


# --------------------------------------------------------------------------------------------- main matrix
def part_main():
    runs = load_suite("main")
    acc = defaultdict(list)
    curves = defaultdict(list)
    for n, r in runs.items():
        m = re.match(r"(qkd|ref_dense)_(\w+?)_(none|sign_flip|scale|label_flip|alie|min_max)_s(\d+)", n)
        if not m:
            continue
        pipe, agg, atk, seed = m.groups()
        key = ("DENSE" if pipe == "ref_dense" else agg, atk)
        acc[key].append((int(seed), r["final_acc"]))
        curves[key].append(r["log"]["acc"])
        if "acc_round" not in globals().get("_rr", {}):
            globals()["_rr"] = {"acc_round": r["log"]["acc_round"]}
    rounds = globals()["_rr"]["acc_round"]
    aggs = ["DENSE", "fedavg", "krum", "multi_krum", "median", "trimmed_mean", "linear", "mlp", "vqad"]
    atks = list(ATK_LABEL)
    table = {}
    for a in aggs:
        for k in atks:
            v = [x for _, x in sorted(acc.get((a, k), []))]
            if v:
                table[f"{a}|{k}"] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                         ci95=ci95(v), n=len(v), values=v)
    # paired tests: VQAD vs MLP and VQAD vs best classical robust aggregator, per attack
    tests = {}
    for k in atks:
        v = {a: np.array([x for _, x in sorted(acc.get((a, k), []))]) for a in aggs}
        if len(v["vqad"]) < 2:
            continue
        rob = {a: v[a].mean() for a in ("krum", "multi_krum", "median", "trimmed_mean") if len(v[a])}
        best = max(rob, key=rob.get) if rob else None
        row = {}
        for other in ("mlp", "linear", "fedavg", best):
            if other and len(v[other]) == len(v["vqad"]):
                t, p = stats.ttest_rel(v["vqad"], v[other])
                row[other] = dict(diff=float((v["vqad"] - v[other]).mean()), p=float(p))
        tests[k] = dict(best_robust=best, vs=row)
    (TAB / "main_accuracy.json").write_text(json.dumps(dict(table=table, tests=tests), indent=1))
    # CSV
    with open(TAB / "main_accuracy.csv", "w") as f:
        f.write("aggregator," + ",".join(ATK_LABEL[k] for k in atks) + "\n")
        for a in aggs:
            f.write(AGG_LABEL.get(a, "Dense FedAvg (ideal channel)") + "," + ",".join(
                f"{100 * table[f'{a}|{k}']['mean']:.1f}+-{100 * table[f'{a}|{k}']['ci95']:.1f}"
                if f"{a}|{k}" in table else "" for k in atks) + "\n")
    # curves figure: 2x3 panels
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.4), sharex=True, sharey=True)
    for ax, k in zip(axs.flat, atks):
        for a in ["DENSE", "fedavg", "krum", "median", "trimmed_mean", "mlp", "vqad"]:
            c = curves.get((a, k))
            if not c:
                continue
            c = np.array(c)
            col = "k" if a == "DENSE" else AGG_COLOR[a]
            ls = "--" if a == "DENSE" else "-"
            ax.plot(rounds, 100 * c.mean(0), color=col, ls=ls, lw=1.2,
                    label="Dense FedAvg, ideal" if a == "DENSE" else AGG_LABEL[a])
            if len(c) > 1:
                ax.fill_between(rounds, 100 * (c.mean(0) - c.std(0)), 100 * (c.mean(0) + c.std(0)), color=col, alpha=0.12)
        ax.set_title(ATK_LABEL[k], fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("FL round")
    for ax in axs[:, 0]:
        ax.set_ylabel("Test accuracy (%)")
    axs[0, 0].legend(fontsize=5.5, loc="lower right")
    savefig("fig2_accuracy_curves.png")

    # detection per round (aggregated over seeds)
    det = {}
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.0), sharex=True, sharey=True)
    for ax, k in zip(axs.flat, atks[1:] + ["none"]):
        for a in ("mlp", "vqad"):
            tps, fps, fns, tns = [], [], [], []
            for n, r in runs.items():
                m = re.match(rf"qkd_{a}_{k}_s(\d+)", n)
                if m:
                    L = r["log"]
                    tps.append(L["tp"]); fps.append(L["fp"]); fns.append(L["fn"]); tns.append(L["tn"])
            if not tps:
                continue
            TP, FP, FN, TN = (np.array(x).sum(0) for x in (tps, fps, fns, tns))
            tpr = TP / np.maximum(1, TP + FN)
            fpr = FP / np.maximum(1, FP + TN)
            det[f"{a}|{k}"] = dict(tpr=float(TP.sum() / max(1, TP.sum() + FN.sum())),
                                   fpr=float(FP.sum() / max(1, FP.sum() + TN.sum())),
                                   fpr_after20=float(FP[20:].sum() / max(1, FP[20:].sum() + TN[20:].sum())),
                                   fpr_first20=float(FP[:20].sum() / max(1, FP[:20].sum() + TN[:20].sum())),
                                   tpr_curve=tpr.tolist(), fpr_curve=fpr.tolist())
            if k != "none":
                ax.plot(np.arange(1, len(tpr) + 1), 100 * np.convolve(tpr, np.ones(5) / 5, "same"),
                        color=AGG_COLOR[a], lw=1.1, label=AGG_LABEL[a])
            else:
                ax.plot(np.arange(1, len(fpr) + 1), 100 * np.convolve(fpr, np.ones(5) / 5, "same"),
                        color=AGG_COLOR[a], lw=1.1, label=AGG_LABEL[a])
        ax.set_title(ATK_LABEL[k] + (" (FPR)" if k == "none" else " (TPR)"), fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("FL round")
    for ax in axs[:, 0]:
        ax.set_ylabel("Rate (%), 5-round mean")
    axs[0, 0].legend(fontsize=5.5)
    savefig("fig3_detection_rates.png")
    (TAB / "detection_rates.json").write_text(json.dumps({k: {x: v[x] for x in v if "curve" not in x}
                                                           for k, v in det.items()}, indent=1))
    (TAB / "detection_curves.json").write_text(json.dumps(det))


# --------------------------------------------------------------------------------------------- key coupling
def summarize_key_run(r):
    L = r["log"]
    n = len(L["n_ok"])
    ok = sum(L["n_ok"]) + sum(L["n_adapted"])
    return dict(final_acc=r["final_acc"],
                it_frac=ok / (4 * n),
                defer_frac=sum(L["n_deferred"]) / (4 * n),
                pq_frac=sum(L["n_pq"]) / (4 * n),
                adapt_frac=sum(L["n_adapted"]) / (4 * n),
                key_bits=(sum(L["key_up_bits"]) + sum(L["key_down_bits"])) / n / 4,
                delay=float(np.mean(L["delay_s"])),
                dens=float(np.mean([d for d in L["density_up_eff"] if d > 0] or [0])),
                macs=L["macs_verified"], auth_fail=L["auth_fail"],
                min_level=float(np.min(L["pool_level"])) if L["pool_level"] else None)


def part_keyrate():
    runs = load_suite("keyrate")
    agg = defaultdict(list)
    for n, r in runs.items():
        m = re.match(r"kr_(\w+)_x([0-9.]+)_s(\d+)", n)
        agg[(m.group(1), float(m.group(2)))].append(summarize_key_run(r))
    out = {f"{p}|{s}": {k: float(np.mean([x[k] for x in v])) for k in v[0] if v[0][k] is not None}
           for (p, s), v in agg.items()}
    (TAB / "keyrate.json").write_text(json.dumps(out, indent=1))
    demand = 76096  # bits per EAN link per round (sim CNN, 5% up/down); supply = scale * 35.0 kbps nominal
    from qse.qkd.keypool import LinkConfig, QKDLink
    probe = QKDLink(LinkConfig(), 10 ** 9, seed=1)
    mean_rate = float(np.mean([probe.sample_state()[1] for _ in range(2000)]))
    ratio = lambda s: s * mean_rate * 60.0 / demand
    scales = sorted({s for _, s in agg})
    pols = ["adapt", "wait", "defer", "pq"]
    cols = {"adapt": "#7b2cbf", "wait": "#2a9d8f", "defer": "#e76f51", "pq": "#6c757d"}
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for p in pols:
        xs = [ratio(s) for s in scales]
        axs[0].plot(xs, [100 * out[f"{p}|{s}"]["final_acc"] for s in scales], "o-", color=cols[p], ms=3, label=p)
        axs[1].plot(xs, [100 * out[f"{p}|{s}"]["it_frac"] for s in scales], "o-", color=cols[p], ms=3, label=p)
    axs[2].plot([ratio(s) for s in scales], [out[f"wait|{s}"]["delay"] for s in scales], "o-", color=cols["wait"], ms=3)
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel("Key supply / demand per round")
        ax.axvline(1.0, color="k", lw=0.6, ls=":")
    axs[0].set_ylabel("Final accuracy (%)"); axs[1].set_ylabel("Exchanges under OTP (%)")
    axs[2].set_ylabel("Mean extra delay per round (s)"); axs[0].legend(fontsize=6)
    savefig("fig5_keyrate_policies.png")
    return out, ratio


def part_qber():
    runs = load_suite("qber")
    fixed = defaultdict(list)
    ev = defaultdict(list)
    ev_curves = defaultdict(list)
    ev_large = defaultdict(list)
    for n, r in runs.items():
        m = re.match(r"qb_(\w+)_q([0-9.]+)_s(\d+)", n)
        if m:
            fixed[(m.group(1), float(m.group(2)))].append(summarize_key_run(r))
        m = re.match(r"qev_(\w+)_s(\d+)", n)
        if m:
            ev_large[m.group(1)].append(summarize_key_run(r))
        m = re.match(r"qevs_(\w+)_s(\d+)", n)
        if m:
            ev[m.group(1)].append(summarize_key_run(r))
            ev_curves[m.group(1)].append((np.array(r["log"]["pool_level"]).mean(1), r["log"]["acc"], r["log"]["acc_round"],
                                          np.array(r["log"]["qber"]).mean(1)))
    out = dict(fixed={f"{p}|{q}": {k: float(np.mean([x[k] for x in v])) for k in v[0] if v[0][k] is not None}
                      for (p, q), v in fixed.items()},
               event={p: {k: float(np.mean([x[k] for x in v])) for k in v[0] if v[0][k] is not None}
                      for p, v in ev.items()},
               event_large_buffer={p: {k: float(np.mean([x[k] for x in v])) for k in v[0] if v[0][k] is not None}
                                   for p, v in ev_large.items()})
    (TAB / "qber.json").write_text(json.dumps(out, indent=1))
    cols = {"adapt": "#7b2cbf", "wait": "#2a9d8f", "defer": "#e76f51", "pq": "#6c757d", "nominal": "k"}
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for p, c in ev_curves.items():
        lvl = np.mean([x[0] for x in c], 0)
        axs[0].plot(np.arange(1, len(lvl) + 1), 100 * lvl, color=cols[p], lw=1.2, label=p)
        accs = np.mean([x[1] for x in c], 0)
        axs[1].plot(c[0][2], 100 * accs, color=cols[p], lw=1.2, label=p)
    axs[0].axvspan(20, 40, color="red", alpha=0.08)
    axs[1].axvspan(20, 40, color="red", alpha=0.08)
    axs[0].set_xlabel("FL round"); axs[0].set_ylabel("Key buffer occupancy (%)")
    axs[1].set_xlabel("FL round"); axs[1].set_ylabel("Test accuracy (%)")
    axs[0].legend(fontsize=6)
    qs = sorted({q for _, q in fixed})
    for p in ("adapt", "defer", "pq"):
        axs[2].plot(qs, [100 * out["fixed"][f"{p}|{q}"]["it_frac"] for q in qs], "o-", color=cols[p], ms=3, label=p)
    axs[2].axvline(0.0981, color="k", lw=0.6, ls=":")
    axs[2].set_xlabel("Fixed QBER"); axs[2].set_ylabel("Exchanges under OTP (%)")
    savefig("fig6_qber_stress.png")
    return out


def part_simple(suite, pat, keyname):
    runs = load_suite(suite)
    agg = defaultdict(list)
    for n, r in runs.items():
        m = re.match(pat, n)
        agg[m.groups()[:-1]].append(summarize_key_run(r))
    out = {"|".join(map(str, k)): {kk: float(np.mean([x[kk] for x in v])) for kk in v[0] if v[0][kk] is not None}
           for k, v in agg.items()}
    (TAB / f"{keyname}.json").write_text(json.dumps(out, indent=1))
    return out


def part_sparsity():
    out = part_simple("sparsity", r"sp_([0-9.]+)_s(\d+)", "sparsity")
    ds = sorted(out, key=float)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot([out[d]["key_bits"] / 60 / 1000 for d in ds], [100 * out[d]["final_acc"] for d in ds], "o-", color="#7b2cbf")
    for d in ds:
        ax.annotate(f"{100 * float(d):g}%", (out[d]["key_bits"] / 60 / 1000, 100 * out[d]["final_acc"]),
                    textcoords="offset points", xytext=(3, 3), fontsize=6)
    ax.set_xscale("log"); ax.set_xlabel("Key rate demanded per link (kbps)"); ax.set_ylabel("Final accuracy (%)")
    savefig("fig8_sparsity.png")


# --------------------------------------------------------------------------------------------- analytic
def part_keys():
    d = json.loads((ROOT / "results" / "key_analysis.json").read_text())
    # Fig 4: required rate vs model size
    A = d["A_required_kbps_per_link"]
    names = list(A)
    P = [A[n]["P"] for n in names]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for dd, col in zip(("0.01", "0.05", "0.25"), ("#2a9d8f", "#7b2cbf", "#e76f51")):
        axs[0].plot(P, [A[n]["sparse_otp"][dd] for n in names], "o-", color=col, ms=3, label=f"K/M={float(dd):.0%}")
    axs[0].plot(P, [A[n]["dense_otp"]["0.05"] for n in names], "s--", color="k", ms=3, label="dense downlink")
    axs[0].axhspan(10, 100, color="grey", alpha=0.15)
    axs[0].text(P[0], 55, "metro QKD\n10-100 kbps", fontsize=6, va="center")
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("Model parameters"); axs[0].set_ylabel("Required key rate per link (kbps)")
    axs[0].legend(fontsize=5.5)
    B = d["B_secret_fraction"]
    q = np.array(B["qber"])
    axs[1].plot(100 * q, B["asymptotic"], "k-", lw=1.4, label="asymptotic")
    for n, col in zip((10_000, 100_000, 1_000_000, 10_000_000), ("#e76f51", "#e9c46a", "#2a9d8f", "#457b9d")):
        axs[1].plot(100 * q, B[f"finite_{n}"], color=col, lw=1.1, label=f"n=1e{int(np.log10(n))}")
    axs[1].axvline(100 * B["threshold_f1.16"], color="k", ls=":", lw=0.6)
    axs[1].set_xlabel("QBER (%)"); axs[1].set_ylabel("Secret-key fraction"); axs[1].legend(fontsize=5.5)
    C = d["C_eavesdropper"]
    axs[2].errorbar([c["eve_fraction"] for c in C], [100 * c["qber_mean"] for c in C], [100 * c["qber_std"] for c in C],
                    fmt="o-", ms=3, color="#7b2cbf", label="BB84 Monte Carlo")
    axs[2].plot([c["eve_fraction"] for c in C], [100 * c["theory"] for c in C], "k--", lw=0.8, label="theory")
    axs[2].axhline(11.0, color="r", lw=0.7, ls=":"); axs[2].text(0.45, 11.4, "11.0% (ideal reconciliation)", fontsize=5.5, color="r")
    axs[2].axhline(100 * B["threshold_f1.16"], color="darkorange", lw=0.7, ls="--")
    axs[2].text(0.0, 8.3, "9.8% (f_ec = 1.16)", fontsize=5.5, color="darkorange")
    axs[2].set_xlabel("Intercept-resend fraction"); axs[2].set_ylabel("Estimated QBER (%)"); axs[2].legend(fontsize=6)
    savefig("fig4_key_demand_and_qkd.png")
    # Fig 7: shared link
    D = d["D_shared_link"]
    provs = sorted(float(k.split("_")[1]) for k in D if k.startswith("prov_"))
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for m, col, lab in (("one_link_each", "#2a9d8f", "dedicated link per EAN"), ("shared_pooled", "#7b2cbf", "shared link, pooled buffer"),
                        ("shared_equal_split", "#e76f51", "shared link, static 1/N split")):
        ax.plot(provs, [100 * D[f"prov_{p}"][m]["mean"] for p in provs], "o-", ms=3, color=col, label=lab)
    ax.set_xlabel("Key provisioning (supply / demand)"); ax.set_ylabel("Rounds without enough key (%)"); ax.legend(fontsize=5.5)
    savefig("fig7_shared_link.png")


# --------------------------------------------------------------------------------------------- LOO detector
def part_loo():
    from qse.vqad.detector import VQAD, MLPDetector, LogisticDetector
    z = np.load(ROOT / "results" / "pretrain_fingerprints.npz")
    X, y, atk = z["X"], z["y"], z["attack"]
    attacks = sorted(set(atk))
    res = {}
    for held in attacks:
        tr = np.where((atk != held) | (y > 0))[0]
        # keep benign samples of held-out run out of training so FPR is truly held-out
        tr = np.where((atk != held))[0]
        te = np.where(atk == held)[0]
        for mk in (VQAD, MLPDetector, LogisticDetector):
            det = mk()
            det.fit(X[tr], y[tr], epochs=300, restarts=4)
            fl = det.flag(X[te])
            yt = y[te]
            res[f"{det.name}|{held}"] = dict(tpr=float(fl[yt < 0].mean()), fpr=float(fl[yt > 0].mean()))
        print("held-out", held, {k: v for k, v in res.items() if k.endswith(held)}, flush=True)
    (TAB / "leave_one_attack_out.json").write_text(json.dumps(res, indent=1))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=False)
    w = 0.26
    for j, (nm, col) in enumerate((("VQAD", "#7b2cbf"), ("MLP", "#e76f51"), ("Linear", "#f4a261"))):
        axs[0].bar(np.arange(len(attacks)) + (j - 1) * w, [100 * res[f"{nm}|{a}"]["tpr"] for a in attacks], w, color=col, label=nm)
        axs[1].bar(np.arange(len(attacks)) + (j - 1) * w, [100 * res[f"{nm}|{a}"]["fpr"] for a in attacks], w, color=col)
    for ax, t in zip(axs, ("TPR on held-out attack (%)", "FPR on held-out benign (%)")):
        ax.set_xticks(range(len(attacks))); ax.set_xticklabels([ATK_LABEL[a] for a in attacks], rotation=20, fontsize=6.5)
        ax.set_ylabel(t)
    axs[0].legend(fontsize=6)
    savefig("fig9_leave_one_attack_out.png")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["main", "keyrate", "qber", "finitekey", "downlink", "sparsity", "keys", "loo"]
    for p in parts:
        print("== part", p, flush=True)
        if p == "main":
            part_main()
        elif p == "keyrate":
            part_keyrate()
        elif p == "qber":
            part_qber()
        elif p == "finitekey":
            part_simple("finitekey", r"fk_(\w+)_x([0-9.]+)_s(\d+)", "finitekey")
        elif p == "downlink":
            part_simple("downlink", r"dl_(\w+?)_x([0-9.]+)_s(\d+)", "downlink")
        elif p == "sparsity":
            part_sparsity()
        elif p == "keys":
            part_keys()
        elif p == "loo":
            part_loo()
        elif p == "ablation":
            runs = load_suite("ablation")
            agg = defaultdict(list)
            for n, r in runs.items():
                m = re.match(r"ab_vqad_fw([0-9.]+)_(\w+?)_s(\d+)", n)
                L = r["log"]
                agg[(m.group(1), m.group(2))].append((r["final_acc"], sum(L["tp"]) / max(1, sum(L["tp"]) + sum(L["fn"])),
                                                       sum(L["fp"]) / max(1, sum(L["fp"]) + sum(L["tn"]))))
            (TAB / "ablation_flagged_weight.json").write_text(json.dumps(
                {"|".join(k): dict(acc=float(np.mean([x[0] for x in v])), tpr=float(np.mean([x[1] for x in v])),
                                   fpr=float(np.mean([x[2] for x in v]))) for k, v in agg.items()}, indent=1))
