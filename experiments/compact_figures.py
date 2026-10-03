"""Compact, column-width (3.4 in) figures for a two-column conference layout. Usage: python experiments/compact_figures.py"""

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import ROOT, load_suite  # noqa: E402

OUT = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "legend.fontsize": 5.8, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.0, "lines.markersize": 2.5,
                     "savefig.dpi": 400})
POL = {"adapt": "#7b2cbf", "wait": "#2a9d8f", "defer": "#e76f51", "pq": "#6c757d", "nominal": "k"}


def save(fig, name):
    fig.tight_layout(pad=0.4, rect=(0, 0.07, 1, 1)) if name.endswith("transient.png") else fig.tight_layout(pad=0.4)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("saved", name)


def fig_arch():
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    B, T, R, G = "#1f4e79", "#2a9d8f", "#b5179e", "#6c757d"

    def box(x, y, w, h, txt, fc, ec, fs=5.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)

    box(2.6, 4.9, 4.8, 1.2, "Cloud aggregator\nverify tag, decrypt, FedAvg;\nsealed sparse model delta", "#e8eef5", B)
    xs = [0.2, 3.6, 7.0]
    for i, x in enumerate(xs):
        box(x, 2.3, 2.8, 1.5, f"EAN {i + 1}\nrobust / detector\naggregation\nTop-K + error feedback\nOTP + MAC", "#e9f5f3", T, fs=5.4)
        ax.add_patch(FancyArrowPatch((x + 1.4, 3.8), (3.2 + i * 1.8, 4.9), arrowstyle="<|-|>", mutation_scale=7, color=R, lw=1.6))
        ax.text(x + 1.4 + (3.2 + i * 1.8 - x - 1.4) / 2 + 0.55, 4.3, "QKD", fontsize=5.5, color=R, ha="left")
        for k in range(3):
            ax.add_patch(plt.Circle((x + 0.55 + k * 0.85, 0.75), 0.22, fc="#f1f1f1", ec=G, lw=0.7))
        ax.add_patch(FancyArrowPatch((x + 1.4, 1.1), (x + 1.4, 2.3), arrowstyle="<|-|>", mutation_scale=6, color=G, lw=0.9, ls="--"))
    ax.text(5.0, 0.12, "IoT leaves: local SGD, some Byzantine", ha="center", fontsize=5.6, color=G)
    save(fig, "compact_fig1_architecture.png")


def fig_demand_finitekey():
    d = json.loads((ROOT / "results" / "key_analysis.json").read_text())
    A = d["A_required_kbps_per_link"]
    names = list(A)
    P = [A[n]["P"] for n in names]
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.7))
    ax = axs[0]
    for dd, col in (("0.01", "#2a9d8f"), ("0.05", "#7b2cbf"), ("0.25", "#e76f51")):
        ax.plot(P, [A[n]["sparse_otp"][dd] for n in names], "o-", color=col, label=f"sparse {float(dd):.0%}")
    ax.plot(P, [A[n]["dense_otp"]["0.05"] for n in names], "s--", color="k", label="dense downlink")
    ax.axhspan(10, 100, color="grey", alpha=0.15)
    ax.text(P[0] * 1.1, 52, "10-100 kbps\n(illustrative)", fontsize=5, va="center")
    ax.set_ylim(0.15, 6000)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Model parameters P"); ax.set_ylabel("Key rate per link (kbps)")
    ax.legend(loc="upper left", fontsize=4.8, ncol=1, framealpha=0.9)
    B = d["B_secret_fraction"]
    q = np.array(B["qber"]) * 100
    ax = axs[1]
    ax.plot(q, B["asymptotic"], "k-", label="asymptotic")
    for n, col in zip((10_000, 100_000, 1_000_000), ("#e76f51", "#e9c46a", "#2a9d8f")):
        ax.plot(q, B[f"finite_{n}"], color=col, label=f"n=1e{int(np.log10(n))}")
    ax.axvline(100 * B["threshold_f1.16"], color="k", ls=":", lw=0.6)
    ax.set_xlabel("QBER (%)"); ax.set_ylabel("Secret-key fraction"); ax.legend(fontsize=5)
    save(fig, "compact_fig2_demand_finitekey.png")


def fig_policies():
    runs = load_suite("keyrate")
    if not runs:
        print("keyrate data missing; skip"); return None
    from qse.qkd.keypool import LinkConfig, QKDLink
    pr = QKDLink(LinkConfig(), 10 ** 9, seed=1)
    mean_rate = float(np.mean([pr.sample_state()[1] for _ in range(2000)]))
    demand = json.loads((ROOT / "results" / "key_analysis.json").read_text())["D_shared_link"]["demand_bits_per_ean_round"]
    ratio = lambda s: s * mean_rate * 60.0 / demand
    agg = defaultdict(list)
    for n, r in runs.items():
        m = re.match(r"kr_(\w+)_x([0-9.]+)_s(\d+)", n)
        L = r["log"]
        ok = sum(L["n_ok"]) + sum(L["n_adapted"])
        agg[(m.group(1), float(m.group(2)))].append((r["final_acc"], ok / (4 * len(L["n_ok"])), float(np.mean(L["delay_s"]))))
    scales = sorted({s for _, s in agg})
    fig, axs = plt.subplots(1, 3, figsize=(3.45, 1.45))
    for p in ("adapt", "wait", "defer", "pq"):
        xs = [ratio(s) for s in scales]
        axs[0].plot(xs, [100 * np.mean([x[0] for x in agg[(p, s)]]) for s in scales], "o-", color=POL[p], label=p)
        axs[1].plot(xs, [100 * np.mean([x[1] for x in agg[(p, s)]]) for s in scales], "o-", color=POL[p])
    axs[2].plot([ratio(s) for s in scales], [np.mean([x[2] for x in agg[("wait", s)]]) for s in scales], "o-", color=POL["wait"])
    for ax in axs:
        ax.set_xscale("log"); ax.axvline(1.0, color="k", lw=0.5, ls=":"); ax.set_xlabel("Supply / demand")
    axs[0].set_ylabel("Accuracy (%)"); axs[1].set_ylabel("OTP exchanges (%)"); axs[2].set_ylabel("Extra delay (s)")
    axs[0].legend(fontsize=5)
    save(fig, "compact_fig3_policies.png")
    return {f"{p}|{s}": dict(ratio=ratio(s), acc=float(np.mean([x[0] for x in agg[(p, s)]])),
                             otp=float(np.mean([x[1] for x in agg[(p, s)]])), delay=float(np.mean([x[2] for x in agg[(p, s)]])))
            for (p, s) in agg}


def fig_transient():
    runs = load_suite("qevsmall")
    if not runs:
        print("qevsmall data missing; skip"); return None
    curves = defaultdict(list)
    for n, r in runs.items():
        m = re.match(r"qevs_(\w+)_s(\d+)", n)
        L = r["log"]
        curves[m.group(1)].append((np.array(L["pool_level"]).mean(1), L["acc"], L["acc_round"],
                                   (np.array(L["n_ok"]) + np.array(L["n_adapted"])) / 4.0))
    fig, axs = plt.subplots(1, 3, figsize=(3.45, 1.7))
    style = {"nominal": dict(color="k", lw=0.9), "pq": dict(color=POL["pq"], lw=2.2), "defer": dict(color=POL["defer"], lw=1.3, ls="--"),
             "wait": dict(color=POL["wait"], lw=1.0, ls=":"), "adapt": dict(color=POL["adapt"], lw=1.0)}
    for p in ("pq", "defer", "wait", "adapt", "nominal"):
        c = curves[p]
        lvl = np.mean([x[0] for x in c], 0)
        axs[0].plot(np.arange(1, len(lvl) + 1), 100 * lvl, label=p, **style[p])
        axs[1].plot(c[0][2], 100 * np.mean([x[1] for x in c], 0), **style[p])
        ok = np.mean([x[3] for x in c], 0)
        axs[2].plot(np.arange(1, len(ok) + 1), 100 * ok, **style[p])
    for ax in axs:
        ax.axvspan(20, 40, color="red", alpha=0.08); ax.set_xlabel("FL round")
    axs[0].set_ylabel("Pool (%)"); axs[1].set_ylabel("Accuracy (%)"); axs[2].set_ylabel("OTP exchanges (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=5.2, frameon=False)
    save(fig, "compact_fig4_transient.png")
    s = {}
    for p, c in curves.items():
        s[p] = dict(final_acc=float(np.mean([x[1][-1] for x in c])), min_level=float(np.min([x[0].min() for x in c])),
                    otp_frac_event=float(np.mean([x[3][19:40].mean() for x in c])))
    (TAB / "transient_small_buffer.json").write_text(json.dumps(s, indent=1))
    return s


if __name__ == "__main__":
    fig_arch()
    fig_demand_finitekey()
    pol = fig_policies()
    if pol:
        (TAB / "keyrate_policies.json").write_text(json.dumps(pol, indent=1))
    fig_transient()
