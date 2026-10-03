"""Graphical abstract for the journal submission: 2656 x 1062 px (500:200 ratio, 2x the 1328 x 531 minimum) at 300 dpi.
All text sits inside the image, as the publisher requires. Numbers are read from the result tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "results" / "tables"
main = json.loads((T / "main_accuracy.json").read_text())["table"]
loo = json.loads((T / "loo_seeds_v2.json").read_text())


def ba(x):
    return 0.5 * (x["tpr"] + 1 - x["fpr"])


dets = ["VQAD", "MLP", "Linear", "FourierMLP", "FourierLinear"]
n = min(len(v) for v in loo.values())
loo_mean = {d: float(np.mean([ba(s) for k, v in loo.items() if k.startswith(d + "|") for s in v[:n]])) for d in dets}

BLUE, TEAL, PURPLE, ORANGE, GREY, RED = "#00467f", "#2a9d8f", "#7b2cbf", "#e76f51", "#6c757d", "#b5179e"
fig = plt.figure(figsize=(8.853, 3.54), dpi=300)  # 8.853 x 3.54 in at 300 dpi = 2656 x 1062 px
fig.patch.set_facecolor("white")

# panel 1: pipeline
ax = fig.add_axes([0.01, 0.03, 0.31, 0.88]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.text(0.2, 9.6, "1  QKD-keyed three-tier pipeline", fontsize=8.5, weight="bold", color=BLUE, va="center")
def box(x, y, w, h, t, fc, ec, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
box(1.6, 7.0, 6.8, 1.6, "Cloud: verify, decrypt, aggregate", "#e8eef5", BLUE, fs=6.2)
for i, x in enumerate((0.3, 3.6, 6.9)):
    box(x, 3.7, 2.8, 2.2, "EAN\ndetector-weighted\naggregation\nOTP + MAC", "#e9f5f3", TEAL, fs=6.0)
    ax.add_patch(FancyArrowPatch((x + 1.4, 5.9), (3.3 + i * 1.7, 7.0), arrowstyle="<|-|>", mutation_scale=9, color=RED, lw=2.0))
    for k in range(3):
        ax.add_patch(plt.Circle((x + 0.6 + k * 0.8, 1.7), 0.32, fc="#f1f1f1", ec=GREY, lw=0.9))
    ax.add_patch(FancyArrowPatch((x + 1.4, 2.2), (x + 1.4, 3.7), arrowstyle="<|-|>", mutation_scale=7, color=GREY, lw=1.0, ls="--"))
ax.text(5.0, 0.55, "magenta: QKD one-time pad + tag;  leaves: some Byzantine", fontsize=5.8, ha="center", color=GREY)

# panel 2: accuracy under attack
ax2 = fig.add_axes([0.375, 0.2, 0.28, 0.62])
names = [("fedavg", "FedAvg", GREY), ("median", "Median", "#2a9d8f"), ("trimmed_mean", "Trim.", "#e9c46a"), ("mlp", "MLP", ORANGE), ("vqad", "VQAD", PURPLE)]
x = np.arange(2)
w = 0.16
for j, (a, lab, col) in enumerate(names):
    vals = [100 * main[f"{a}|{k}"]["mean"] for k in ("sign_flip", "min_max")]
    ax2.bar(x + (j - 2) * w, vals, w, color=col, label=lab)
ax2.set_xticks(x); ax2.set_xticklabels(["Sign flip", "Min-max"], fontsize=7)
ax2.set_ylabel("Accuracy (%)", fontsize=7); ax2.tick_params(labelsize=6.5); ax2.set_ylim(0, 36)
ax2.legend(fontsize=5.6, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
fig.text(0.375, 0.93, "2  Edge weighting vs robust rules", fontsize=8.5, weight="bold", color=BLUE, va="center")

# panel 3: leave-one-attack-out
ax3 = fig.add_axes([0.71, 0.2, 0.27, 0.62])
order = ["VQAD", "FourierLinear", "FourierMLP", "MLP", "Linear"]
lab = {"VQAD": "VQC\n(24)", "FourierLinear": "Fourier\nlin. (9)", "FourierMLP": "Fourier\nMLP (31)", "MLP": "MLP\n(25)", "Linear": "Linear\n(5)"}
cols = [PURPLE, "#8d99ae", "#8d99ae", ORANGE, GREY]
ax3.bar(range(5), [loo_mean[d] for d in order], color=cols)
for i, d in enumerate(order):
    ax3.text(i, loo_mean[d] + 0.015, f"{loo_mean[d]:.2f}", ha="center", fontsize=6.5)
ax3.set_xticks(range(5)); ax3.set_xticklabels([lab[d] for d in order], fontsize=5.8)
ax3.set_ylim(0, 1); ax3.set_ylabel("Balanced acc., unseen attack", fontsize=6.6); ax3.tick_params(labelsize=6.5)
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)
fig.text(0.71, 0.93, "3  Quantum vs classical, unseen", fontsize=8.5, weight="bold", color=BLUE, va="center")
fig.text(0.71, 0.035, "No quantum advantage claimed: simulation, small model", fontsize=6, color=GREY)

out = ROOT / "results" / "figures" / "graphical_abstract.png"
fig.savefig(out, dpi=300)
from PIL import Image
im = Image.open(out)
print("size", im.size)
im.convert("RGB").save(ROOT / "results" / "figures" / "graphical_abstract.tiff", dpi=(300, 300))
