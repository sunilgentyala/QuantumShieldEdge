"""Fig. 1: three-tier architecture, QKD links, VQAD placement and per-link security assumption."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
BLUE, TEAL, ORANGE, GREY, RED = "#1f4e79", "#2a9d8f", "#e76f51", "#6c757d", "#b5179e"


def box(ax, x, y, w, h, text, fc, ec=None, fs=8, tc="black", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc,
                                ec=ec or fc, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, p, q, color, style="-", lw=1.8, both=True, label=None, lx=0, ly=0, fs=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=10, color=color,
                                 lw=lw, linestyle=style))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, fontsize=fs, color=color, ha="center",
                va="center", bbox=dict(fc="white", ec="none", pad=0.6))


def main(out: Path | None = None):
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")

    # Tier 3: cloud
    box(ax, 2.2, 5.35, 5.6, 1.0, "", "#e8eef5", BLUE, lw=1.5)
    ax.text(5.0, 6.17, "Tier 3: cloud aggregator", ha="center", fontsize=9, weight="bold", color=BLUE)
    box(ax, 2.45, 5.45, 2.3, 0.55, "Decrypt + verify tag\nFedAvg over EAN deltas", "white", BLUE, fs=7)
    box(ax, 5.25, 5.45, 2.3, 0.55, "Top-K model delta\nOTP + MAC to each EAN", "white", BLUE, fs=7)

    # Tier 2: EANs
    xs = [0.35, 2.85, 5.35, 7.85]
    for i, x in enumerate(xs):
        box(ax, x, 2.75, 1.8, 1.55, "", "#e9f5f3", TEAL, lw=1.3)
        ax.text(x + 0.9, 4.15, f"EAN {i + 1}", ha="center", fontsize=8, weight="bold", color=TEAL)
        box(ax, x + 0.1, 3.45, 1.6, 0.5, "VQAD / robust agg.\n(12-13 updates)", "#fff3e0", ORANGE, fs=6.5)
        box(ax, x + 0.1, 2.85, 1.6, 0.5, "Top-K + EF, fp16\nOTP + WC tag", "white", TEAL, fs=6.5)
        # QKD link EAN <-> cloud
        cx = x + 0.9
        tx = 3.0 + i * 1.33
        arrow(ax, (cx, 4.3), (tx, 5.35), RED, style="-", lw=2.2)
        # QKD transceivers
        box(ax, x + 0.55, 4.34, 0.7, 0.2, "QKD pool", "#fde2f3", RED, fs=5.5)

    # Tier 1: leaves
    for i, x in enumerate(xs):
        for k in range(4):
            lx = x + 0.1 + k * 0.45
            ax.add_patch(plt.Circle((lx + 0.2, 1.05), 0.17, fc="#f1f1f1", ec=GREY, lw=0.9))
        ax.text(x + 0.9, 0.45, "IoT leaf devices\n(local SGD, some Byzantine)", ha="center", fontsize=6.5,
                color=GREY)
        arrow(ax, (x + 0.9, 1.3), (x + 0.9, 2.75), GREY, style="--", lw=1.4)
    ax.text(0.0, 1.9, "Tier 1", fontsize=9, weight="bold", color=GREY, rotation=90, va="center")
    ax.text(0.0, 3.55, "Tier 2", fontsize=9, weight="bold", color=TEAL, rotation=90, va="center")

    # legend
    ax.plot([0.4, 1.1], [0.02, 0.02], color=RED, lw=2.2)
    ax.text(1.2, 0.02, "EAN-cloud (both directions): QKD keys, one-time pad + Wegman-Carter tag, "
            "information-theoretic", fontsize=6.8, va="center")
    ax.plot([0.4, 1.1], [-0.3, -0.3], color=GREY, lw=1.4, ls="--")
    ax.text(1.2, -0.3, "leaf-EAN: post-quantum KEM + AEAD, computational security (assumed, not simulated)",
            fontsize=6.8, va="center")
    ax.set_ylim(-0.55, 6.6)
    fig.tight_layout()
    out = out or (ROOT / "results" / "figures" / "fig1_architecture.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main()
