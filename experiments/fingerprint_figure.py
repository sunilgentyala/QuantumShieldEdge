"""Distribution of the four scale-free fingerprint features for benign and attacked updates (pre-training data)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
z = np.load(ROOT / "results" / "pretrain_fingerprints.npz")
X, y, atk = z["X"], z["y"], z["attack"]
order = ["benign", "sign_flip", "scale", "label_flip", "alie", "min_max"]
labels = ["Benign", "Sign flip", "Scale", "Label flip", "ALIE", "Min-max"]
names = [r"$f_0$ (log norm ratio)", r"$f_1$ (cosine to median)", r"$f_2$ (kurtosis shift)", r"$f_3$ (positive fraction)"]
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
stats = {}
for j, ax in enumerate(axes):
    data = []
    for o in order:
        v = X[y > 0, j] if o == "benign" else X[(y < 0) & (atk == o), j]
        data.append(v)
        stats[f"{o}|f{j}"] = dict(mean=float(v.mean()), std=float(v.std()), n=int(len(v)))
    bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=0.6)
    for k, b in enumerate(bp["boxes"]):
        b.set(facecolor="#9db7d5" if k == 0 else "#e6a8a0", edgecolor="#333", linewidth=0.7)
    for m in bp["medians"]:
        m.set(color="#111", linewidth=1.0)
    ax.set_title(names[j], fontsize=7.5)
    ax.set_xlim(-1.05, 1.05)
    ax.grid(alpha=0.25, lw=0.4)
axes[0].set_yticks(range(1, 7))
axes[0].set_yticklabels(labels)
fig.supxlabel("Feature value (all features scaled to [-1, 1])", fontsize=8)
plt.tight_layout(pad=0.5)
out = ROOT / "results" / "figures" / "fig_fingerprints.png"
plt.savefig(out, dpi=300)
(ROOT / "results" / "tables" / "fingerprint_stats.json").write_text(json.dumps(stats))
print("saved", out)
