"""Approach A vs B on the test set, numbers from the report tables (A) and the notebook output (B)."""
import matplotlib.pyplot as plt
import numpy as np
from common import *

style()
r = load("report_tables.json")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, task in zip(axes, ("sentiment", "company")):
    names = list(r[task])
    x = np.arange(3)
    for k, (n, col) in enumerate(zip(names, (BLUE, ORANGE))):
        v = [r[task][n][m] for m in ("accuracy", "f1_macro", "f1_weighted")]
        bars = ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=col, label=n)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=GREY)
    ax.set_xticks(x, ["accuracy", "F1 macro", "F1 weighted"])
    ax.set_title(task.capitalize(), fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
axes[0].set_ylabel("test score")
fig.tight_layout()
fig.savefig(MEDIA / "results.png", dpi=150)
