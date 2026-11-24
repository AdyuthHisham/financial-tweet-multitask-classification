"""Hero GIF: validation accuracy of the multi-task model growing epoch by epoch, with a held-out test readout."""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from common import *

style(card=True)
f, t = load("approachB_full_phase.json"), load("approachB_test.json")
sel = t["checkpoints"]["final_best_model_A.pt"]
n = len(f["epochs"])
fig, ax = plt.subplots(figsize=(7, 3.6), dpi=70)
ax.set(xlim=(1, n), ylim=0.4, xlabel="epoch", ylabel="validation accuracy")
ax.set_ylim(0.4, 0.9); ax.set_xticks(range(2, n + 1, 2))
ls, = ax.plot([], [], color=GREEN, lw=2.5, label="sentiment")
lc, = ax.plot([], [], color=BLUE, lw=2.5, label="company")
ax.legend(loc="lower right", frameon=False)
title = ax.set_title("One encoder, two heads: financial tweet classification", fontsize=11)
box = ax.text(1.4, 0.855, "", fontsize=9)

def upd(k):
    k = min(k, n - 1) + 1
    ls.set_data(f["epochs"][:k], f["val_acc_sentiment_L"][:k])
    lc.set_data(f["epochs"][:k], f["val_acc_company_L"][:k])
    box.set_text(f"epoch {k}/{n}" + (f"   test: sentiment {sel['accuracy']['sentiment']:.2f}, company {sel['accuracy']['company']:.2f}" if k == n else ""))
    return ls, lc, box

fig.tight_layout()
FuncAnimation(fig, upd, frames=n + 12, interval=1000 / 6).save(MEDIA / "hero.gif", writer=PillowWriter(fps=6))
