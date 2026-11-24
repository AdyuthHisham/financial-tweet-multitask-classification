"""Two-phase training curves of the multi-task model (from notebook outputs)."""
import matplotlib.pyplot as plt
from common import *

style()
h, f = load("approachB_head_phase.json"), load("approachB_full_phase.json")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(h["epochs"], h["val_acc_company_L"], color=BLUE, label="copy L")
ax[0].plot(h["epochs"], h["val_acc_company_R"], color=ORANGE, label="copy R")
ax[0].set(title="Phase 1: company head only (encoder frozen)", xlabel="epoch", ylabel="validation accuracy, company")
ax[1].plot(f["epochs"], f["val_acc_sentiment_L"], color=GREEN, label="sentiment, L")
ax[1].plot(f["epochs"], f["val_acc_sentiment_R"], color=GREEN, ls="--", label="sentiment, R")
ax[1].plot(f["epochs"], f["val_acc_company_L"], color=BLUE, label="company, L")
ax[1].plot(f["epochs"], f["val_acc_company_R"], color=BLUE, ls="--", label="company, R")
ax[1].set(title="Phase 2: full fine-tuning", xlabel="epoch", ylabel="validation accuracy")
for a in ax:
    a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    a.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(MEDIA / "training_curves.png", dpi=150)
