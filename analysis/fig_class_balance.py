"""Company class balance of the Kaggle stock-tweet set (counts from docs/metrics/stock_class_counts.json)."""
import matplotlib.pyplot as plt
from common import *

style()
c = load("stock_class_counts.json")
names, vals = list(c["counts"]), list(c["counts"].values())
colors = [BLUE if v >= 200 else VERMILION for v in vals]
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(names, vals, color=colors)
ax.set_yscale("log")
ax.axhline(200, color=GREY, ls="--", lw=1)
ax.text(len(names) - 0.5, 230, "200-tweet threshold (rarer tickers merged into OTHERS)", ha="right", fontsize=8)
ax.set_ylabel("tweets (log scale)")
ax.set_title(f"Kaggle stock tweets: {c['rows']:,} rows, {c['tickers']} tickers; TSLA is {c['share_tsla']:.0%}", fontsize=10)
plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(MEDIA / "class_balance.png", dpi=150)
