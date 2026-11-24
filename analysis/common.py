"""Shared paths, palette and plot style. Okabe-Ito colours (colour-blind safe)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "docs" / "metrics"
MEDIA = ROOT / "media"
MEDIA.mkdir(exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, VERMILION, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#888888"
CARD_BG, CARD_FG = "#F6F8FA", "#24292F"


def style(card=False):
    """Mid-grey text on transparent background (PNG), or dark text on a light card (GIF)."""
    fg = CARD_FG if card else GREY
    plt.rcParams.update({
        "text.color": fg, "axes.labelcolor": fg, "xtick.color": fg, "ytick.color": fg,
        "axes.edgecolor": fg, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "figure.facecolor": CARD_BG if card else "none",
        "axes.facecolor": CARD_BG if card else "none",
        "savefig.facecolor": CARD_BG if card else "none", "savefig.transparent": not card,
    })


def load(name):
    return json.loads((METRICS / name).read_text())


def dump(name, obj):
    (METRICS / name).write_text(json.dumps(obj, indent=2) + "\n")
