"""
Shared plotting style and color palette for CLEO (notebooks and scripts).

Notebooks can keep ``from plot_utils import ...`` via the thin re-export in
``notebooks/plot_utils.py``.
"""

import matplotlib as mpl
import seaborn as sns

# --- Fragment-inspired palette (matches protein structure figures) ---
PRIMARY = "#5B8DB8"  # muted blue
SECONDARY = "#E07B73"  # soft coral
TEAL = "#5DAE8B"  # teal green
PURPLE = "#9B7CB8"  # purple / lavender
AMBER = "#E8B15E"  # warm amber

CLEO_PALETTE = [PRIMARY, TEAL, PURPLE, SECONDARY, AMBER]

# Semantic colors
COLOR_REFERENCE = "#888888"  # gray — baselines, y=x lines
COLOR_THRESHOLD = "#C62828"  # dark red — cutoffs, thresholds


def setup_style() -> None:
    """Apply the standard CLEO plotting style (ticks theme + typography)."""
    sns.set_theme(style="ticks")
    mpl.rcParams["axes.grid"] = False
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    mpl.rcParams["axes.linewidth"] = 1.4
    mpl.rcParams["xtick.major.width"] = 1.2
    mpl.rcParams["ytick.major.width"] = 1.2
    mpl.rcParams["figure.dpi"] = 120
    mpl.rcParams["savefig.dpi"] = 150
    mpl.rcParams["savefig.bbox"] = "tight"


def campaign_color_map(categories: list[str]) -> dict[str, str]:
    """Map each campaign name to a CLEO palette color (cycles if needed)."""
    cats = sorted(set(categories))
    return {c: CLEO_PALETTE[i % len(CLEO_PALETTE)] for i, c in enumerate(cats)}
