"""
Shared plotting style and color palette for CLEO notebooks.

Usage:
    from plot_utils import setup_style, CLEO_PALETTE, PRIMARY, SECONDARY

    setup_style()
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# --- Fragment-inspired palette (matches protein structure figures) ---
PRIMARY   = "#5B8DB8"  # muted blue
SECONDARY = "#E07B73"  # soft coral
TEAL      = "#5DAE8B"  # teal green
PURPLE    = "#9B7CB8"  # purple / lavender
AMBER     = "#E8B15E"  # warm amber

CLEO_PALETTE = [PRIMARY, TEAL, PURPLE, SECONDARY, AMBER]

# Semantic colors
COLOR_REFERENCE = "#888888"  # gray — baselines, y=x lines
COLOR_THRESHOLD = "#C62828"  # dark red — cutoffs, thresholds


def setup_style():
    """Apply the standard CLEO notebook plotting style.

    Call this once at the top of each notebook after imports.
    """
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
