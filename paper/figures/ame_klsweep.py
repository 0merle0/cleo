#!/usr/bin/env python3
"""The KL anchor is a dial with an interior optimum, not a monotone knob.

Two things move in opposite directions as the anchor weight `w` rises, and the
library cares about neither of them alone:

  success rate      falls monotonically   83 -> 10 %
  per-design cover  rises monotonically   U_15 510 -> 881

What a library actually contains is the *total* distinct substitution count U,
which is the product of those two effects and therefore unimodal. It is flat
across w in [0.01, 0.05] and collapses at w = 0.1, where so few designs pass
that per-design exploration has nothing to accumulate on.

  Left   the trade-off itself: success against per-design coverage, one point
         per w. This is the frontier the anchor moves along.
  Right  the punchline: U against w, with the plateau and the cliff.

Deliberately NOT one panel with two y-axes. Success and U_15 have unrelated
scales, and a dual axis lets the crossing point be placed anywhere by choosing
the scales -- the apparent "optimum" would be an artefact of the drawing.

Encoding: w is an ordinal ladder, so it gets a single hue ramped light to dark
rather than categorical hues. w = 0.02 is drawn in the accent hue because it is
the recommended default, not because it is another rung.

    uv run python paper/figures/ame_klsweep.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.titlesize": 15, "axes.linewidth": 1.0,
})
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figio import save  # noqa: E402
from palette import PALETTE  # noqa: E402

ACCENT = PALETTE["red"]
CHOSEN = 0.02

# Final 40 steps of each 200-step run; diversity on passing, deduplicated
# designs. Produced by the E13 sweep -- see experiments.tex Table (tab:e13).
W      = [0.0,  0.005, 0.01, 0.02, 0.05, 0.10]
PASS   = [83.1, 81.1,  57.7, 56.7, 40.5, 10.2]
U15    = [510,  562,   760,  808,  858,  881]
U      = [2109, 2230,  2498, 2501, 2480, 1743]


def ramp(hex_color, n, lo=0.35, hi=1.0):
    """-> n shades of one hue, light to dark. Sequential encoding for an ordinal family."""
    base = np.array(matplotlib.colors.to_rgb(hex_color))
    return [tuple(1 - (1 - base) * f) for f in np.linspace(lo, hi, n)]


def colors():
    """-> one colour per w: the blue ramp, with the recommended default in accent."""
    c = ramp(PALETTE["blue"], len(W))
    return [ACCENT if w == CHOSEN else ci for w, ci in zip(W, c)]


def label(w):
    return "$w$=0" if w == 0 else f"{w:g}"


def panel_frontier(ax):
    """Success against per-design coverage: the trade-off the anchor moves along."""
    cols = colors()
    ax.plot(U15, PASS, "-", color=PALETTE["gray"], lw=1.4, alpha=0.6, zorder=1)
    for w, x, y, c in zip(W, U15, PASS, cols):
        big = w == CHOSEN
        ax.scatter([x], [y], s=190 if big else 120, color=c, zorder=3,
                   edgecolors="white", linewidths=2.0)
        # Labels outboard of the curve so they never sit on the connecting line.
        ax.annotate(label(w), (x, y), textcoords="offset points",
                    xytext=(10, 8), fontsize=10, color=c,
                    fontweight="bold" if big else "normal")
    ax.annotate("recommended", (U15[W.index(CHOSEN)], PASS[W.index(CHOSEN)]),
                textcoords="offset points", xytext=(14, -20), fontsize=10,
                color=ACCENT, fontweight="bold")
    ax.set_xlabel("per-design coverage $U_{15}$ (distinct substitutions)")
    ax.set_ylabel("passing designs (%)")
    ax.set_title("The anchor trades success for exploration...", pad=8)
    ax.set_ylim(0, 95)
    ax.grid(color="#E5E7EB", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_total(ax):
    """Total distinct substitutions against w: flat, then a cliff."""
    cols = colors()
    x = np.arange(len(W))
    ax.bar(x, U, color=cols, width=0.62, zorder=2)
    for xi, u in zip(x, U):
        ax.text(xi, u + 55, f"{u}", ha="center", fontsize=10, color="#374151")

    # The plateau, marked with a span between the two ends rather than asserted
    # in prose. No shaded band behind it: the bars already carry the value, and
    # a floating rectangle at their tops reads as a second, unexplained series.
    lo, hi = 1.75, 4.25
    ax.annotate("", xy=(lo, 2760), xytext=(hi, 2760),
                arrowprops=dict(arrowstyle="<->", color="#374151", lw=1.2))
    ax.text((lo + hi) / 2, 2810, "flat within noise", ha="center", fontsize=10,
            color="#374151")
    # Above the last bar, in the only genuinely empty region of the panel. Bars
    # fill from zero, so there is no free space *below* a bar top: placing it
    # there put dark text on a mid-blue fill and dragged the arrow across three
    # other bars. The x limit is extended to make room rather than letting the
    # text run past the axis.
    ax.annotate("too few designs\npass to cover\nanything",
                xy=(5, 1900), xytext=(5, 2560), fontsize=10, color="#374151",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.3))
    ax.set_xlim(-0.6, 5.9)

    ax.set_xticks(x)
    ax.set_xticklabels([label(w) for w in W])
    ax.set_xlabel("KL weight $w$")
    ax.set_ylabel("total distinct substitutions $U$")
    ax.set_title("...but total library coverage peaks in between", pad=8)
    ax.set_ylim(0, 3100)
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    panel_frontier(axes[0])
    panel_total(axes[1])
    fig.suptitle("The KL anchor has an interior optimum: 0.02 sits in the plateau, "
                 "not on a slope")
    fig.tight_layout()
    save(fig, "ame_klsweep")


if __name__ == "__main__":
    main()
