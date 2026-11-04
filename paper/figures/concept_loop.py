#!/usr/bin/env python3
"""Figure 2, panel (a) --- the post-training loop.

Moved out of Figure 1. Figure 1 is the conceptual argument -- what a
combinatorial library is, what it costs, why a filtered one searches better,
and how the experimental loop around it closes. None of that depends on how
the policy is trained. THIS is the training, it is what Figure 2's results are
measurements of, and it belongs at the head of that figure.

The experimental loop that briefly lived here as panel (b) is now Figure 1's
panel (d): it is part of the concept, not part of the method.

Two things this states that otherwise live only in CLAUDE.md: the fold is
best-of-5, which is the AME benchmark's own unit; and the recombination step
includes the parent itself (the rho=0 rotation), which is why an in-loop
chimera pass rate is NOT comparable to a T2 retention number.

    uv run python paper/figures/concept_loop.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figio import save  # noqa: E402
from palette import (INK, PAGE_WIDTH_IN, PALETTE,  # noqa: E402
                     PANEL_LABEL_PT, fs)
from schematic import FRAGMENT, arrow, bar, box, place  # noqa: E402

PNG = HERE / "structures" / "png"
K = 4
GREY = INK["muted"]


def panel_loop(ax, rng):
    """backbone -> policy -> sequences -> fragments -> chimeras -> reward."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.1)
    ax.axis("off")

    place(ax, "grey", (0.72, 2.05), 64)
    ax.text(0.72, 1.14, "backbone", ha="center", fontsize=fs(8))
    arrow(ax, (1.32, 2.05), (1.92, 2.05))

    box(ax, (1.95, 1.62), 1.30, 0.86, "ProteinMPNN", "decoder as policy")
    arrow(ax, (3.28, 2.05), (3.85, 2.05))

    # B sequences, undifferentiated.
    for j in range(4):
        bar(ax, 3.90, 2.46 - 0.21 * j, 1.25, 0.14, [GREY] * K,
            edge=INK['ghost'])
    ax.text(4.52, 1.44, "$B$ sequences", ha="center", fontsize=fs(8))
    arrow(ax, (5.22, 2.05), (5.78, 2.05))

    # Cut into k fragments: the same bars, now coloured by slot.
    for j in range(4):
        bar(ax, 5.83, 2.46 - 0.21 * j, 1.25, 0.14, FRAGMENT)
    ax.text(6.45, 1.44, f"cut into $k={K}$", ha="center", fontsize=fs(8))
    arrow(ax, (7.15, 2.05), (7.70, 2.05))

    # Recombinants: one fragment per slot, drawn from different parents.
    for j in range(4):
        bar(ax, 7.75, 2.46 - 0.21 * j, 1.25, 0.14, FRAGMENT,
            shades=rng.uniform(0.45, 1.0, K))
    ax.text(8.37, 1.44, "recombinants", ha="center", fontsize=fs(8))

    box(ax, (7.72, 0.42), 1.32, 0.68, "fold & score",
        "best-of-5, motif RMSD")
    arrow(ax, (8.38, 1.30), (8.38, 1.12))

    # The feedback edge: this is the whole method.
    arrow(ax, (7.70, 0.76), (2.60, 0.76), color=PALETTE["blue"], rad=0.0,
          lw=1.6)
    arrow(ax, (2.60, 0.76), (2.60, 1.60), color=PALETTE["blue"], lw=1.6)
    # Black, like every other label. The arrow it names stays blue, so the
    # association is carried by adjacency rather than by matching ink.
    # Centred at 4.6, not 5.15. At page-size type this string is ~4.1 x-units
    # wide, and centred on the arrow's midpoint it ran under the fold & score
    # box at x=7.72.
    ax.text(4.6, 0.88, "GRPO update: each fragment scored by its "
                       "recombinants", ha="center", fontsize=fs(7))
    # Right-aligned to the panel edge: centred under the box it overflowed
    # the axes entirely.
    ax.text(10.0, 0.16, r"$\rho=0$ returns the parent, so this rate is not a "
                        "retention", ha="right", fontsize=fs(6.4),
            style="italic")






def main():
    rng = np.random.default_rng(3)
    plt.rcParams["text.usetex"] = False
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 1.95))
    ax_a = fig.add_subplot(111)
    panel_loop(ax_a, rng)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    x = fig.transFigure.inverted().transform(
        (ax_a.get_tightbbox(rend).x0, 0))[0]
    ax_a.text(x, 1.02, "a", transform=blended_transform_factory(
        fig.transFigure, ax_a.transAxes), fontsize=PANEL_LABEL_PT, fontweight="bold",
        va="top", ha="left")
    save(fig, "concept_loop", dpi=900)


if __name__ == "__main__":
    main()
