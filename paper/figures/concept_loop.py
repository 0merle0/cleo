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
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from schematic import FRAGMENT, arrow, bar, box, place  # noqa: E402

PNG = HERE / "structures" / "png"
K = 4
GREY = INK["muted"]


def panel_loop(ax, rng):
    """How post-training works here: one fragment, many chimeras, one reward.

    This panel used to be a pipeline -- backbone, decoder, sequences, cut,
    recombine, fold, update -- which is the ORDER of operations and not the
    idea. Every box in it is standard; a reader who knows RLHF learns nothing
    from the arrows.

    The idea worth a panel is the credit assignment. ProteinMPNN is pretrained
    and emits whole sequences, so there is no gradient that says "this
    fragment recombines well". The method manufactures one: cut the batch into
    k slots, build chimeras by swapping slots between parents, fold each
    chimera, and then give every FRAGMENT the average outcome of the chimeras
    it appeared in. A fragment that carries its partners scores high however
    good its own parent was; a fragment that only works beside its original
    neighbours does not.

    That is why the picture is a grid and not a chain: the highlighted
    fragment occurs in three chimeras, two of which fold, so its reward is
    2/3 -- and that number, not the parent's score, is what the GRPO update
    sees.

    Pass/fail marks rather than invented numbers. The reward really is an
    average over chimera outcomes, so the panel can show the actual quantity
    instead of a plausible-looking decimal.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.1)
    ax.axis("off")

    # --- the decoder, pretrained and then post-trained ---------------------
    box(ax, (0.12, 1.52), 1.72, 0.74, "ProteinMPNN",
        "pretrained decoder", title_size=8)
    arrow(ax, (1.90, 1.89), (2.36, 1.89))

    # --- a batch of designs, cut into k slots ------------------------------
    for j in range(3):
        bar(ax, 2.45, 2.28 - 0.42 * j, 1.30, 0.30, FRAGMENT)
    ax.text(3.10, 1.12, f"cut into $k={K}$", ha="center", fontsize=fs(7))
    ax.text(3.10, 2.82, "$B$ designs", ha="center", fontsize=fs(7))
    arrow(ax, (3.83, 1.89), (4.29, 1.89))

    # --- chimeras: slots swapped between parents, each one folded ----------
    #
    # The highlighted slot-0 part is deliberately the SAME tint in rows 0, 2
    # and 3 and a different one in row 1, so the ringed cells are visibly one
    # fragment recurring rather than four unrelated blue blocks.
    cx, cw, rh = 4.38, 1.90, 0.30
    HILITE, OTHER = 1.0, 0.45
    rows = [
        ([HILITE, 0.55, 1.0, 0.45], True),
        ([OTHER, 1.0, 0.50, 1.0], True),
        ([HILITE, 1.0, 0.45, 0.95], False),
        ([HILITE, 0.45, 0.95, 0.55], True),
    ]
    ys = []
    for j, (shades, ok) in enumerate(rows):
        y = 2.34 - 0.42 * j
        ys.append(y)
        bar(ax, cx, y, cw, rh, FRAGMENT, shades=shades)
        # pass / fail, as the oracle pool returned it
        ax.plot(cx + cw + 0.22, y + rh / 2, "o", ms=5.5,
                mfc=PALETTE["blue"] if ok else "white",
                mec=PALETTE["blue"] if ok else INK["muted"], mew=1.2,
                zorder=5)
    ax.text(cx + cw / 2, 2.78, "chimeras", ha="center", fontsize=fs(7))
    # Above the dots, not below them: below, it sat in the band the credit
    # note and the GRPO label already share.
    ax.text(cx + cw + 0.22, 2.78, "folds", ha="center", fontsize=fs(6.5),
            color=PALETTE["gray"])

    # Ring the one fragment whose reward is being computed.
    sw = cw / K
    for j, (shades, _) in enumerate(rows):
        if shades[0] != HILITE:
            continue
        ax.add_patch(FancyBboxPatch(
            (cx - 0.035, ys[j] - 0.035), sw + 0.07, rh + 0.07,
            boxstyle="square,pad=0", fill=False, ec="black", lw=1.1,
            zorder=6))
    ax.text(cx + sw / 2, 0.92, "same fragment, 3 chimeras", ha="center",
            va="center", fontsize=fs(6.5), color=PALETTE["gray"])

    arrow(ax, (cx + cw + 0.52, 1.89), (cx + cw + 0.98, 1.89))

    # --- the credit assignment --------------------------------------------
    bx = cx + cw + 1.08
    box(ax, (bx, 1.46), 9.86 - bx, 0.86,
        r"fragment reward $=\,2/3$", "mean outcome of its chimeras",
        title_size=8)

    # --- back to the decoder ----------------------------------------------
    arrow(ax, (bx + (9.86 - bx) / 2, 1.42), (bx + (9.86 - bx) / 2, 0.46),
          color=PALETTE["blue"], lw=1.5)
    arrow(ax, (bx + (9.86 - bx) / 2, 0.46), (0.98, 0.46),
          color=PALETTE["blue"], lw=1.5)
    arrow(ax, (0.98, 0.46), (0.98, 1.48), color=PALETTE["blue"], lw=1.5)
    # BELOW the arrow, not above it. Above, it shared a band with the
    # "same fragment" note, and the two closed up whenever the panel got
    # shorter.
    ax.text(5.1, 0.28, "GRPO update, per-fragment advantage", ha="center",
            fontsize=fs(7), va="center")

    # The two facts a reader needs before reading any number that follows.
    ax.text(5.1, 0.04, "folded best-of-5 against a pool of RF3 / Boltz / "
                       "Protenix; AF3 is held out and scores every "
                       "reported number", ha="center", fontsize=fs(6.4),
            color=PALETTE["blue"])


def main():
    rng = np.random.default_rng(3)
    plt.rcParams["text.usetex"] = False
    # Same fixed margins as Figure 1, and the save below does not crop to
    # content, so both figures are exactly PAGE_WIDTH_IN wide and their panels
    # line up with each other on the page rather than each being cropped to
    # whatever its own widest element happened to be.
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 1.95))
    gs = fig.add_gridspec(1, 1, left=0.085, right=0.985,
                          top=0.90, bottom=0.03)
    ax_a = fig.add_subplot(gs[0])
    panel_loop(ax_a, rng)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    x = fig.transFigure.inverted().transform(
        (ax_a.get_tightbbox(rend).x0, 0))[0]
    ax_a.text(x, 1.02, "a", transform=blended_transform_factory(
        fig.transFigure, ax_a.transAxes), fontsize=PANEL_LABEL_PT, fontweight="bold",
        va="top", ha="left")
    save(fig, "concept_loop", dpi=900, tight=False)


if __name__ == "__main__":
    main()
