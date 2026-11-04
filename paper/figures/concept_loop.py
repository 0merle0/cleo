#!/usr/bin/env python3
"""Figure 2, panel (a) --- the post-training loop.

Moved out of Figure 1. Figure 1 is the conceptual argument: what a
combinatorial library is, why its cost scales the way it does, and why a
filtered one searches better than a mutagenesis cloud or an unfiltered
assembly. None of that depends on how the policy is trained. The loop is the
METHOD, it is what Figure 2's results are measurements of, and it belongs at
the head of that figure rather than as a fourth panel under an argument it
plays no part in.

Two things this states that otherwise live only in CLAUDE.md: the fold is
best-of-5, which is the AME benchmark's own unit; and the recombination step
includes the parent itself (the rho=0 rotation), which is why an in-loop
chimera pass rate is NOT comparable to a T2 retention number.

    uv run python paper/figures/concept_loop.py
"""

import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figio import save  # noqa: E402
from palette import (FRAGMENT, INK, PAGE_WIDTH_IN, PALETTE,  # noqa: E402
                     SEQUENTIAL, fs)

PNG = HERE / "structures" / "png"
K = 4
GREY = INK["muted"]


def trimmed(name):
    """-> RGBA array cropped to its non-transparent bounding box."""
    im = mpimg.imread(PNG / f"{name}.png")
    if im.shape[2] == 4:
        ys, xs = np.nonzero(im[:, :, 3] > 0.02)
        if len(ys):
            im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return im


def place(ax, name, xy, width_pts):
    """Draw a structure PNG `width_pts` wide, whatever its pixel size.

    Sized in POINTS, not by OffsetImage's zoom. zoom is per-pixel, so a figure
    written against it silently rescales every structure whenever the render
    resolution changes.
    """
    im = trimmed(name)
    ax.add_artist(AnnotationBbox(
        OffsetImage(im, zoom=width_pts / im.shape[1]), xy,
        frameon=False, box_alignment=(0.5, 0.5)))


# ---------------------------------------------------------------- panel (a)
def bar(ax, x, y, w, h, colors, shades=None, lw=0.7, edge=None):
    """A sequence drawn as k abutting blocks, optionally shade-varied."""
    n = len(colors)
    for i, c in enumerate(colors):
        a = x + w * i / n
        f = 1.0 if shades is None else shades[i]
        # Soft fill, saturated stroke. The soft palette sits in the 6-8 CVD
        # band and is only legal with a second cue -- the stroke is that cue.
        ax.add_patch(FancyBboxPatch(
            (a + 0.004, y), w / n - 0.008, h,
            boxstyle="round,pad=0,rounding_size=0.008",
            fc=_shade(c, f),
            ec=edge if edge else FRAGMENT[i % len(FRAGMENT)], lw=0.7,
            zorder=3))


def _shade(hex_c, f):
    """Lighten toward white by (1-f); f=1 is the base colour."""
    r, g, b = (int(hex_c.lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
    return tuple(v + (1 - v) * (1 - f) for v in (r, g, b))


def box(ax, xy, w, h, label, sub=None, fc="white", title_size=8.5):
    ax.add_patch(FancyBboxPatch(xy, w, h,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                fc=fc, ec=PALETTE["gray"], lw=1.1, zorder=3))
    ax.text(xy[0] + w / 2, xy[1] + h * (0.62 if sub else 0.5), label,
            ha="center", va="center", fontsize=fs(title_size), zorder=4)
    if sub:
        ax.text(xy[0] + w / 2, xy[1] + h * 0.26, sub, ha="center", va="center",
                fontsize=fs(6.8), zorder=4)


def arrow(ax, p0, p1, color=None, rad=0.0, lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11,
        color=color or PALETTE["gray"], lw=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=2,
        shrinkA=2, shrinkB=2))


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





# ---------------------------------------------------------------- panel (b)
# Colours for the three measured variants. One hue, three steps -- the
# variants are not categories, they are three draws from one library, and the
# same three steps key the structures, the progress curves and the points.
VAR_C = [SEQUENTIAL[1], SEQUENTIAL[3], SEQUENTIAL[5]]


def panel_lab(ax, rng):
    """The experimental loop: library -> select -> assemble -> assay -> select.

    The counterpart of panel (a). Panel (a) closes in silico against a folding
    oracle and produces a library of parts; this one closes in the lab against
    measured activity and chooses which parts to use next.

    The returning arrow lands on the SELECTOR, not on the library. That is the
    substantive claim of the panel and it is also what the code does: the
    library is fixed once it is ordered, and what the measurements change is
    the model that picks parts out of it. An arrow drawn back to the library
    would say the parts are resynthesised each round, which would throw away
    the cost argument Figure 1 makes.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    # Horizontal budget. The axes is ~440 pt wide, so one x-unit is ~44 pt and
    # a 7 pt label of n characters is roughly n*3.5/44 units. Every element
    # below is placed against that, which is the step that was missing when
    # this panel was first laid out against an 11-inch canvas.
    #
    # --- the fragment pool ------------------------------------------------
    # Drawn as k slot-columns of tinted parts rather than as whole sequences,
    # because that is what is physically in the tube: one pool per slot, and a
    # design is one part taken from each.
    x0, y0, bw, bh = 0.30, 1.02, 0.21, 0.145
    n_per_slot = 5
    for i in range(K):
        for j in range(n_per_slot):
            ax.add_patch(FancyBboxPatch(
                (x0 + i * (bw + 0.07), y0 + j * (bh + 0.07)), bw, bh,
                boxstyle="round,pad=0,rounding_size=0.03",
                fc=_shade(FRAGMENT[i], rng.uniform(0.40, 1.0)),
                ec=FRAGMENT[i], lw=0.5, zorder=3))
    lib_w = K * bw + (K - 1) * 0.07
    lib_cx = x0 + lib_w / 2
    ax.text(lib_cx, y0 + n_per_slot * (bh + 0.07) + 0.07, "fragment library",
            ha="center", fontsize=fs(7))
    ax.text(lib_cx, 0.80, "one pool per slot", ha="center", fontsize=fs(6.5),
            color=PALETTE["gray"])

    arrow(ax, (x0 + lib_w + 0.08, 1.61), (1.95, 1.61))

    # --- the selector -----------------------------------------------------
    sel_x, sel_w = 2.00, 1.80
    sel_cx = sel_x + sel_w / 2
    box(ax, (sel_x, 1.26), sel_w, 0.70, "fragment selector",
        "surrogate ensemble", title_size=8)
    arrow(ax, (sel_x + sel_w, 1.61), (4.25, 1.61))

    # --- the assembled variants -------------------------------------------
    # One part per slot, so every design is the same fold in a different
    # combination. Rendered from one camera (render.py --variants) so the
    # reader sees the colour change and not a pose change. 30 pt at 0.90-unit
    # spacing leaves a visible gap between them -- at 38 pt they touched.
    for n in range(3):
        cx = 4.65 + n * 0.90
        place(ax, f"var{n+1}", (cx, 1.72), 30)
        ax.add_patch(FancyBboxPatch((cx - 0.19, 1.22), 0.38, 0.055,
                                    boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=VAR_C[n], ec="none", zorder=4))
    ax.text(5.55, 0.99, "assembled variants", ha="center", fontsize=fs(7))
    ax.text(5.55, 0.76, "one part per slot", ha="center", fontsize=fs(6.5),
            color=PALETTE["gray"])

    arrow(ax, (6.85, 1.61), (7.26, 1.61))

    # --- the activity readout ---------------------------------------------
    # Saturating progress curves, because an enzyme assay runs toward
    # substrate depletion. Straight lines would say initial rates were the
    # measurement, which is a different experiment. Each curve carries its
    # variant's colour, so structure and trace are linked without a legend.
    axi = ax.inset_axes([7.32, 1.20, 2.20, 1.10], transform=ax.transData)
    t = np.linspace(0, 1, 160)
    for k_rate, c in zip((1.1, 2.4, 5.2), VAR_C):
        axi.plot(t, 1 - np.exp(-k_rate * t), color=c, lw=1.5, zorder=3)
    axi.set_xlim(0, 1)
    axi.set_ylim(0, 1.05)
    axi.set_xticks([])
    axi.set_yticks([])
    axi.set_xlabel("time", fontsize=fs(6.5), labelpad=1.5)
    axi.set_ylabel("activity", fontsize=fs(6.5), labelpad=1.5)
    for sp in ("top", "right"):
        axi.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axi.spines[sp].set_color(PALETTE["gray"])
        axi.spines[sp].set_linewidth(0.8)
    # Below the inset's own x-label, which is what it collided with before.
    ax.text(8.42, 0.76, "measured activity", ha="center", fontsize=fs(7))

    # --- back to the selector ---------------------------------------------
    # Dropped from the plot's right edge, not from under its label: at x=8.42
    # the descending arrow ran through "measured activity".
    ret_x = 9.45
    arrow(ax, (ret_x, 1.12), (ret_x, 0.40), color=PALETTE["blue"], lw=1.5)
    arrow(ax, (ret_x, 0.40), (sel_cx, 0.40), color=PALETTE["blue"], lw=1.5)
    arrow(ax, (sel_cx, 0.40), (sel_cx, 1.24), color=PALETTE["blue"], lw=1.5)
    ax.text((sel_cx + ret_x) / 2, 0.19,
            "measured activity retrains the selector", ha="center",
            fontsize=fs(6.6))


def main():
    rng = np.random.default_rng(3)
    plt.rcParams["text.usetex"] = False
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 4.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.12], hspace=0.16)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    panel_loop(ax_a, rng)
    panel_lab(ax_b, rng)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    x = min(inv.transform((a.get_tightbbox(rend).x0, 0))[0]
            for a in (ax_a, ax_b))
    for ax, lab in ((ax_a, "a"), (ax_b, "b")):
        ax.text(x, 1.02, lab, transform=blended_transform_factory(
            fig.transFigure, ax.transAxes), fontsize=12, fontweight="bold",
            va="top", ha="left")
    save(fig, "concept_loop", dpi=900)


if __name__ == "__main__":
    main()
