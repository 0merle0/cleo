#!/usr/bin/env python3
"""Figure 1 --- what a combinatorial library is, and how the policy is trained.

The opening figure has to do two jobs that the paper currently splits across
three scattered cost panels and no schematic at all:

  the argument   ordering N unique genes costs N x $42; ordering the SAME N as
                 a combinatorial library costs N^(1/k) x $42, because you buy
                 parts and assemble them. That is the whole reason to build
                 libraries this way, and it is arithmetic -- true before any
                 experiment is run.
  the method     a backbone goes into ProteinMPNN, the decoder emits sequences,
                 the sequences are cut into k fragments, the fragments are
                 recombined, the recombinants are folded and scored, and each
                 fragment's score is pushed back into the decoder. Nothing in
                 the paper draws this, and every later number is unreadable
                 without it.

Panel (b) deliberately ends on a question rather than a payoff. Library size
grows as P^k, so the obvious reading is "use a larger k" -- and the answer, that
recombinants stop folding below 11-residue fragments, is the k sweep later in
the paper. The forward pointer is the point of the panel.

Two things the schematic states that otherwise live only in CLAUDE.md:
the fold is best-of-5, which is the AME benchmark's own unit; and the
recombination step includes the parent itself (the rho=0 rotation), which is
why an in-loop chimera pass rate is NOT comparable to a T2 retention number.

    uv run python paper/figures/concept_library.py
"""

import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figio import save  # noqa: E402
from palette import FRAGMENT, FRAGMENT_SOFT, INK, PALETTE, SEQUENTIAL  # noqa: E402

PNG = HERE / "structures" / "png"
K = 4
L_AA = 247                      # 1TIM chain A, the subject of panel (a)
BP_PER_AA = 3
COST_PER_BP = 0.07              # Twist eBlock, approx; $42 for a 200-aa gene
GREY = INK["muted"]


def gene_cost(n_aa=200):
    return n_aa * BP_PER_AA * COST_PER_BP


def trimmed(name):
    """-> RGBA array cropped to its non-transparent bounding box."""
    im = mpimg.imread(PNG / f"{name}.png")
    if im.shape[2] == 4:
        ys, xs = np.nonzero(im[:, :, 3] > 0.02)
        if len(ys):
            im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return im


def place(ax, name, xy, zoom):
    ax.add_artist(AnnotationBbox(OffsetImage(trimmed(name), zoom=zoom), xy,
                                 frameon=False, box_alignment=(0.5, 0.5)))


def grid(ax):
    ax.grid(True, color=INK["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- panel (a)
def panel_cost(ax):
    """What N sequences cost, bought whole against bought as parts.

    One hue, dark to light, because k is an ordered quantity -- whole genes are
    k=1 and belong on the same ramp as the rest, not in a separate colour.
    """
    S = np.logspace(1, 12, 300)
    unit = gene_cost()
    series = [("whole gene", 1, SEQUENTIAL[5]),
              ("$k=2$", 2, SEQUENTIAL[4]),
              ("$k=4$", 4, SEQUENTIAL[3]),
              ("$k=8$", 8, SEQUENTIAL[2])]
    for lab, k, c in series:
        ax.loglog(S, S ** (1 / k) * unit, "-", color=c, lw=2.2, label=lab,
                  zorder=3)
    ax.axvline(1e12, color=GREY, lw=1, ls=":", zorder=1)
    for k, c, lab, dy in ((1, SEQUENTIAL[5], r"\$42T", 3.2),
                          (8, SEQUENTIAL[2], r"\$1.3k", 0.26)):
        y = 1e12 ** (1 / k) * unit
        ax.plot([1e12], [y], "o", ms=7, mfc=c, mec="white", mew=1.4, zorder=4)
        ax.annotate(lab, xy=(1e12, y), xytext=(1e12 * 0.55, y * dy),
                    fontsize=9.5, color=c, ha="right", zorder=5)
    ax.set_xlim(10, 3e13)
    ax.set_ylim(40, 1e16)
    ax.text(1e12, 2e15, r"$10^{12}$ variants", ha="center", fontsize=7.5,
            color=PALETTE["gray"])
    ax.set_xlabel("library size, distinct sequences")
    ax.set_ylabel("synthesis cost (\\$)")
    ax.set_title("buying parts, not products", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    grid(ax)


# ---------------------------------------------------------------- panel (c)
def panel_object(ax):
    """The thing itself: one fold, cut into k equal fragments.

    1TIM rather than the PETase this project optimises -- the idea is not
    specific to the target, and showing the target here would suggest it is.
    The residue ranges that used to sit under this are gone: they are an
    implementation detail of one protein and were competing with the fold.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    place(ax, "whole", (0.5, 0.50), 0.125)
    ax.set_title(f"one fold, {K} parts", fontsize=10)


# ---------------------------------------------------------------- panel (d)
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


def box(ax, xy, w, h, label, sub=None, fc="white"):
    ax.add_patch(FancyBboxPatch(xy, w, h,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                fc=fc, ec=PALETTE["gray"], lw=1.1, zorder=3))
    ax.text(xy[0] + w / 2, xy[1] + h * (0.62 if sub else 0.5), label,
            ha="center", va="center", fontsize=8.5, zorder=4)
    if sub:
        ax.text(xy[0] + w / 2, xy[1] + h * 0.26, sub, ha="center", va="center",
                fontsize=6.8, color=PALETTE["gray"], zorder=4)


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

    place(ax, "grey", (0.72, 2.05), 0.040)
    ax.text(0.72, 1.14, "backbone", ha="center", fontsize=8)
    arrow(ax, (1.32, 2.05), (1.92, 2.05))

    box(ax, (1.95, 1.62), 1.30, 0.86, "ProteinMPNN", "decoder as policy")
    arrow(ax, (3.28, 2.05), (3.85, 2.05))

    # B sequences, undifferentiated.
    for j in range(4):
        bar(ax, 3.90, 2.46 - 0.21 * j, 1.25, 0.14, [GREY] * K,
            edge=INK['ghost'])
    ax.text(4.52, 1.44, "$B$ sequences", ha="center", fontsize=8)
    arrow(ax, (5.22, 2.05), (5.78, 2.05))

    # Cut into k fragments: the same bars, now coloured by slot.
    for j in range(4):
        bar(ax, 5.83, 2.46 - 0.21 * j, 1.25, 0.14, FRAGMENT_SOFT)
    ax.text(6.45, 1.44, f"cut into $k={K}$", ha="center", fontsize=8)
    arrow(ax, (7.15, 2.05), (7.70, 2.05))

    # Recombinants: one fragment per slot, drawn from different parents.
    for j in range(4):
        bar(ax, 7.75, 2.46 - 0.21 * j, 1.25, 0.14, FRAGMENT_SOFT,
            shades=rng.uniform(0.45, 1.0, K))
    ax.text(8.37, 1.44, "recombinants", ha="center", fontsize=8)

    box(ax, (7.72, 0.42), 1.32, 0.68, "fold & score",
        "best-of-5, motif RMSD")
    arrow(ax, (8.38, 1.30), (8.38, 1.12))

    # The feedback edge: this is the whole method.
    arrow(ax, (7.70, 0.76), (2.60, 0.76), color=PALETTE["blue"], rad=0.0,
          lw=1.6)
    arrow(ax, (2.60, 0.76), (2.60, 1.60), color=PALETTE["blue"], lw=1.6)
    ax.text(5.15, 0.88, "GRPO update: each fragment scored by the "
                        "recombinants it appears in",
            ha="center", fontsize=7.6, color=PALETTE["blue"])
    ax.text(8.38, 0.22, r"$\rho=0$ returns the parent, so this rate is not a "
                        "retention", ha="center", fontsize=6.4,
            color=PALETTE["gray"], style="italic")
    ax.set_title("the policy is trained on how well its fragments recombine",
                 fontsize=10, loc="left")


def main():
    rng = np.random.default_rng(3)
    plt.rcParams["text.usetex"] = False
    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 0.78],
                          width_ratios=[1.06, 1.0], hspace=0.30, wspace=0.22)
    panel_object(fig.add_subplot(gs[0, 0]))
    panel_cost(fig.add_subplot(gs[0, 1]))
    panel_loop(fig.add_subplot(gs[1, :]), rng)
    for ax, lab in zip(fig.axes, "abc"):
        ax.text(-0.02, 1.12, f"\\textbf{{{lab}}}" if False else lab,
                transform=ax.transAxes, fontsize=12, fontweight="bold",
                va="top", ha="right")
    save(fig, "concept_library")


if __name__ == "__main__":
    main()
