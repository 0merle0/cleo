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
from palette import FRAGMENT, INK, PALETTE, SEQUENTIAL  # noqa: E402

PNG = HERE / "structures" / "png"
K = 4
L_AA = 247                      # 1TIM chain A, the subject of panel (a)
BP_PER_AA = 3
COST_PER_BP = 0.07              # Twist eBlock, approx
MIN_BP = 300                    # eBlock minimum billed length
PROT_AA = 200                   # the enzyme being priced
GREY = INK["muted"]


def gene_cost(n_aa=PROT_AA):
    """-> cost of one full-length gene. 200 aa = 600 bp = $42."""
    return n_aa * BP_PER_AA * COST_PER_BP


def library_cost(S, k, n_aa=PROT_AA):
    """-> cost of a library of S sequences built from k slots.

    Equal numbers of variants per slot, so S = P^k, P = S^(1/k), and the order
    is k*P fragments. The floor is what makes this non-trivial: an eBlock is
    billed at MIN_BP even when the fragment is shorter, so cutting finer stops
    buying shorter DNA and only buys MORE pieces of it. At 200 aa the floor
    binds from k=2 on -- 600/k bp is exactly 300 at k=2 and below it after --
    so beyond k=2 the per-fragment price is flat and the k*P factor dominates.

    An earlier version priced the library as 3*L*P bases with no floor, which
    made every k look equally cheap per part: it understated k=4 by 2x and k=8
    by 4x, and implied cutting finer was always cheaper. It is not.
    """
    bp = np.maximum(n_aa * BP_PER_AA / k, MIN_BP)
    return k * S ** (1 / k) * bp * COST_PER_BP


def money(v):
    """-> a price as it would be written, not in scientific notation."""
    for cut, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if v >= cut:
            x = v / cut
            return f"\\${x:.0f}{suf}" if x >= 10 else f"\\${x:.1f}{suf}"
    return f"\\${v:.0f}"


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


def place_raw(ax, name, xy, zoom):
    """Place a structure PNG UNTRIMMED, so every image keeps one frame.

    `place` crops to the ink, which is right for a single structure and fatal
    for the exploded panel: cropping rescales each fragment to fill its own
    box, so they stop sharing a camera and stop sharing a size. Untrimmed, a
    fragment sits at its true position inside a mostly transparent frame, and
    offsetting the frame slides the fragment outward by exactly that much.
    """
    im = mpimg.imread(PNG / f"{name}.png")
    ax.add_artist(AnnotationBbox(OffsetImage(im, zoom=zoom), xy,
                                 frameon=False, box_alignment=(0.5, 0.5),
                                 annotation_clip=False))


def grid(ax):
    ax.grid(True, color=INK["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- panel (a)
def mark_level(ax, S, series, label):
    """Draw one library size as a reference line with a price on every arm.

    Only 10^12 is drawn. A second level at 10^6 was tried and removed: it
    doubled the labels for a ratio the single level already states, and the
    panel got busy enough that the arms stopped being the thing you see.

    Label placement stays parameterised by level because that is where the
    difficulty is. The arms are not evenly spaced in cost, and how close the
    two cheapest sit depends on the level -- 16x apart at 10^12 but only 2.8x
    at 10^6, far closer than a label is wide. Prices therefore alternate ABOVE
    and BELOW the line rather than being staggered in height, which separates
    neighbours whatever the spacing, and the dearest arm labels inward so it
    cannot run off the axis.
    """
    ax.axhline(S, color=GREY, lw=1, ls=":", zorder=1)
    # Caption at the LEFT end, above the line. The first price placed above
    # the line is the second-cheapest arm, which is decades to the right, so
    # this corner is empty. (It was on the right while a second level was
    # drawn, where it crowded the dearest arm's price.)
    ax.text(1.4e2, S * 1.6, label, ha="left", fontsize=7.5,
            color=PALETTE["gray"])
    marks = sorted(((float(library_cost(np.array(S), k)), c)
                    for _, k, c in series), key=lambda t: t[0])
    for i, (x, c) in enumerate(marks):
        ax.plot([x], [S], "o", ms=6.5, mfc=c, mec="white", mew=1.4, zorder=4)
        up = i % 2 == 1
        y, va = (S * 3.2, "bottom") if up else (S * 0.32, "top")
        if i == len(marks) - 1:
            xt, ha = x * 0.5, "right"
        else:
            xt, ha = x * 1.6, "left"
        ax.annotate(money(x), xy=(x, S), xytext=(xt, y), fontsize=8.5,
                    color=c, ha=ha, va=va, zorder=5)


def panel_cost(ax):
    """What a given spend buys, as a library size.

    Axes are cost on x and library size on y: the question is what you GET for
    what you PAY, so the thing bought belongs on the dependent axis. Plotted
    the other way the four arms converge at the left and their cost labels
    stack on top of each other; this way the reference line at 10^12 variants
    is horizontal and the four prices spread along it.

    One hue, dark to light, because k is ordered -- whole genes are k=1 and
    belong on the same ramp, not in a colour of their own. k=8 sits on the
    palette's categorical blue exactly.
    """
    cost = np.logspace(2, 15, 400)
    series = [("whole gene", 1, SEQUENTIAL[5]),
              ("$k=2$", 2, SEQUENTIAL[4]),
              ("$k=4$", 4, SEQUENTIAL[3]),
              ("$k=8$", 8, SEQUENTIAL[2])]
    for lab, k, c in series:
        bp = max(PROT_AA * BP_PER_AA / k, MIN_BP)
        S = (cost / (k * bp * COST_PER_BP)) ** k
        ax.loglog(cost, S, "-", color=c, lw=2.2, label=lab, zorder=3)
    mark_level(ax, 1e12, series, r"$10^{12}$ variants")
    ax.set_xlim(1e2, 1e15)
    ax.set_ylim(10, 1e16)
    ax.set_xlabel("synthesis cost (\\$)")
    ax.set_ylabel("library size, distinct sequences")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------- panel (c)
def panel_object(ax):
    """The fold itself, cut into k fragments and coloured by slot.

    Just the assembled structure. Two richer arrangements were built and
    dropped: an exploded view with the fold dimmed at the centre and the
    fragments pushed out around it, and a two-row version with the fold above
    and the pieces below. Both showed the same thing the colours already show,
    and both cost most of the panel to say it.

    render.py still emits the extra views (`dim`, `iso1`..`isoK`) -- they are
    cheap, they share this camera, and either arrangement is a few lines away
    if a later figure wants one.

    2FX2 rather than the PETase this project optimises: Figure 1 is about what
    a combinatorial library IS, and using the target would imply the idea is
    specific to it. Chosen by measuring secondary structure across a dozen
    candidates (39.5% helix, 32.7% strand), and at 147 aa its k=4 fragments
    are 37 residues -- inside the 20-50 band the method recommends.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    place(ax, "whole", (0.5, 0.5), 0.082)


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
        bar(ax, 5.83, 2.46 - 0.21 * j, 1.25, 0.14, FRAGMENT)
    ax.text(6.45, 1.44, f"cut into $k={K}$", ha="center", fontsize=8)
    arrow(ax, (7.15, 2.05), (7.70, 2.05))

    # Recombinants: one fragment per slot, drawn from different parents.
    for j in range(4):
        bar(ax, 7.75, 2.46 - 0.21 * j, 1.25, 0.14, FRAGMENT,
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
