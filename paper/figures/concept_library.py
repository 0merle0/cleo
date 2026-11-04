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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figio import save  # noqa: E402
from palette import (INK, PAGE_WIDTH_IN, PALETTE,  # noqa: E402
                     PANEL_LABEL_PT, SEQUENTIAL, fs)
from schematic import (FRAGMENT, _shade, arrow, box,  # noqa: E402
                       place, trimmed)

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


def grid(ax):
    ax.grid(True, color=INK["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- panel (a)
#
# A toy fitness landscape, drawn to contrast three ways of spending the SAME
# number of ordered designs. Nothing here is measured -- it is an argument
# about search strategy, made before any data, and the panel should be read
# that way.
#
# Four optima of unequal height, over a rugged background. The parent design
# sits on a decent but not-best one, which is the situation every campaign
# actually starts from.
PEAKS = [
    # cx,   cy,    width, height
    (0.26, 0.32, 0.085, 0.78),   # the parent's own optimum -- good, not best
    (0.70, 0.74, 0.095, 1.00),   # the global optimum
    (0.79, 0.27, 0.078, 0.88),
    (0.31, 0.78, 0.070, 0.66),
]
START = (0.26, 0.32)
# Variants actually built and screened -- the SAME for all three panels, so
# they differ in WHERE the budget goes and not in how much of it there is.
# Deliberately not called "ordered": what gets ordered in the two
# combinatorial arms is FRAGMENTS, and the full-length variants are assembled
# from them. That distinction is the method, so the panel must not blur it.
N_VARIANTS = 36
DOT = "#1F2937"

# White through the house blue ramp, stopping at the mid step. The darkest two
# steps are deliberately unused: peaks have to stay light enough that the
# sampled designs drawn on top of them still read.
TERRAIN = LinearSegmentedColormap.from_list(
    "cleo_terrain", ["#FFFFFF", "#FFFFFF", "#DCEBFB", SEQUENTIAL[0],
                     SEQUENTIAL[1], SEQUENTIAL[2], SEQUENTIAL[3]])


def fitness(x, y):
    """-> the toy landscape: four gaussian optima plus gentle ruggedness."""
    z = sum(a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
            for cx, cy, s, a in PEAKS)
    return z + 0.02 * np.sin(4 * np.pi * x) * np.sin(3.5 * np.pi * y)


def _clip(p):
    return np.clip(p, 0.03, 0.97)


def draw_landscape(ax, title, axis_labels=False):
    n = 320
    g = np.linspace(0, 1, n)
    X, Y = np.meshgrid(g, g)
    Z = fitness(X, Y)
    ax.contourf(X, Y, Z, levels=12, cmap=TERRAIN, zorder=0)
    ax.contour(X, Y, Z, levels=12, colors="white", linewidths=0.5,
               alpha=0.85, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(INK["grid"])
        sp.set_linewidth(0.9)
    # Black, not the grey used for secondary labels elsewhere: these three
    # name the arms being contrasted and are the panel's primary text. The
    # third is not coloured either -- tinting only "this work" would say the
    # comparison was decided before the reader got to it.
    ax.set_title(title, fontsize=fs(9), pad=7, color="black")
    if axis_labels:
        ax.set_xlabel("Sequence space", fontsize=fs(8), labelpad=3)
        ax.set_ylabel("Sequence space", fontsize=fs(8), labelpad=3)


def draw_designs(ax, pts, ms=3.8):
    """The full-length variants that were built and screened."""
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=ms, mfc=DOT, mec="white", mew=0.55,
            ls="none", zorder=5)


def draw_start(ax):
    ax.plot(*START, marker="*", ms=12, mfc="white", mec=DOT, mew=1.1,
            ls="none", zorder=6)


def panel_epcr(ax, rng):
    """Mutagenesis walks: dense, but it never leaves the parent's basin."""
    draw_landscape(ax, "Error-prone PCR", axis_labels=True)
    pts = _clip(rng.normal(START, 0.055, size=(N_VARIANTS, 2)))
    draw_designs(ax, pts)
    draw_start(ax)


def panel_naive(ax, rng):
    """Combinatorial assembly with parts chosen without regard to fitness.

    Drawn as a PRODUCT SET rather than a cloud, because that is what assembly
    actually builds: pick 6 variants for block 1 and 6 for block 2 and you get
    their 36 combinations, not 36 free points. The variants are drawn
    uniformly -- naive means the parts were not selected for anything.

    The seed is chosen, and that is worth stating plainly: of the first 400,
    seed 50 is one where the uniform draws happen to leave the global optimum
    uncovered while still catching two others. That outcome is typical rather
    than engineered -- a coarse product set in a landscape of narrow optima
    misses most of them -- but the specific miss shown is a pick, and no
    reader should take it for a measurement.
    """
    draw_landscape(ax, "Naive combinatorial assembly", axis_labels=True)
    r = np.random.default_rng(50)
    xs = np.sort(r.uniform(0.05, 0.95, 6))
    ys = np.sort(r.uniform(0.05, 0.95, 6))
    pts = _clip(np.array([(x, y) for x in xs for y in ys], float)
                + rng.normal(0, 0.012, (36, 2)))
    draw_designs(ax, pts)
    draw_start(ax)
    # The ring and "missed" callout that used to mark PEAKS[1] are gone. The
    # empty optimum reads on its own against the two that carry points, and the
    # annotation was the only red in the row -- a colour that elsewhere in this
    # figure means a fragment.


def panel_ours(ax, rng):
    """Generate the same combinatorial reach, then filter hard before building.

    The point of the panel is that the filter IS the search. A policy-designed
    library spans the space as widely as the naive one -- the faint open
    candidates -- but only what survives folding and scoring stays in the
    library, and those survivors sit on the optima. The filled points number
    N_VARIANTS, identical to the other two panels.

    The legend says "assembled from library" and not "ordered" on purpose.
    What is ordered is FRAGMENTS -- each full-length variant is built by
    assembling one fragment per slot, which is exactly why the cost argument
    in panel (c) holds. A legend reading "ordered" would quietly describe this
    as a catalogue of full genes, which is the thing the method exists to
    avoid.
    """
    draw_landscape(ax, "This work", axis_labels=True)
    xs = rng.uniform(0.05, 0.95, 14)
    ys = rng.uniform(0.05, 0.95, 14)
    cand = _clip(np.array([(x, y) for x in xs for y in ys], float)
                 + rng.normal(0, 0.010, (196, 2)))
    keep = np.argsort(fitness(cand[:, 0], cand[:, 1]))[-N_VARIANTS:]
    mask = np.zeros(len(cand), bool)
    mask[keep] = True
    ax.plot(cand[~mask, 0], cand[~mask, 1], "o", ms=2.6, mfc="none",
            mec=INK["ghost"], mew=0.7, ls="none", zorder=4)
    draw_designs(ax, cand[mask])
    draw_start(ax)


def fitness_key(ax):
    """A direction, not a scale -- the landscape is illustrative."""
    cb = ax.inset_axes([1.105, 0.16, 0.032, 0.78])
    g = np.linspace(0, 1, 256).reshape(-1, 1)
    cb.imshow(g, aspect="auto", origin="lower", cmap=TERRAIN,
              extent=(0, 1, 0, 1))
    cb.set_xticks([])
    cb.set_yticks([])
    for sp in cb.spines.values():
        sp.set_color(INK["grid"])
    cb.text(1.5, 1.0, "high", ha="left", va="center", fontsize=fs(6.6),
            transform=cb.transAxes)
    cb.text(1.5, 0.0, "low", ha="left", va="center", fontsize=fs(6.6),
            transform=cb.transAxes)
    cb.text(-0.9, 0.5, "Fitness", rotation=90, ha="center", va="center",
            fontsize=fs(7.5), transform=cb.transAxes)


def landscape_legend(ax):
    """One legend for all three landscapes, in a row, in its OWN gridspec cell.

    It was two legends inside two panels: the star keyed in the first, the
    point styles in the third. Both sat on terrain and had to be squeezed into
    whichever corner the samples left free, which is why they ended up at
    different corners from each other.

    Pulled out, it needs a cell of its own rather than the gap between rows.
    Anchored into that gap it kept landing beside the NEXT row's panel letter,
    which also lives there -- close enough to read as one band, even when the
    boxes did not technically overlap. A dedicated row cannot do that.
    """
    from matplotlib.lines import Line2D
    ax.axis("off")
    handles = [
        Line2D([], [], marker="*", ms=10, mfc="white", mec=DOT, mew=1.1,
               ls="none", label="parent design"),
        Line2D([], [], marker="o", ms=2.6, mfc="none", mec=INK["ghost"],
               mew=0.7, ls="none", label="filtered out"),
        Line2D([], [], marker="o", ms=3.8, mfc=DOT, mec="white", mew=0.55,
               ls="none", label="assembled from library"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
              fontsize=fs(7), handletextpad=0.4, columnspacing=3.0)


# ---------------------------------------------------------------- panel (b)
def crossing_costs(S, series):
    """-> {label: cost} where each arm reaches a library of `S` sequences."""
    return {lab: float(library_cost(np.array(S), k)) for lab, k, _ in series}


def mark_level(ax, S, series):
    """Draw the reference level, a dot per arm, and each arm's price beside it.

    Placement is per rank in cost, as (y-factor, x-factor, ha). The rule is
    not "alternate above and below" -- that was the old rule and it fails
    here, because the three cheap arms are NEAR-VERTICAL at this level. A
    steep line stays within a few points of the same x over a wide range of
    y, so moving a label up or down does not move it off the line; only
    moving it sideways does.

    So every label sits just off the level line and is pushed sideways into
    the gap beside its own arm, with the side chosen per arm:

        k=8   left   nothing is left of the cheapest arm but the axis
        k=4   right  its left is where k=8 sits, 1.2 decades away
        k=2   left   six decades of clear space
        whole left   same, and it cannot run off the right edge

    The gaps are what the panel's extra width bought. At the old width the
    k=8/k=4 gap was 20 pt and the label 24 pt, which is why these lived in
    the legend for a while.
    """
    ax.axhline(S, color=GREY, lw=1, ls=":", zorder=1,
               label=r"$10^{12}$ variants")
    # The side follows from the direction the curve moves, which is the part
    # that was got wrong first: every arm RISES, so dropping below the level
    # line puts the curve to your LEFT and lifting above it puts the curve to
    # your RIGHT. A label therefore goes opposite to the way it was offset --
    # below-line labels to the right of the arm, above-line labels to its
    # left. Offsetting a below-line label leftward, which looks symmetric,
    # walks it straight into its own curve.
    PLACE = [(2.2, 0.80, "right"),    # k=8    above, so left
             (0.45, 1.25, "left"),    # k=4    below, so right
             (0.45, 1.25, "left"),    # k=2    below, so right
             (2.2, 0.80, "right")]    # whole  above, so left (and cannot
                                      #        run off the right edge)
    marks = sorted(((float(library_cost(np.array(S), k)), c)
                    for _, k, c in series), key=lambda t: t[0])
    for i, (x, c) in enumerate(marks):
        ax.plot([x], [S], "o", ms=6.5, mfc=c, mec="white", mew=1.4, zorder=4)
        yf, xf, ha = PLACE[i]
        ax.annotate(money(x), xy=(x, S), xytext=(x * xf, S * yf),
                    fontsize=fs(7.5), color=c, ha=ha, va="center", zorder=5)


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
    LEVEL = 1e12
    for lab, k, c in series:
        bp = max(PROT_AA * BP_PER_AA / k, MIN_BP)
        S = (cost / (k * bp * COST_PER_BP)) ** k
        ax.loglog(cost, S, "-", color=c, lw=2.2, zorder=3,
                  label=lab)
    mark_level(ax, LEVEL, series)
    ax.set_xlim(1e2, 1e15)
    ax.set_ylim(10, 1e16)
    ax.set_xlabel("Synthesis cost (\\$)")
    ax.set_ylabel("Library size (unique sequences)")
    # Opaque, because the whole-gene arm runs straight through where this
    # sits. A frameless legend over a data line is worse than a framed one.
    # Tight enough to sit entirely UNDER the whole-gene arm, which is the
    # only line that reaches this corner. Opaque anyway, as a backstop.
    ax.legend(fontsize=fs(7), loc="lower right", labelspacing=0.22,
              handlelength=1.4, handletextpad=0.5, frameon=True,
              facecolor="white", edgecolor="none", framealpha=1.0,
              borderpad=0.3, borderaxespad=0.4).set_zorder(6)
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
    place(ax, "whole", (0.5, 0.5), 135)


# ---------------------------------------------------------------- panel (d)
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
    # 2.35, not 3.2. The schematic only ever reaches y=2.17, so a third of the
    # axes box was empty -- and since the panel letter is placed at the box's
    # top, "d" floated in that emptiness, level with panel (b)'s x-axis and
    # reading as though it belonged there. The row's height ratio comes down
    # with it so the drawing keeps its proportions.
    ax.set_ylim(0, 2.35)
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
                boxstyle="square,pad=0",
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
                                    boxstyle="square,pad=0",
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
    # Authored at page width, so a point here is a point on the page and the
    # MIN_FONT floor is real. See palette.PAGE_WIDTH_IN.
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 7.3))
    # Row 0 is given enough height for the landscapes to reach their CELL
    # WIDTH. They are square (aspect equal), so matplotlib shrinks them to a
    # square inside each cell and centres it: while the row was short they
    # were height-limited at 1.12 in inside cells 1.55 in wide, and every
    # spare inch of the row went to margin instead of to the plots.
    outer = fig.add_gridspec(3, 1, height_ratios=[1.90, 1.40, 0.76],
                             hspace=0.34)
    # The legend belongs to the landscapes, so it lives INSIDE their row --
    # tight under the three panels, with the full row gap below it.
    #
    # It was a row of its own between row 0 and row 1, and before that it was
    # anchored into that gap. Both put it midway between the landscapes it
    # explains and panels b/c, which it has nothing to do with, and the gap is
    # also where panel b's letter sits. Pulling it up into row 0 settles the
    # question of which panels it belongs to and clears the gap in one move.
    top = outer[0].subgridspec(2, 1, height_ratios=[1.0, 0.11], hspace=0.30)
    # The three landscapes share a row and a letter: they are one argument
    # seen three ways, not three findings, and lettering them a/b/c would
    # promise the reader three separate results.
    land = top[0].subgridspec(1, 3, wspace=0.11)
    ax_e = fig.add_subplot(land[0, 0])
    ax_n = fig.add_subplot(land[0, 1])
    ax_o = fig.add_subplot(land[0, 2])
    panel_epcr(ax_e, rng)
    panel_naive(ax_n, rng)
    panel_ours(ax_o, rng)
    fitness_key(ax_o)
    landscape_legend(fig.add_subplot(top[1]))

    # Cost on the LEFT and wider than the fold beside it. The prices are drawn
    # at their crossings again rather than carried in the legend, and that
    # only works if the arms are far enough apart in x to take a label
    # between them -- which is bought with axes width, not with font size.
    mid = outer[1].subgridspec(1, 2, width_ratios=[1.45, 0.72], wspace=0.20)
    ax_cost = fig.add_subplot(mid[0, 0])
    ax_obj = fig.add_subplot(mid[0, 1])
    ax_lab = fig.add_subplot(outer[2])
    panel_object(ax_obj)
    panel_cost(ax_cost)
    panel_lab(ax_lab, rng)

    # Panel letters are placed in FIGURE x, not axes x.
    #
    # The old placement was transAxes -0.02, i.e. 2% of each axes' OWN width to
    # the left of it. The four axes have four different widths and four
    # different left edges, so that put the four letters at four different
    # absolute positions -- a ragged left margin that is visible down the page.
    #
    # Instead: measure each column's true left edge, tick labels and axis
    # labels included, and justify every letter in that column to it. a and b
    # share the left column and therefore share one x; c is justified to the
    # left edge of the right-hand column's content, which is the same rule
    # applied to the only panel that is not in the left column.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def extent(ax):
        """-> the axes' true display bbox, labels and ticks included.

        NOT ax.get_tightbbox alone. On the cost panel it reports a top of
        386.8 px while the rotated y-label actually reaches 400.2 -- it
        under-measures rotated text -- and a panel letter placed against it
        lands ON the y-label however much padding is added. That is what kept
        dragging "c" back onto "Library size (unique sequences)" after three
        attempts to fix it by moving the letter.
        """
        from matplotlib.transforms import Bbox
        boxes = [ax.get_tightbbox(rend), ax.get_window_extent(rend)]
        parts = [ax.title, ax.xaxis.label, ax.yaxis.label]
        parts += ax.get_xticklabels() + ax.get_yticklabels()
        for t in parts:
            if t.get_text().strip() and t.get_visible():
                boxes.append(t.get_window_extent(rend))
        return Bbox.union(boxes)

    def left_edge(axes):
        return min(inv.transform((extent(a).x0, 0))[0] for a in axes)

    x_left = left_edge([ax_e, ax_cost, ax_lab])
    x_right = left_edge([ax_obj])

    # Y comes from each panel's MEASURED top, not from a fraction of its own
    # height. The fractions (1.12, 1.20, 1.06) had to be hand-tuned per panel
    # because each axes is a different height, and they then moved again every
    # time the row spacing changed -- which is how "c" kept landing on the
    # cost panel's rotated y-label. A fixed gap above the measured top, title
    # and axis labels included, holds whatever the layout does.
    def top_of(ax):
        return inv.transform((0, extent(ax).y1))[1] + 0.006

    for ax, lab, x in ((ax_e, "a", x_left), (ax_cost, "b", x_left),
                       (ax_obj, "c", x_right), (ax_lab, "d", x_left)):
        fig.text(x, top_of(ax), lab, fontsize=PANEL_LABEL_PT,
                 fontweight="bold", va="bottom", ha="left")

    # dpi=900, NOT the house default of 300.
    #
    # This is what actually set the structures' resolution, and it is easy to
    # miss: matplotlib resamples an embedded image to the size it is DRAWN at,
    # in pixels. The fold is placed 228 pt = 3.17 in wide, so at dpi=300 it was
    # resampled to 950 px -- throwing away all but a fifth of a 6000 px render
    # and landing at exactly 300 dpi on the page, which is print-minimum and
    # soft on screen. Raising PX in structures/render.py could not fix this and
    # never will: the ceiling is here.
    #
    # The ink in whole.png is 4615 px across, so the source supports 1457 dpi
    # at this width. 900 asks for 2850 px, comfortably inside that; going past
    # ~1450 would just upsample.
    save(fig, "concept_library", dpi=900)


if __name__ == "__main__":
    main()
