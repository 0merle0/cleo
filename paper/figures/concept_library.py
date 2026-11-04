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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

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


def place(ax, name, xy, width_pts):
    """Draw a structure PNG `width_pts` wide, whatever its pixel size.

    Sized in POINTS, not by OffsetImage's zoom. zoom is per-pixel, so a figure
    written against it silently rescales every structure whenever the render
    resolution changes -- raising PX from 3600 to 6000 blew the fold out of
    its panel and over the one below. Point width is what the layout actually
    means, and it holds across re-renders.
    """
    im = trimmed(name)
    ax.add_artist(AnnotationBbox(
        OffsetImage(im, zoom=width_pts / im.shape[1]), xy,
        frameon=False, box_alignment=(0.5, 0.5)))


def place_raw(ax, name, xy, width_pts):
    """Place a structure PNG UNTRIMMED, so every image keeps one frame.

    `place` crops to the ink, which is right for a single structure and fatal
    for the exploded panel: cropping rescales each fragment to fill its own
    box, so they stop sharing a camera and stop sharing a size. Untrimmed, a
    fragment sits at its true position inside a mostly transparent frame, and
    offsetting the frame slides the fragment outward by exactly that much.
    """
    im = mpimg.imread(PNG / f"{name}.png")
    ax.add_artist(AnnotationBbox(
        OffsetImage(im, zoom=width_pts / im.shape[1]), xy,
        frameon=False, box_alignment=(0.5, 0.5), annotation_clip=False))


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
N_ORDER = 36            # designs actually ordered -- the SAME for all three
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


def draw_landscape(ax, title, title_color=None):
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
    ax.set_title(title, fontsize=9, pad=7,
                 color=title_color or PALETTE["gray"])


def draw_designs(ax, pts, ms=3.8):
    """The designs that were ordered and built."""
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=ms, mfc=DOT, mec="white", mew=0.55,
            ls="none", zorder=5)


def draw_start(ax):
    ax.plot(*START, marker="*", ms=12, mfc="white", mec=DOT, mew=1.1,
            ls="none", zorder=6)


def panel_epcr(ax, rng):
    """Mutagenesis walks: dense, but it never leaves the parent's basin."""
    draw_landscape(ax, "error-prone PCR")
    pts = _clip(rng.normal(START, 0.055, size=(N_ORDER, 2)))
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
    draw_landscape(ax, "naive combinatorial assembly")
    r = np.random.default_rng(50)
    xs = np.sort(r.uniform(0.05, 0.95, 6))
    ys = np.sort(r.uniform(0.05, 0.95, 6))
    pts = _clip(np.array([(x, y) for x in xs for y in ys], float)
                + rng.normal(0, 0.012, (36, 2)))
    draw_designs(ax, pts)
    draw_start(ax)
    cx, cy, w, _ = PEAKS[1]
    ax.add_patch(Circle((cx, cy), w * 1.55, fill=False, ec=PALETTE["red"],
                        lw=1.1, ls=(0, (2.5, 2)), zorder=6))
    ax.annotate("missed", xy=(cx + w * 1.45, cy - w * 0.6),
                xytext=(0.985, 0.55), fontsize=7.5, color=PALETTE["red"],
                ha="right", va="center", zorder=7,
                bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85),
                arrowprops=dict(arrowstyle="-", lw=0.8,
                                color=PALETTE["red"], shrinkA=1, shrinkB=1))


def panel_ours(ax, rng):
    """Generate the same combinatorial reach, then filter hard before ordering.

    The point of the panel is that the filter IS the search. A policy-designed
    library spans the space as widely as the naive one -- the faint open
    candidates -- but only what survives folding and scoring gets built, and
    those survivors sit on the optima. The budget of filled points is
    N_ORDER, identical to the other two panels, so the three differ in WHERE
    the designs go and not in how many there are.
    """
    draw_landscape(ax, "stringent filtering (this work)",
                   title_color=PALETTE["blue"])
    xs = rng.uniform(0.05, 0.95, 14)
    ys = rng.uniform(0.05, 0.95, 14)
    cand = _clip(np.array([(x, y) for x in xs for y in ys], float)
                 + rng.normal(0, 0.010, (196, 2)))
    keep = np.argsort(fitness(cand[:, 0], cand[:, 1]))[-N_ORDER:]
    mask = np.zeros(len(cand), bool)
    mask[keep] = True
    ax.plot(cand[~mask, 0], cand[~mask, 1], "o", ms=2.6, mfc="none",
            mec=INK["ghost"], mew=0.7, ls="none", zorder=4)
    draw_designs(ax, cand[mask])
    draw_start(ax)
    ax.plot([], [], "o", ms=2.6, mfc="none", mec=INK["ghost"], mew=0.7,
            ls="none", label="filtered out")
    ax.plot([], [], "o", ms=3.8, mfc=DOT, mec="white", mew=0.55, ls="none",
            label="ordered")
    ax.legend(fontsize=6.8, loc="lower left", handletextpad=0.3,
              borderpad=0.35, labelspacing=0.25, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.88)


def fitness_key(ax):
    """A direction, not a scale -- the landscape is illustrative."""
    cb = ax.inset_axes([1.045, 0.02, 0.032, 0.80])
    g = np.linspace(0, 1, 256).reshape(-1, 1)
    cb.imshow(g, aspect="auto", origin="lower", cmap=TERRAIN,
              extent=(0, 1, 0, 1))
    cb.set_xticks([])
    cb.set_yticks([])
    for sp in cb.spines.values():
        sp.set_color(INK["grid"])
    cb.text(1.5, 1.0, "high", ha="left", va="center", fontsize=6.6,
            color=PALETTE["gray"], transform=cb.transAxes)
    cb.text(1.5, 0.0, "low", ha="left", va="center", fontsize=6.6,
            color=PALETTE["gray"], transform=cb.transAxes)
    cb.text(-0.9, 0.5, "fitness", rotation=90, ha="center", va="center",
            fontsize=7.5, color=PALETTE["gray"], transform=cb.transAxes)


# ---------------------------------------------------------------- panel (b)
def mark_level(ax, S, series, label, label_x):
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
    # Caption goes in the widest gap ALONG the line, which is between the two
    # dearest arms -- k=2 meets this level at ~$42M and whole genes at ~$42T,
    # six decades apart, and nothing else crosses in between. Both ends of the
    # line were tried and both crowd a price: the left end sits in the cluster
    # where the cheap arms converge, the right end runs into the dearest one.
    ax.text(label_x, S * 1.6, label, ha="center", fontsize=7.5,
            color=PALETTE["gray"])
    marks = sorted(((float(library_cost(np.array(S), k)), c)
                    for _, k, c in series), key=lambda t: t[0])
    # The alternation is a good default and gets two of the four right. The
    # cheapest arm is the one it cannot place: at this level k=8 crosses so
    # far left that a label set below-right of the point lies along the k=4
    # curve and is struck through by it. Overrides are keyed by rank in cost,
    # which is what the alternation itself is keyed by.
    OVERRIDE = {
        0: (0.62, 3.2, "right", "bottom"),   # $5.3k -- above-left, off the line
        1: (2.6, 3.2, "left", "bottom"),     # $84k  -- further right, clear of 0
    }
    for i, (x, c) in enumerate(marks):
        ax.plot([x], [S], "o", ms=6.5, mfc=c, mec="white", mew=1.4, zorder=4)
        up = i % 2 == 1
        y, va = (S * 3.2, "bottom") if up else (S * 0.32, "top")
        if i == len(marks) - 1:
            xt, ha = x * 0.5, "right"
        else:
            xt, ha = x * 1.6, "left"
        if i in OVERRIDE:
            fx, fy, ha, va = OVERRIDE[i]
            xt, y = x * fx, S * fy
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
    mark_level(ax, 1e12, series, r"$10^{12}$ variants", 1e10)
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
    place(ax, "whole", (0.5, 0.5), 228)


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

    place(ax, "grey", (0.72, 2.05), 108)
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
    fig = plt.figure(figsize=(11, 11.0))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.12, 1.22, 0.78],
                             hspace=0.30)
    # The three landscapes share a row and a letter: they are one argument
    # seen three ways, not three findings, and lettering them a/b/c would
    # promise the reader three separate results.
    top = outer[0].subgridspec(1, 3, wspace=0.11)
    ax_e = fig.add_subplot(top[0, 0])
    ax_n = fig.add_subplot(top[0, 1])
    ax_o = fig.add_subplot(top[0, 2])
    panel_epcr(ax_e, rng)
    panel_naive(ax_n, rng)
    panel_ours(ax_o, rng)
    fitness_key(ax_o)

    mid = outer[1].subgridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.22)
    ax_obj = fig.add_subplot(mid[0, 0])
    ax_cost = fig.add_subplot(mid[0, 1])
    ax_loop = fig.add_subplot(outer[2])
    panel_object(ax_obj)
    panel_cost(ax_cost)
    panel_loop(ax_loop, rng)

    for ax, lab, dy in ((ax_e, "a", 1.20), (ax_obj, "b", 1.12),
                        (ax_cost, "c", 1.12), (ax_loop, "d", 1.12)):
        ax.text(-0.02, dy, lab, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="top", ha="right")
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
