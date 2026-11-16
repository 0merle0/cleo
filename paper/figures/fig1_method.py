#!/usr/bin/env python3
"""Figure 1 --- the method, and the benchmark that says it works.

Blocked out against the story draft (see paper/figure_plan_story.md), in the
draft's own FOUR-figure scheme: this figure carries the concept, the method
and the AME benchmark together, which is what every "figure 1?" in the draft
is pointing at.

Eight panels, so it is a tall figure -- 9.9 in against a 6.5 in page. That is
a deliberate consequence of the four-figure format and the thing to watch
when it is placed. If it has to come down, (d) is the panel whose argument
the text can carry alone.

    (a) three landscapes      (e) recombinants fold
    (b) fragment shuffling    (f) diversity over passing members
    (c) the GRPO objective    (g) where in sequence space
    (d) cost vs library size  (h) how fine it can be cut

Panel (a) is REAL and drawn in 3D, which is the change the draft asks for --
see landscape3d.py for why it is a hand-rolled projection and not mplot3d.
The rest are placeholders carrying what goes in them, so the page budget and
panel order can be edited now.

    uv run python paper/figures/fig1_method.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import landscape3d as L3  # noqa: E402
from blockout import slot  # noqa: E402
from concept_library import N_VARIANTS, _clip  # noqa: E402
from figio import save  # noqa: E402
from palette import INK, PAGE_WIDTH_IN, fs  # noqa: E402

# --- the toy landscape ---------------------------------------------------
#
# THREE optima, not the four the 2D version carries. At the size this panel
# prints, a fourth peak is a fourth thing to track in a drawing whose whole
# argument is "the parent sits on one hill and there are others" -- three
# says it and leaves the mesh legible. concept_library keeps the four-peak
# 2D version; this landscape is defined here so changing one cannot silently
# change the other.
#
#        cx,   cy,   width, height
# Placed by their PROJECTED x (px = x + SHEAR*y), not their x. Two peaks a
# long way apart in the plane can land on top of each other in an oblique
# view: (0.73, 0.70) and (0.79, 0.25) project to 1.02 and 0.90 and read as
# one ridge. These three project to 0.39 / 0.66 / 1.03 and stay separate.
PEAKS = [
    (0.25, 0.34, 0.094, 0.74),   # the parent's own optimum -- good, not best
    (0.72, 0.74, 0.104, 1.00),   # the global optimum
    (0.61, 0.11, 0.086, 0.86),
]
START = (0.25, 0.34)


def fitness(x, y):
    """-> the toy landscape: three gaussian optima plus gentle ruggedness."""
    z = sum(a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
            for cx, cy, s, a in PEAKS)
    return z + 0.02 * np.sin(4 * np.pi * x) * np.sin(3.5 * np.pi * y)


# One zmax for all three landscapes. Computed once from the shared fitness
# surface rather than per panel: normalising each panel to its own maximum
# would make three different vertical scales out of one landscape, and the
# panel's entire job is that the landscape is the same in all three.
def _zmax():
    g = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(g, g)
    return float(np.max(fitness(X, Y)))


ZMAX = _zmax()


def _surface(ax):
    L3.frame(ax)
    L3.draw_surface(ax, fitness, zmax=ZMAX)


def panel_epcr(ax, rng):
    """Mutagenesis walks: dense, and it never leaves the parent's basin."""
    _surface(ax)
    L3.title(ax, "Error-prone PCR")
    pts = _clip(rng.normal(START, 0.055, size=(N_VARIANTS, 2)))
    L3.draw_points(ax, fitness, pts, ZMAX)
    L3.draw_start(ax, fitness, START, ZMAX)
    L3.axis_cue(ax)


def panel_naive(ax, rng):
    """Combinatorial assembly with parts chosen without regard to fitness.

    Drawn as a PRODUCT SET rather than a cloud, because that is what assembly
    builds: 6 variants for block 1 crossed with 6 for block 2 is their 36
    combinations, not 36 free points.
    """
    _surface(ax)
    L3.title(ax, "Naive combinatorial")
    r = np.random.default_rng(50)
    xs = np.sort(r.uniform(0.05, 0.95, 6))
    ys = np.sort(r.uniform(0.05, 0.95, 6))
    pts = _clip(np.array([(x, y) for x in xs for y in ys], float)
                + rng.normal(0, 0.012, (36, 2)))
    L3.draw_points(ax, fitness, pts, ZMAX)
    L3.draw_start(ax, fitness, START, ZMAX)


def panel_ours(ax, rng):
    """Generate the same reach, then filter hard before building anything.

    The filter IS the search. The library spans the space as widely as the
    naive one -- the faint open candidates -- but only what survives folding
    and scoring is kept, and the survivors sit on the optima. The filled
    points number N_VARIANTS, identical to the other two panels, so the three
    differ in WHERE the budget goes and not in how much of it there is.
    """
    _surface(ax)
    L3.title(ax, "Filtered combinatorial")
    xs = rng.uniform(0.05, 0.95, 14)
    ys = rng.uniform(0.05, 0.95, 14)
    cand = _clip(np.array([(x, y) for x in xs for y in ys], float)
                 + rng.normal(0, 0.010, (196, 2)))
    keep = np.argsort(fitness(cand[:, 0], cand[:, 1]))[-N_VARIANTS:]
    mask = np.zeros(len(cand), bool)
    mask[keep] = True

    rej = cand[~mask]
    z = fitness(rej[:, 0], rej[:, 1]) / ZMAX + 0.012
    px, py = L3.iso(rej[:, 0], rej[:, 1], z)
    ax.plot(px, py, "o", ms=2.3, mfc="none", mec=INK["muted"], mew=0.5,
            ls="none", zorder=5)

    L3.draw_points(ax, fitness, cand[mask], ZMAX)
    L3.draw_start(ax, fitness, START, ZMAX)


def main():
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 9.9))
    outer = fig.add_gridspec(
        4, 1, height_ratios=[0.74, 1.24, 1.56, 1.20],
        left=0.055, right=0.985, top=0.972, bottom=0.028, hspace=0.26)

    # (a) three landscapes, one budget spent three ways.
    row = outer[0].subgridspec(1, 3, wspace=0.07)
    rng = np.random.default_rng(7)
    for i, fn in enumerate((panel_epcr, panel_naive, panel_ours)):
        fn(fig.add_subplot(row[0, i]), rng)
    fig.text(0.012, 0.972, "(a)", fontsize=fs(9), fontweight="bold",
             ha="left", va="top")

    # (b) the shuffling scheme, (c) the objective it feeds.
    mid = outer[1].subgridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.09)
    slot(fig.add_subplot(mid[0, 0]), "b",
         "Fragment shuffling during training",
         "have", "schematic; concept_loop.py has the credit half",
         lines=[
             "N_batch = 16 decoded from the policy, each cut into k fragments",
             "N_sampled = 64 chimeras drawn by recombining across parents",
             "every fragment appears N_sampled // N_batch = 4 times",
             "-> reward per fragment averaged over the chimeras holding it",
             "NOTE: rho = 0 returns the parent, so 1/4 of the batch is not",
             "a chimera. The scheme must show that, not hide it.",
         ])
    slot(fig.add_subplot(mid[0, 1]), "c",
         "The per-fragment GRPO objective",
         "have", "typeset block -- symbols labelled against (b)",
         lines=[
             "pi(F | str) rather than pi(seq | str)",
             "R(f_k,i) = mean over chimeras containing f_k,i",
             "advantage normalised within the group",
             "place it INSIDE the figure so each symbol sits",
             "beside the step of (b) that produces it",
         ])

    # (d) what it costs, (e) and (f): what the benchmark says it buys.
    r3 = outer[2].subgridspec(1, 3, wspace=0.10)
    slot(fig.add_subplot(r3[0, 0]), "d",
         "Oracle calls vs\nlibrary size",
         "have", "panel_cost.py",
         lines=[
             "~3.2e4 oracle calls",
             "(200 steps x 32 x",
             "best-of-5) buys a",
             "library of >1e12",
             "assemblable variants",
             "second axis: $ per",
             "member, full genes",
             "vs k fragments",
             "S = (B / 3Lc)^k",
         ])
    slot(fig.add_subplot(r3[0, 1]), "e",
         "Recombinants fold,\nand training is why",
         "have", "key_results.md s6; arm_table.py",
         lines=[
             "recombinant pass rate,",
             "AF3 best-of-5, held out",
             "POOLED over the three",
             "backbones; one bar per",
             "arm with the 3 points",
             "drawn on it:",
             "  plain RL  53.5",
             "  chimera   93.1",
             "  = +39.6 pt",
             "keep the points: M0255",
             "is the hard one (82.4)",
             "and the mean hides it",
             "DO NOT plot retention",
         ])
    slot(fig.add_subplot(r3[0, 2]), "f",
         "More information,\nnot just more members",
         "have", "ame_fragk_diversity.csv",
         lines=[
             "distinct (position,",
             "residue) substitutions",
             "over members that PASS,",
             "rarefied to a matched",
             "passing count --",
             "diversity among failures",
             "is diversity you cannot",
             "order",
             "already pooled: 721.6",
             "-> 859.6 over 8",
             "backbones, 6 of 8 won",
             "per-backbone points",
             "here too, as in (e)",
         ])

    # (g) where the library sits, (h) how fine it can be cut.
    r4 = outer[3].subgridspec(1, 2, wspace=0.10)
    slot(fig.add_subplot(r4[0, 0]), "g",
         "Where in sequence space each library sits",
         "derive", "archived ame_pca.py; third arm needs building",
         lines=[
             "one embedding, three arms drawn in it:",
             "  random mutagenesis (scoped)",
             "  naive ProteinMPNN",
             "  post-trained policy",
             "GAP: no random-mutagenesis arm at matched size.",
             "jq_e0307_250311_petase_library_random_mutation",
             "has AF3 output for one -- check it can serve",
         ])
    slot(fig.add_subplot(r4[0, 1]), "h",
         "How fine the library can be cut",
         "have", "ame_fragk.csv -- trained-at-k vs cut-at-k",
         lines=[
             "retention vs k, POOLED over the three backbones:",
             "mean curve, with the min-max band behind it",
             "  k      2     4     8    16    36    64",
             "  trained   0.91  0.90  0.94  0.92  0.60  0.29",
             "  cut at k  0.83  --    0.76  0.64  0.28   --",
             "flat through k=16, then a cliff at 36 -- and the gap",
             "over a k=4-trained policy is +27.9 pt at k=16",
             "the draft's 'out to k=16' is the conservative read",
             "CAVEAT: M0255 at k=36/64 was still climbing at step",
             "200, so the pooled tail is a lower bound",
         ])

    save(fig, "fig1_method", dpi=900, tight=False, subdir="main")


if __name__ == "__main__":
    main()
