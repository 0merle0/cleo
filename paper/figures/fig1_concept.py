#!/usr/bin/env python3
"""Figure 1 --- why combinatorial, and what the method is.

Blocked out against the story draft (see paper/figure_plan_story.md). The
draft put the concept, the method, the objective, the benchmark, the diversity
result, the embedding, the k sweep and the cost argument all in Figure 1,
which is why every reference in it reads "figure 1?". That is split here:
Figure 1 is the concept and the method, Figure 2 is the benchmark.

Panel (a) is REAL and drawn in 3D, which is the change the draft asks for --
see landscape3d.py for why it is a hand-rolled projection and not mplot3d.
Panels (b), (c) and (d) are placeholders carrying what goes in them, so the
page budget and panel order can be edited now.

    uv run python paper/figures/fig1_concept.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import landscape3d as L3  # noqa: E402
from blockout import slot  # noqa: E402
from concept_library import N_VARIANTS, START, _clip, fitness  # noqa: E402
from figio import save  # noqa: E402
from palette import INK, PAGE_WIDTH_IN, fs  # noqa: E402


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
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 6.6))
    outer = fig.add_gridspec(
        3, 1, height_ratios=[0.72, 1.60, 1.05],
        left=0.055, right=0.985, top=0.955, bottom=0.045, hspace=0.34)

    # (a) three landscapes, one budget spent three ways.
    row = outer[0].subgridspec(1, 3, wspace=0.07)
    rng = np.random.default_rng(7)
    for i, fn in enumerate((panel_epcr, panel_naive, panel_ours)):
        fn(fig.add_subplot(row[0, i]), rng)
    fig.text(0.012, 0.955, "(a)", fontsize=fs(9), fontweight="bold",
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

    # (d) what it costs, against what it buys.
    bot = outer[2].subgridspec(1, 1)
    slot(fig.add_subplot(bot[0, 0]), "d",
         "Oracle calls spent vs library size reached",
         "have", "panel_cost.py; concept_library.library_cost",
         lines=[
             "~3.2e4 oracle calls (200 steps x 32 x best-of-5) buys a library",
             "of >1e12 assemblable variants -- per-variant folding cost is nil",
             "second axis: $ per library member, full genes vs k fragments",
             "S = (B / 3Lc)^k is the arithmetic behind both",
         ])

    save(fig, "fig1_concept", dpi=900, tight=False)


if __name__ == "__main__":
    main()
