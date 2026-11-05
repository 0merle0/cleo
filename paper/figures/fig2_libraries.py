#!/usr/bin/env python3
"""Figure 2 --- designing high-fidelity combinatorial libraries, computationally.

The claim this figure has to carry, in one sentence: TRAINING A DESIGNER FOR
RECOMBINATION IS WHAT MAKES ITS FRAGMENTS COMBINABLE, AND THE ADVANTAGE GROWS
THE FINER YOU CUT.

That is the k sweep (E18/E18b), and it is the strongest result in the project:
three backbones, six trained k levels against seven post-hoc levels, every
trained cell at a matched 200 steps so budget cannot covary with k.

    (a) the method -- what the policy is optimised for
    (b) retention vs k, trained-at-k against a k=4 policy re-cut at k,
        one sub-panel per backbone
    (c) the gap, which widens monotonically with k
    (d) substitution diversity, rarefied, which does NOT order in k

(d) is not decoration. A policy can satisfy "make recombinable fragments" by
emitting near-identical sequences -- every chimera then reproduces a parent and
passes. Retention alone cannot tell that apart from genuine modularity.
Diversity holding up while retention rises is what rules it out, so the two
belong in one figure.

WHAT THIS FIGURE DOES NOT CLAIM. The three-arm contrast (sampled / RL-trained /
trained-for-recombination, ame_three_arm.py) is the comparison the figure plan
calls the method's own claim, and on retention it currently runs the WRONG WAY:
fragrl beats policy on 0 of 3 backbones, 53.8% against 76.7%. It wins on
rarefied diversity, 3 of 3. The arms also differ in parent count by 2-6x, so
they are not matched. Until that is understood or matched, the recombination
claim rests on trained-at-k vs post-hoc-cut, which is consistent on 3/3
backbones at every k, and this figure is built on that instead.

    uv run python paper/figures/fig2_libraries.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ame_fragk import (add_diversity, collect_posthoc,  # noqa: E402
                       collect_trained, panel_diversity, panel_gap,
                       panel_retention)
from concept_loop import panel_loop  # noqa: E402
from figio import save  # noqa: E402
from palette import PAGE_WIDTH_IN, PANEL_LABEL_PT  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    plt.rcParams["text.usetex"] = False

    d = pd.DataFrame(collect_trained() + collect_posthoc())
    if not len(d):
        sys.exit("no cells found -- is experiments/ame mounted?")
    d = add_diversity(d, 200, rng)

    # Page width and fixed margins, like Figure 1: a point here is a point on
    # the page, so palette.MIN_FONT means something, and the rows share one
    # left and right edge instead of each being cropped to its own widest ink.
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 8.7))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.80, 1.00, 1.00],
                             hspace=0.55, left=0.095, right=0.985,
                             top=0.930, bottom=0.05)

    ax_m = fig.add_subplot(outer[0])
    panel_loop(ax_m, np.random.default_rng(3))

    # panel_retention indexes gs[0, 2i:2i+2], so it needs SIX columns but
    # only one row. Handing it the (2, 6) grid its own script uses left row 1
    # empty -- half the band under panel (b) was dead space.
    ret = outer[1].subgridspec(1, 6, wspace=0.55)
    panel_retention(d, fig, ret)

    bot = outer[2].subgridspec(1, 2, wspace=0.42)
    ax_g = fig.add_subplot(bot[0, 0])
    ax_d = fig.add_subplot(bot[0, 1])
    panel_gap(d, ax_g)
    panel_diversity(d, ax_d)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def extent(ax):
        from matplotlib.transforms import Bbox
        boxes = [ax.get_tightbbox(rend), ax.get_window_extent(rend)]
        parts = [ax.title, ax.xaxis.label, ax.yaxis.label]
        parts += ax.get_xticklabels() + ax.get_yticklabels()
        for t in parts:
            if t.get_text().strip() and t.get_visible():
                boxes.append(t.get_window_extent(rend))
        return Bbox.union(boxes)

    first_ret = fig.axes[1]
    x_left = min(inv.transform((extent(a).x0, 0))[0]
                 for a in (ax_m, first_ret, ax_g))
    for ax, lab in ((ax_m, "a"), (first_ret, "b"), (ax_g, "c"), (ax_d, "d")):
        x = x_left if ax is not ax_d else \
            inv.transform((extent(ax_d).x0, 0))[0]
        y = inv.transform((0, extent(ax).y1))[1] + 0.006
        fig.text(x, y, lab, fontsize=PANEL_LABEL_PT, fontweight="bold",
                 va="bottom", ha="left")

    save(fig, "fig2_libraries", dpi=900, tight=False)


if __name__ == "__main__":
    main()
