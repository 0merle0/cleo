#!/usr/bin/env python3
"""Supplementary --- the control dropped from Figure 2(d).

Figure 2(d) shows only the trained-at-k arm, which says retention falls as a
library is cut finer but not WHY it holds up as long as it does. The
counterfactual is here: one policy trained at k=4 and re-cut at every k, with
no training at the finer granularity.

    (a) both arms, one facet per backbone
    (b) the difference, which widens monotonically with k

This is the paper's strongest evidence for the recombination claim, since the
three-arm contrast (ame_three_arm.py) currently runs the other way on
retention. It is supplementary only because Figure 2 is already carrying
three columns per backbone, not because it is weak.

Matched budget is the point of care: every TRAINED cell ran 200 steps, so
training length cannot covary with k. The post-hoc arm needs no training at
all -- it is one k=4 policy, re-cut -- which is exactly what makes it the
right control.

    uv run python paper/figures/figS_posthoc.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ame_fragk import (C, add_diversity,  # noqa: E402
                       collect_posthoc, collect_trained)
from figio import save  # noqa: E402
from palette import INK, PAGE_WIDTH_IN, PALETTE, PANEL_LABEL_PT, fs  # noqa: E402

SITES = ("M0097", "M0255", "M0315")

# A lighter grey for the control than ame_fragk's own PALETTE["gray"],
# so the trained arm reads as the subject and the control as the
# baseline it is measured against.
C = dict(C, posthoc=INK["muted"])


def facet(ax, d, site, show_y):
    for cond in ("posthoc", "trained"):
        g = d[(d.site == site) & (d.cond == cond)].sort_values("k")
        if not len(g):
            continue
        ax.plot(g.k, 100 * g.retention, "-", color=C[cond], lw=1.8, zorder=2)
        for r in g.itertuples():
            hollow = cond == "trained" and not r.matched
            ax.plot(r.k, 100 * r.retention, "o", ms=6, zorder=3,
                    mfc="white" if hollow else C[cond], mec=C[cond], mew=1.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 36, 64])
    ax.set_xticklabels(["2", "4", "8", "16", "36", "64"])
    ax.set_ylim(0, 108)
    ax.set_title(site, fontsize=fs(8))
    ax.set_xlabel("fragments per design, $k$", fontsize=fs(7.5))
    if show_y:
        ax.set_ylabel("recombinants passing (\\%)", fontsize=fs(7.5))
    else:
        ax.tick_params(labelleft=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=fs(7))


def main():
    rng = np.random.default_rng(0)
    plt.rcParams["text.usetex"] = False
    d = pd.DataFrame(collect_trained() + collect_posthoc())
    if not len(d):
        sys.exit("no cells found")
    d = add_diversity(d, 200, rng)

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 5.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.52,
                             left=0.095, right=0.985, top=0.94, bottom=0.085)
    top = outer[0].subgridspec(1, 3, wspace=0.12)
    axes = []
    for i, site in enumerate(SITES):
        ax = fig.add_subplot(top[0, i], sharey=axes[0] if axes else None)
        axes.append(ax)
        facet(ax, d, site, show_y=(i == 0))
    axes[0].legend(handles=[
        Line2D([], [], color=C["trained"], lw=1.8, label="trained at $k$"),
        Line2D([], [], color=C["posthoc"], lw=1.8,
               label="$k=4$ policy, cut at $k$")],
        frameon=False, fontsize=fs(6.6), loc="lower left",
        handletextpad=0.6, labelspacing=0.25)

    # The gap, drawn here rather than imported from ame_fragk: that version
    # keys each backbone by marker SHAPE, and this document uses circles
    # throughout.
    ax_g = fig.add_subplot(outer[1])
    piv = {}
    for site in SITES:
        t = d[(d.site == site) & (d.cond == "trained")].set_index("k")
        h = d[(d.site == site) & (d.cond == "posthoc")].set_index("k")
        ks = sorted(set(t.index) & set(h.index))
        piv[site] = (ks, [100 * (t.retention[k] - h.retention[k]) for k in ks])
    for site, (ks, gaps) in piv.items():
        ax_g.plot(ks, gaps, "-o", ms=6, lw=1.8, color=C["trained"],
                  mfc=C["trained"], mec=C["trained"], label=site)
    ax_g.axhline(0, color=INK["muted"], lw=1, zorder=1)
    ax_g.set_xscale("log", base=2)
    ax_g.set_xticks([2, 4, 8, 16, 36])
    ax_g.set_xticklabels(["2", "4", "8", "16", "36"])
    ax_g.set_xlabel("fragments per design, $k$", fontsize=fs(7.5))
    ax_g.set_ylabel("retention gained by training at $k$ (pts)",
                    fontsize=fs(7.5))
    ax_g.tick_params(labelsize=fs(7))
    for sp in ("top", "right"):
        ax_g.spines[sp].set_visible(False)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def extent(ax):
        from matplotlib.transforms import Bbox
        boxes = [ax.get_tightbbox(rend), ax.get_window_extent(rend)]
        for t in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                  + ax.get_xticklabels() + ax.get_yticklabels()):
            if t.get_text().strip() and t.get_visible():
                boxes.append(t.get_window_extent(rend))
        return Bbox.union(boxes)

    x = min(inv.transform((extent(a).x0, 0))[0] for a in (axes[0], ax_g))
    for ax, lab in ((axes[0], "a"), (ax_g, "b")):
        y = inv.transform((0, extent(ax).y1))[1] + 0.006
        fig.text(x, y, lab, fontsize=PANEL_LABEL_PT, fontweight="bold",
                 va="bottom", ha="left")

    save(fig, "figS_posthoc", dpi=900, tight=False)


if __name__ == "__main__":
    main()
