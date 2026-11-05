#!/usr/bin/env python3
"""Figure 2 --- designing high-fidelity combinatorial libraries, computationally.

One row per backbone, and each row is that backbone's whole story:

    (b) the target, coloured at k=4, with its motif as sticks and its ligand
        in grey -- so the reader can see the cut lines run THROUGH the active
        site, not around it
    (c) what post-training buys, vanilla against post-trained, measured on
        recombinants
    (d) how far that holds as the library is cut finer

Panel (a) is the method: what the policy is optimised for.

THE MOTIF SPANS FRAGMENTS, which is why (c) and (d) are the right things to
measure. Under the same equal-count split the libraries use:

    M0097   A26 -> 1, A59 -> 2, A61 -> 2, A93 -> 3     3 of 4 fragments
    M0255   A55 -> 2, A89 -> 2, A91 -> 3, A100 -> 3    2 of 4
    M0315   A90 -> 2, A124 -> 3                        2 of 4

Every recombination here is reassembling an active site from parts that came
from different designs.

WHAT THIS FIGURE DOES NOT CLAIM. The three-arm contrast (sampled / RL-trained
/ trained-for-recombination, ame_three_arm.py) is the comparison the figure
plan calls the method's own claim, and on retention it currently runs the
WRONG WAY: fragrl beats policy on 0 of 3 backbones, 53.8% against 76.7%. It
wins on rarefied diversity 3 of 3, and the arms differ in parent count by
2-6x, so they are not matched. Until that is understood the recombination
claim rests on trained-at-k vs post-hoc-cut, which is consistent 3/3 at every
k, and this figure is built on that.

    uv run python paper/figures/fig2_libraries.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ame_fragk import (C, MARK, add_diversity, collect_posthoc,  # noqa: E402
                       collect_trained)
from concept_loop import panel_loop  # noqa: E402
from figio import save  # noqa: E402
from palette import (INK, PAGE_WIDTH_IN, PALETTE,  # noqa: E402
                     PANEL_LABEL_PT, fs)
from panel_vanilla import ARMC, gradient_marker  # noqa: E402
from panel_vanilla import collect as collect_vanilla
from schematic import place  # noqa: E402

SITES = ("M0097", "M0255", "M0315")


def _pale(hex_c, f=0.42):
    """-> `hex_c` lightened, so a legend key reads like the gradient marker."""
    r, g, b = (int(hex_c.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(c + (1 - c) * f for c in (r, g, b))


def panel_target(ax, site):
    """The backbone, coloured at k=4, motif as sticks, ligand in grey."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    place(ax, f"target_{site}", (0.56, 0.50), 92)


def panel_move(ax, v, p, site, show_x):
    """Vanilla -> post-trained for ONE backbone, measured on recombinants."""
    ax.annotate("", xy=(p.subs, p.success), xytext=(v.subs, v.success),
                arrowprops=dict(arrowstyle="-|>", color=INK["ghost"], lw=1.3,
                                shrinkA=7, shrinkB=9))
    for arm, r in (("vanilla", v), ("post", p)):
        gradient_marker(ax, r.subs, r.success, ARMC[arm], size_pts=13.0)
    ax.set_ylim(0, 108)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylabel("passing (\\%)", fontsize=fs(7.5))
    if show_x:
        ax.set_xlabel("distinct mutations (rarefied)", fontsize=fs(7.5))
    else:
        ax.tick_params(labelbottom=False)
    ax.tick_params(labelsize=fs(7))


def panel_k(ax, d, site, show_x):
    """Retention against k for ONE backbone, trained-at-k vs post-hoc cut."""
    for cond in ("posthoc", "trained"):
        g = d[(d.site == site) & (d.cond == cond)].sort_values("k")
        if not len(g):
            continue
        ax.plot(g.k, 100 * g.retention, "-", color=C[cond], lw=1.8, zorder=2)
        for r in g.itertuples():
            hollow = cond == "trained" and not r.matched
            ax.plot(r.k, 100 * r.retention, MARK[site], ms=6, zorder=3,
                    mfc="white" if hollow else C[cond], mec=C[cond], mew=1.6)
    ax.axhline(50, color=PALETTE["red"], lw=1, ls=":", zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 36, 64])
    ax.set_xticklabels(["2", "4", "8", "16", "36", "64"])
    ax.set_ylim(0, 108)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if show_x:
        ax.set_xlabel("fragments per design, $k$", fontsize=fs(7.5))
    else:
        ax.tick_params(labelbottom=False)
    ax.tick_params(labelsize=fs(7))


def main():
    rng = np.random.default_rng(0)
    plt.rcParams["text.usetex"] = False

    d = pd.DataFrame(collect_trained() + collect_posthoc())
    if not len(d):
        sys.exit("no cells found -- is experiments/ame mounted?")
    d = add_diversity(d, 200, rng)
    van = collect_vanilla(rng=np.random.default_rng(0)).set_index(
        ["site", "arm"])

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 8.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.72, 2.55], hspace=0.26,
                             left=0.095, right=0.985, top=0.945, bottom=0.055)
    ax_a = fig.add_subplot(outer[0])
    panel_loop(ax_a, np.random.default_rng(3))

    # One row per backbone. The structure column is narrower than the two
    # plots: it carries no axes, and at equal width the folds would dominate
    # a row whose argument is the two plots beside them.
    grid = outer[1].subgridspec(3, 3, width_ratios=[0.78, 1.0, 1.0],
                                hspace=0.20, wspace=0.46)
    ax_v0 = ax_k0 = None
    first = {}
    for i, site in enumerate(SITES):
        ax_s = fig.add_subplot(grid[i, 0])
        panel_target(ax_s, site)
        ax_s.text(0.02, 0.5, site, transform=ax_s.transAxes, rotation=90,
                  ha="center", va="center", fontsize=fs(8))

        ax_v = fig.add_subplot(grid[i, 1], sharex=ax_v0, sharey=ax_v0)
        ax_v0 = ax_v0 or ax_v
        panel_move(ax_v, van.loc[(site, "vanilla")], van.loc[(site, "post")],
                   site, show_x=(i == len(SITES) - 1))

        ax_k = fig.add_subplot(grid[i, 2], sharex=ax_k0, sharey=ax_k0)
        ax_k0 = ax_k0 or ax_k
        panel_k(ax_k, d, site, show_x=(i == len(SITES) - 1))
        if i == 0:
            first = dict(s=ax_s, v=ax_v, k=ax_k)

    # One legend per plot column, on the top row only.
    first["v"].legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=7,
               mfc=_pale(ARMC["vanilla"]), mec=ARMC["vanilla"], mew=1.1,
               label="vanilla, $T$ pooled"),
        Line2D([], [], marker="o", ls="none", ms=7,
               mfc=_pale(ARMC["post"]), mec=ARMC["post"], mew=1.1,
               label="post-trained")],
        frameon=False, fontsize=fs(6.4), loc="lower right",
        handletextpad=0.4, labelspacing=0.25)
    first["k"].legend(handles=[
        Line2D([], [], color=C["trained"], lw=1.8, label="trained at $k$"),
        Line2D([], [], color=C["posthoc"], lw=1.8,
               label="$k=4$ policy, cut at $k$")],
        frameon=False, fontsize=fs(6.4), loc="lower left",
        handletextpad=0.6, labelspacing=0.25)

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

    y_row = max(inv.transform((0, extent(a).y1))[1]
                for a in (first["s"], first["v"], first["k"]))
    for ax, lab, y in ((ax_a, "a", inv.transform((0, extent(ax_a).y1))[1]),
                       (first["s"], "b", y_row),
                       (first["v"], "c", y_row),
                       (first["k"], "d", y_row)):
        x = inv.transform((extent(ax).x0, 0))[0]
        fig.text(x, y + 0.006, lab, fontsize=PANEL_LABEL_PT,
                 fontweight="bold", va="bottom", ha="left")

    save(fig, "fig2_libraries", dpi=900, tight=False)


if __name__ == "__main__":
    main()
