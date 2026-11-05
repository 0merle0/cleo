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
from ame_fragk import (C, add_diversity, collect_posthoc,  # noqa: E402
                       collect_trained)
from concept_loop import panel_loop  # noqa: E402
from figio import save  # noqa: E402
from palette import (INK, PAGE_WIDTH_IN, PALETTE,  # noqa: E402
                     PANEL_LABEL_PT, fs)
from panel_cost import (TRAIN_FOLDS, brute_force_folds,  # noqa: E402
                        crossover)
from panel_vanilla import ARMC  # noqa: E402
from panel_vanilla import collect as collect_vanilla
from schematic import place  # noqa: E402

SITES = ("M0097", "M0255", "M0315")

# What each target actually is, checked against the PDB entries rather than
# inferred from the ligand code -- 1EY3's ligand is DAK, which is NOT
# dihydroxyacetone kinase but 4-(N,N-dimethylamino)cinnamoyl-CoA bound to an
# enoyl-CoA hydratase. The three are deliberately unalike: a metal-dependent
# hydrolase, a cofactor-dependent oxidoreductase and an acyl-CoA enzyme.
ENZYME = {
    "M0097": ("cytidine deaminase", "Zn$^{2+}$, nucleoside analogue"),
    "M0255": ("alcohol dehydrogenase", "NADH, acetate"),
    "M0315": ("enoyl-CoA hydratase", "acyl-CoA analogue"),
}
PDBID = {"M0097": "1ctt", "M0255": "1mg5", "M0315": "1ey3"}


def panel_target(ax, site):
    """The backbone, coloured at k=4, motif as sticks, ligand in grey."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # 74 pt, not 88, and pushed up. Giving every row its own x-axis shortened
    # these cells, and at the old size the fold sat on its own caption.
    place(ax, f"target_{site}", (0.56, 0.68), 74)
    name, cofactor = ENZYME[site]
    ax.text(0.56, 0.17, name, ha="center", va="center", fontsize=fs(7))
    ax.text(0.56, 0.04, f"{cofactor} \u00b7 {PDBID[site]}", ha="center",
            va="center", fontsize=fs(6.2), color=PALETTE["gray"])


def panel_move(ax, v, p, site, show_x):
    """Vanilla -> post-trained for ONE backbone, measured on recombinants."""
    ax.annotate("", xy=(p.subs, p.success), xytext=(v.subs, v.success),
                arrowprops=dict(arrowstyle="-|>", color=INK["ghost"], lw=1.3,
                                shrinkA=7, shrinkB=9))
    for arm, r in (("vanilla", v), ("post", p)):
        ax.plot(r.subs, r.success, "o", ms=8, zorder=4,
                mfc=ARMC[arm], mec=ARMC[arm], mew=1.5)
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
    # The trained-at-k arm only. The post-hoc arm -- one k=4 policy re-cut at
    # every k -- was drawn here in grey and has been dropped to simplify.
    #
    # Worth knowing what went with it: the grey line was the control. Alone,
    # this panel shows that retention falls as the library is cut finer; it
    # no longer shows that training AT that k is what holds it up. That
    # comparison now survives only as numbers in the caption (+8.3 points at
    # k=2 rising to +32.2 at k=36), and ame_fragk.py still plots both.
    for cond in ("trained",):
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
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if show_x:
        ax.set_xlabel("fragments per design, $k$", fontsize=fs(7.5))
    else:
        ax.tick_params(labelbottom=False)
    ax.tick_params(labelsize=fs(7))


def panel_cost(ax, van):
    """Folds to reach a library of size S: brute-force filtering vs training.

    One grey line per backbone, because the brute-force bill depends on that
    backbone's vanilla pass rate and those differ by more than 2x. The
    training line is flat: the enrichment lives in the policy, so it does not
    get more expensive as the library gets bigger.
    """
    S = np.logspace(2, 12, 200)
    for site in SITES:
        p = van.loc[(site, "vanilla")].success / 100
        ax.loglog(S, brute_force_folds(S, p), "-", color=INK["muted"],
                  lw=1.6, zorder=2)
        ax.plot([crossover(p)], [TRAIN_FOLDS], "o", ms=5,
                mfc="white", mec=INK["muted"], mew=1.3, zorder=4)
    ax.axhline(TRAIN_FOLDS, color=ARMC["post"], lw=2.0, zorder=3)
    ax.set_xlim(1e2, 1e12)
    ax.set_ylim(1e3, 1e13)
    ax.set_xlabel("library members that fold", fontsize=fs(7.5))
    ax.set_ylabel("folds required", fontsize=fs(7.5))
    ax.tick_params(labelsize=fs(7))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # Labels go where the OTHER line is not. The brute-force arms climb away
    # to the right, so the flat training line is captioned out there; the
    # break-even note sits below the flat line, left of where the arms cross
    # it.
    ax.text(2e7, TRAIN_FOLDS * 2.4, "post-training, once "
            f"({TRAIN_FOLDS // 1000}k folds)", fontsize=fs(6.8),
            color=ARMC["post"], va="bottom")
    ax.text(2e9, 2e12, "fold every candidate", fontsize=fs(6.8),
            color=PALETTE["gray"], ha="right", va="top")
    ax.text(2e5, 6e3, "break-even", fontsize=fs(6.4),
            color=PALETTE["gray"], va="center")


def main():
    rng = np.random.default_rng(0)
    plt.rcParams["text.usetex"] = False

    d = pd.DataFrame(collect_trained() + collect_posthoc())
    if not len(d):
        sys.exit("no cells found -- is experiments/ame mounted?")
    d = add_diversity(d, 200, rng)
    van = collect_vanilla(rng=np.random.default_rng(0)).set_index(
        ["site", "arm"])

    # 8.8 in, not 10.2. The page's text block is 9 in tall (letter, 1 in
    # margins) and this figure prints at its authored size, so anything taller
    # simply does not fit on a float page -- adding the cost row had pushed it
    # over. The rows are trimmed rather than the figure scaled, because
    # scaling would take the type below palette.MIN_FONT.
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 8.95))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.86, 2.10, 0.72],
                             hspace=0.34, left=0.095, right=0.985,
                             top=0.935, bottom=0.05)
    ax_a = fig.add_subplot(outer[0])
    panel_loop(ax_a, np.random.default_rng(3))

    # One row per backbone. The structure column is narrower than the two
    # plots: it carries no axes, and at equal width the folds would dominate
    # a row whose argument is the two plots beside them.
    grid = outer[1].subgridspec(3, 3, width_ratios=[0.78, 1.0, 1.0],
                                hspace=0.52, wspace=0.46)
    ax_v0 = ax_k0 = None
    first = {}
    for i, site in enumerate(SITES):
        ax_s = fig.add_subplot(grid[i, 0])
        panel_target(ax_s, site)

        ax_v = fig.add_subplot(grid[i, 1], sharex=ax_v0, sharey=ax_v0)
        ax_v0 = ax_v0 or ax_v
        # Every row gets its own x-axis, not just the bottom one: the rows
        # are three separate backbones rather than facets of one series, so
        # each should be readable without tracking a shared axis two rows
        # down.
        panel_move(ax_v, van.loc[(site, "vanilla")], van.loc[(site, "post")],
                   site, show_x=True)

        ax_k = fig.add_subplot(grid[i, 2], sharex=ax_k0, sharey=ax_k0)
        ax_k0 = ax_k0 or ax_k
        panel_k(ax_k, d, site, show_x=True)
        if i == 0:
            first = dict(s=ax_s, v=ax_v, k=ax_k)

    # One legend per plot column, on the top row only.
    first["v"].legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=7, mfc=ARMC["vanilla"],
               mec=ARMC["vanilla"], label="vanilla, $T$ pooled"),
        Line2D([], [], marker="o", ls="none", ms=7, mfc=ARMC["post"],
               mec=ARMC["post"], label="post-trained")],
        frameon=False, fontsize=fs(6.4), loc="lower right",
        handletextpad=0.4, labelspacing=0.25)
    first["k"].legend(handles=[
        Line2D([], [], color=C["trained"], lw=1.8, label="trained at $k$")],
        frameon=False, fontsize=fs(6.4), loc="lower left",
        handletextpad=0.6, labelspacing=0.25)

    # (c) the fold bill. Full width but short: it is one crossing of two
    # lines, and the reader needs the decades on the axes more than area.
    cost = outer[2].subgridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.30)
    ax_c = fig.add_subplot(cost[0, 0])
    panel_cost(ax_c, van)

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

    # Two letters, not four. The three columns are one argument read across
    # a row -- this target, what post-training did to it, how far that held --
    # so lettering them separately would promise three findings per row
    # instead of one. The caption names the columns in words.
    y_row = max(inv.transform((0, extent(a).y1))[1]
                for a in (first["s"], first["v"], first["k"]))
    for ax, lab, y in ((ax_a, "a", inv.transform((0, extent(ax_a).y1))[1]),
                       (first["s"], "b", y_row),
                       (ax_c, "c", inv.transform((0, extent(ax_c).y1))[1])):
        x = inv.transform((extent(ax).x0, 0))[0]
        fig.text(x, y + 0.006, lab, fontsize=PANEL_LABEL_PT,
                 fontweight="bold", va="bottom", ha="left")

    save(fig, "fig2_libraries", dpi=900, tight=False)


if __name__ == "__main__":
    main()
