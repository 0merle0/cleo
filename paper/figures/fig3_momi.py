#!/usr/bin/env python3
"""Figure 3 --- the PETase library, how it was built and what it did.

Focused on the SECOND big library. The first is in the same table and is
shown only where it is the honest comparison; the two share 5 parts out of
3,629, so they are separate designs rather than a continuation.

    (a) the G20 parent coloured by its five slots, and the active site --
        the catalytic triad is SPLIT, Ser66 on slot 2 and His107/Asp122 on
        slot 3, so recombination reassembles it across a cut
    (b) how far each slot's parts sit from the G20 part they replace
    (c) activity by round, against G20 itself

Fragment boundaries are measured, not assumed; see momi_fragments.py.

    uv run python paper/figures/fig3_momi.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figio import save  # noqa: E402
from palette import (FRAGMENT, INK, PAGE_WIDTH_IN, PALETTE,  # noqa: E402
                     PANEL_LABEL_PT, SEQUENTIAL, fs)
from schematic import place  # noqa: E402

LIB = "second_big_library"
SLOTS = [(1, 40), (45, 84), (89, 126), (131, 163), (168, 209)]
COLORS = list(FRAGMENT) + [PALETTE["purple"]]
K = 5


def panel_design(ax_fold, ax_site):
    """The parent by slot, and the active site split across two of them."""
    for ax in (ax_fold, ax_site):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    place(ax_fold, "momi_fold", (0.5, 0.60), 132)
    place(ax_site, "momi_site", (0.5, 0.60), 142)
    ax_fold.text(0.5, 0.10, "G20 parent, cut into $k=5$", ha="center",
                 fontsize=fs(7))
    ax_site.text(0.5, 0.10, "catalytic triad, split across slots 2 and 3",
                 ha="center", fontsize=fs(7))


def panel_distance(ax, parts):
    """How far each slot's parts sit from the G20 part they replace."""
    sub = parts[(parts.is_parent == 0)
                & parts.libraries.str.contains(LIB)]
    for i in range(K):
        v = sub[sub.slot == i + 1].n_mut_from_g20.to_numpy()
        if not len(v):
            continue
        # a horizontal violin per slot, drawn from the counts themselves
        lo, hi = int(v.min()), int(v.max())
        edges = np.arange(lo - 0.5, hi + 1.5)
        h, _ = np.histogram(v, bins=edges)
        h = h / h.max() * 0.38
        cen = (edges[:-1] + edges[1:]) / 2
        ax.fill_betweenx(cen, i + 1 - h, i + 1 + h, color=COLORS[i],
                         alpha=0.85, lw=0, zorder=3)
        ax.plot([i + 1], [np.median(v)], "o", ms=4.5, mfc="white",
                mec=PALETTE["gray"], mew=1.2, zorder=5)
    ax.set_xticks(range(1, K + 1))
    ax.set_xticklabels([f"{lo}–{hi}" for lo, hi in SLOTS],
                       fontsize=fs(6.4))
    ax.set_xlabel("slot (residues)", fontsize=fs(7.5))
    ax.set_ylabel("mutations from the G20 part", fontsize=fs(7.5))
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=fs(7))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # Top RIGHT: slot 1 reaches 17 mutations and was sitting under this.
    ax.text(0.98, 0.97, "white dot: median", transform=ax.transAxes,
            fontsize=fs(6.4), color=PALETTE["gray"], va="top", ha="right")


def panel_activity(ax, d):
    """Activity by round, on a log axis, against G20 itself."""
    s = d[(d.library_name == LIB) & (d.sample_type == "sample")]
    rounds = sorted(s.round_number.unique())
    shades = [SEQUENTIAL[1], SEQUENTIAL[2], SEQUENTIAL[3], SEQUENTIAL[5]]
    rng = np.random.default_rng(0)
    for i, r in enumerate(rounds):
        v = s[s.round_number == r].g20_norm_rate.to_numpy()
        v = v[v > 0]
        x = i + 1 + rng.uniform(-0.26, 0.26, len(v))
        ax.plot(x, v, "o", ms=1.6, mfc=shades[i], mec="none", alpha=0.45,
                zorder=2)
        ax.plot([i + 0.66, i + 1.34], [np.median(v)] * 2, "-",
                color="black", lw=1.8, zorder=5)
    ax.axhline(1, color=PALETTE["gray"], lw=1, ls=":", zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(rounds) + 1))
    ax.set_xticklabels([f"{int(r)}" for r in rounds])
    ax.set_xlabel("round", fontsize=fs(7.5))
    ax.set_ylabel("activity ($\\times$ G20)", fontsize=fs(7.5))
    ax.tick_params(labelsize=fs(7))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.98, 0.03, "dotted: G20 parent\nbar: median", ha="right",
            va="bottom", transform=ax.transAxes, fontsize=fs(6.4),
            color=PALETTE["gray"])


def main():
    plt.rcParams["text.usetex"] = False
    d = pd.read_csv(HERE / "data" / "momi_libraries.csv")
    parts = pd.read_csv(HERE / "data" / "momi_fragment_parts.csv")

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 5.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.42,
                             left=0.095, right=0.985, top=0.945, bottom=0.075)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.05)
    ax_f = fig.add_subplot(top[0, 0])
    ax_s = fig.add_subplot(top[0, 1])
    panel_design(ax_f, ax_s)

    bot = outer[1].subgridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.36)
    ax_d = fig.add_subplot(bot[0, 0])
    ax_a = fig.add_subplot(bot[0, 1])
    panel_distance(ax_d, parts)
    panel_activity(ax_a, d)

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

    y_bot = max(inv.transform((0, extent(a).y1))[1] for a in (ax_d, ax_a))
    for ax, lab, y in ((ax_f, "a", inv.transform((0, extent(ax_f).y1))[1]),
                       (ax_d, "b", y_bot), (ax_a, "c", y_bot)):
        x = inv.transform((extent(ax).x0, 0))[0]
        fig.text(x, y + 0.006, lab, fontsize=PANEL_LABEL_PT,
                 fontweight="bold", va="bottom", ha="left")

    save(fig, "fig3_momi", dpi=900, tight=False)


if __name__ == "__main__":
    main()
