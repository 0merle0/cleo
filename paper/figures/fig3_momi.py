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


def only(df, what):
    """-> `df` gated to LIB, loudly.

    Figure 3 is about the SECOND library. The two share 5 parts out of 3,629
    and their design sets are disjoint, so anything pooled is wrong rather
    than merely noisy -- when panel (b) was drawn over both, slots 2 and 3
    came out at 8.1 and 7.0 mean mutations against the true 4.4 and 4.4.
    Every table this figure reads goes through here.
    """
    col = "library_name" if "library_name" in df.columns else "library"
    if col not in df.columns:
        raise SystemExit(f"{what}: no library column to gate on")
    g = df[df[col] == LIB]
    if not len(g):
        raise SystemExit(f"{what}: nothing left after gating to {LIB}")
    return g
SLOTS = [(1, 40), (45, 84), (89, 126), (131, 163), (168, 209)]
COLORS = list(FRAGMENT) + [PALETTE["purple"]]
K = 5


def panel_design(ax_fold, ax_site):
    """The parent by slot, and the active site split across two of them."""
    for ax in (ax_fold, ax_site):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    # Each column is ~220 pt wide, so these are close to filling it. The
    # row ratio is set to match rather than leaving the images floating in
    # an oversized cell.
    place(ax_fold, "momi_fold", (0.5, 0.62), 186)
    place(ax_site, "momi_site", (0.5, 0.62), 196)
    ax_fold.text(0.5, 0.045, "G20 parent, cut into $k=5$", ha="center",
                 fontsize=fs(7))
    ax_site.text(0.5, 0.045, "catalytic triad, split across slots 2 and 3",
                 ha="center", fontsize=fs(7))


def panel_distance(axes, parts):
    """One histogram per slot: how far its parts sit from the G20 part.

    Every part the library carries, not just the ones that were built into a
    measured design -- the question is what the slot's menu looked like.
    Counts, not density, so a slot offering fewer options reads as offering
    fewer options.
    """
    sub = only(parts, "fragment parts")
    sub = sub[sub.is_parent == 0]
    hi = int(sub.n_mut_from_g20.max())
    for i, ax in enumerate(axes):
        v = sub[sub.slot == i + 1].n_mut_from_g20.to_numpy()
        ax.hist(v, bins=np.arange(-0.5, hi + 1.5), color=COLORS[i],
                lw=0, zorder=3)
        ax.axvline(np.median(v), color="black", lw=1.2, zorder=4)
        lo_r, hi_r = SLOTS[i]
        ax.set_title(f"slot {i+1}\n{lo_r}\u2013{hi_r}", fontsize=fs(6.6),
                     pad=3)
        ax.set_xlim(0, hi + 1)
        ax.set_xlabel("mutations", fontsize=fs(6.8))
        ax.tick_params(labelsize=fs(6.4))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.set_ylabel("parts", fontsize=fs(7.5))
        else:
            ax.tick_params(labelleft=False)


def panel_activity(ax, d):
    """Activity by round, LINEAR, against G20 itself."""
    s = only(d, "activity")
    s = s[s.sample_type == "sample"]
    rounds = sorted(s.round_number.unique())
    shades = [SEQUENTIAL[1], SEQUENTIAL[2], SEQUENTIAL[3], SEQUENTIAL[5]]
    rng = np.random.default_rng(0)
    for i, r in enumerate(rounds):
        v = s[s.round_number == r].g20_norm_rate.to_numpy()
        x = i + 1 + rng.uniform(-0.27, 0.27, len(v))
        ax.plot(x, v, "o", ms=1.7, mfc=shades[i], mec="none", alpha=0.45,
                zorder=2)
        ax.plot([i + 0.64, i + 1.36], [np.median(v)] * 2, "-", color="black",
                lw=1.8, zorder=5)
    ax.axhline(1, color=PALETTE["gray"], lw=1, ls=":", zorder=1)
    ax.set_xticks(range(1, len(rounds) + 1))
    ax.set_xticklabels([f"{int(r)}" for r in rounds])
    ax.set_xlabel("round", fontsize=fs(7.5))
    ax.set_ylabel("activity ($\\times$ G20)", fontsize=fs(7.5))
    ax.tick_params(labelsize=fs(7))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.03, 0.97, "dotted: G20\nbar: median", ha="left", va="top",
            transform=ax.transAxes, fontsize=fs(6.4), color=PALETTE["gray"])


def panel_dist_activity(ax, d, parent):
    """Activity against how far the whole design sits from G20."""
    s = only(d, "distance/activity")
    s = s[s.sample_type == "sample"].copy()
    s["nmut"] = [sum(1 for a, b in zip(q, parent) if a != b)
                 for q in s.sequence]
    shades = {0: SEQUENTIAL[1], 1: SEQUENTIAL[2],
              2: SEQUENTIAL[3], 3: SEQUENTIAL[5]}
    for r in sorted(s.round_number.unique()):
        g = s[s.round_number == r]
        ax.plot(g.nmut, g.g20_norm_rate, "o", ms=1.8, mfc=shades[int(r)],
                mec="none", alpha=0.5, zorder=2, label=f"round {int(r)}")
    ax.axhline(1, color=PALETTE["gray"], lw=1, ls=":", zorder=1)
    ax.set_xlabel("mutations from G20 (whole sequence)", fontsize=fs(7.5))
    ax.set_ylabel("activity ($\\times$ G20)", fontsize=fs(7.5))
    ax.tick_params(labelsize=fs(7))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    leg = ax.legend(frameon=False, fontsize=fs(6.4), loc="upper left",
                    handletextpad=0.2, labelspacing=0.2, borderpad=0.2)
    for h in leg.legend_handles:
        h.set_markersize(5)
        h.set_alpha(1)


def main():
    plt.rcParams["text.usetex"] = False
    d = pd.read_csv(HERE / "data" / "momi_libraries.csv")
    parts = pd.read_csv(HERE / "data" / "momi_fragment_parts.csv")
    ctrl = d[d.sample_type == "positive_control"]
    parent = ctrl.sequence.iloc[0]

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 8.2))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.20, 0.60, 0.84],
                             hspace=0.52, left=0.095, right=0.985,
                             top=0.955, bottom=0.06)

    top = outer[0].subgridspec(1, 2, wspace=0.04)
    ax_f = fig.add_subplot(top[0, 0])
    ax_s = fig.add_subplot(top[0, 1])
    panel_design(ax_f, ax_s)

    # (b) one histogram per slot, sharing a y-axis so the heights compare
    mid = outer[1].subgridspec(1, K, wspace=0.14)
    axes_b = []
    for i in range(K):
        ax = fig.add_subplot(mid[0, i], sharey=axes_b[0] if axes_b else None)
        axes_b.append(ax)
    panel_distance(axes_b, parts)

    bot = outer[2].subgridspec(1, 2, width_ratios=[0.92, 1.0], wspace=0.34)
    ax_a = fig.add_subplot(bot[0, 0])
    ax_m = fig.add_subplot(bot[0, 1])
    panel_activity(ax_a, d)
    panel_dist_activity(ax_m, d, parent)

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

    y_bot = max(inv.transform((0, extent(a).y1))[1] for a in (ax_a, ax_m))
    for ax, lab, y in ((ax_f, "a", inv.transform((0, extent(ax_f).y1))[1]),
                       (axes_b[0], "b",
                        inv.transform((0, extent(axes_b[0]).y1))[1]),
                       (ax_a, "c", y_bot), (ax_m, "d", y_bot)):
        x = inv.transform((extent(ax).x0, 0))[0]
        fig.text(x, y + 0.006, lab, fontsize=PANEL_LABEL_PT,
                 fontweight="bold", va="bottom", ha="left")

    save(fig, "fig3_momi", dpi=900, tight=False)


if __name__ == "__main__":
    main()
