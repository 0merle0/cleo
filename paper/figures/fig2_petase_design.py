#!/usr/bin/env python3
"""Figure 2 --- the PETase, and what post-training did to it.

Four-figure scheme. Figure 1 is the method and the benchmark, this is the
design, Figure 3 the platform, Figure 4 the search. See figure_plan_story.md.

Data panels, built from the runs and libraries the paper shipped:

    (a) every reward term over training, naive (step 0) -> post-trained
    (b) the same, as the collapse in batch spread
    (c) distance to the top-5 of library 2, both libraries

The structure panels -- the theozyme-to-momi funnel and the superposed
active-site clouds -- stay placeholders. They are renders, not data plots.

    uv run python paper/figures/fig2_petase_design.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import petase_design as PD  # noqa: E402
from blockout import slot  # noqa: E402
from figio import save  # noqa: E402
from palette import INK, PAGE_WIDTH_IN, PALETTE, SEQUENTIAL, fs  # noqa: E402

NAIVE = INK["muted"]
POST = SEQUENTIAL[3]
LIBC = {PD.LIB2: INK["muted"], PD.LIB3: SEQUENTIAL[3]}
LIBL = {PD.LIB2: "library 2", PD.LIB3: "library 3"}


def _tidy(ax, ylab=None, xlab=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color(INK["muted"])
        ax.spines[k].set_linewidth(0.8)
    ax.tick_params(labelsize=fs(7), colors="black", width=0.8, length=2.5)
    if ylab:
        ax.set_ylabel(ylab, fontsize=fs(7.5))
    if xlab:
        ax.set_xlabel(xlab, fontsize=fs(7.5))


# ================================================================ panel (a)

def panel_training(fig, gs, run="refw5", ncol=4):
    """Every reward term across training, with the batch min-max band.

    STEP 0 IS NAIVE ProteinMPNN. The policy starts from the stock decoder, so
    the step-0 rollout is a stock sample scored on exactly the reward the run
    then optimises -- the naive comparison is the run's own first step, not a
    matched baseline that has to be argued for.

    The band is batch MIN-MAX over the 16 sequences of one rollout, not a
    confidence interval. Its collapse is the "reduces variance" claim
    measured: His NE2 - Ser OG spans 32.1 A at step 0 and 2.5 A at the end.

    The dashed line is the reward's own target bound from the run config --
    the value at which that term stops earning more. Plotting it keeps the
    panel honest about what was asked for versus what was achieved.
    """
    d = PD.load_training(run)
    terms = [t for t in PD.TERMS if f"{t[0]}_batch_mean" in d.columns]
    nrow = int(np.ceil(len(terms) / ncol))
    grid = gs.subgridspec(nrow, ncol, wspace=0.42, hspace=0.62)

    for i, (key, label, lo, hi, mode) in enumerate(terms):
        ax = fig.add_subplot(grid[i // ncol, i % ncol])
        m = d[f"{key}_batch_mean"]
        locol, hicol = f"{key}_batch_min", f"{key}_batch_max"
        if locol in d.columns and hicol in d.columns:
            ax.fill_between(d.step, d[locol], d[hicol], color=POST,
                            alpha=0.16, lw=0, zorder=2)
        ax.plot(d.step, m, color=POST, lw=1.1, zorder=3)

        # Naive = the first rollout, marked rather than described.
        ax.plot([0], [m.iloc[0]], "o", ms=4, mfc="white", mec=NAIVE, mew=1.2,
                zorder=5)

        target = hi if mode == "max" else lo
        ax.axhline(target, color=PALETTE["red"], lw=0.8, ls=(0, (3, 2)),
                   zorder=1)

        ax.set_title(label, fontsize=fs(7), pad=3)
        _tidy(ax)
        ax.set_xlim(-6, d.step.max())
        ax.set_xticks([0, 100, 200])
        if i // ncol == nrow - 1:
            ax.set_xlabel("Training step", fontsize=fs(7))


# ================================================================ panel (b)

def panel_spread(ax, runs=("refw5", "refw8"), final=20):
    """The collapse in batch spread, one row per reward term.

    (a) shows this as eight narrowing bands and the eye has to integrate
    them; this is the same thing as one number per term, so the claim can be
    read instead of inferred. Spread is batch max minus min, naive against
    the mean of the last 20 steps.

    Log x, because the terms live in different units -- angstroms, degrees,
    a dimensionless confidence -- and the comparable quantity is the FACTOR
    each one shrinks by, not its size.

    `Distance to reference` is absent: it logs only a batch mean, so it has
    no spread to report. Showing it at zero would read as "no variance".
    """
    s = PD.training_summary(runs[0], final=final).dropna(subset=["spread_naive"])
    ys = np.arange(len(s))[::-1]
    for y, (_, r) in zip(ys, s.iterrows()):
        ax.plot([r.spread_post, r.spread_naive], [y, y], color=INK["ghost"],
                lw=1.1, zorder=2, solid_capstyle="round")
        ax.plot(r.spread_naive, y, "o", ms=4.5, mfc=NAIVE, mec=NAIVE,
                zorder=4)
        ax.plot(r.spread_post, y, "o", ms=4.5, mfc=POST, mec=POST, zorder=4)
        ax.text(r.spread_post * 0.78, y, f"{r.spread_naive / r.spread_post:.0f}x",
                fontsize=fs(6), color=POST, ha="right", va="center", zorder=5)

    ax.set_yticks(ys)
    ax.set_yticklabels(s.label.tolist(), fontsize=fs(6.5))
    ax.set_xscale("log")
    ax.set_xlim(0.04, 400)
    ax.set_ylim(-0.7, len(s) - 0.3)
    ax.tick_params(axis="y", length=0)
    _tidy(ax, xlab="Batch spread (max - min)")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ms=4.5, mfc=NAIVE, mec=NAIVE, ls="none",
               label="naive (step 0)"),
        Line2D([], [], marker="o", ms=4.5, mfc=POST, mec=POST, ls="none",
               label="post-trained")],
        fontsize=fs(6.5), loc="lower right", frameon=False,
        handlelength=0.8, borderpad=0.0, labelspacing=0.22,
        handletextpad=0.4)


# ================================================================ panel (c)

def panel_topk(ax, dist=None, bins=36):
    """Distance to the top-5 of library 2, for both libraries.

    x is the MINIMUM Hamming distance over the five best-measured library-2
    variants -- the set library 3 was designed around. Minimum, not mean: a
    design built around reference 3 is close to the library's target even if
    it is far from the other four, and a mean would score it as distant.

    READ THIS PANEL CAREFULLY, because it does not say what the draft says.
    Library 3 is NOT centred closer on the top-5 than library 2 is -- the
    medians are 17 and 19 mutations. What library 3 is, is narrower: its IQR
    is 8 mutations against 13, and its most distant member is 42 against 64.
    The aiming shows up as a tighter distribution, not a shifted one.

    Library 2's left edge at 0 is the references themselves: they ARE library
    2 members, so that arm is guaranteed to touch zero and the comparison at
    the extreme left is not meaningful.
    """
    if dist is None:
        dist, _ = PD.distance_to_topk()
    hi = int(dist.dmin.max()) + 1
    edges = np.linspace(0, hi, bins + 1)
    for lib in (PD.LIB2, PD.LIB3):
        v = dist[dist.library == lib].dmin.to_numpy()
        ax.hist(v, bins=edges, density=True, color=LIBC[lib], alpha=0.5,
                edgecolor=LIBC[lib], linewidth=0.7, zorder=2,
                label=f"{LIBL[lib]}  (n={len(v)})")
        ax.axvline(np.median(v), color=LIBC[lib], lw=1.0, ls=(0, (3, 2)),
                   zorder=3)

    _tidy(ax, ylab="Fraction of library",
          xlab="Minimum distance to the top-5 of library 2 (mutations)")
    ax.set_xlim(0, hi)
    ax.legend(fontsize=fs(6.5), loc="upper right", frameon=False,
              handlelength=1.0, borderpad=0.0, labelspacing=0.25,
              handletextpad=0.5)


# ================================================================== assembly

def main():
    dist, _ = PD.distance_to_topk()

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 7.6))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.30, 1.00, 1.00],
                             left=0.085, right=0.985, top=0.955,
                             bottom=0.055, hspace=0.52)

    panel_training(fig, outer[0])

    mid = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30)
    panel_spread(fig.add_subplot(mid[0, 0]))
    panel_topk(fig.add_subplot(mid[0, 1]), dist)

    bot = outer[2].subgridspec(1, 2, wspace=0.10)
    slot(fig.add_subplot(bot[0, 0]), "d",
         "Where momi came from: theozyme to an assayed enzyme",
         "have", "structures/png/momi_fold.png, momi_site.png",
         lines=[
             "PET polymer docked to catalytic residues -> RFD2 backbone ->",
             "ProteinMPNN -> geometric filters -> screen -> momi",
             "the filters belong ON this panel: Ser OG - His NE2 < 3 A,",
             "oxyanion < 3.5 A, Asp O(-) - His N < 3.5 A, site pLDDT > 80",
             "a render, not a data plot",
         ])
    slot(fig.add_subplot(bot[0, 1]), "e",
         "Catalytic geometry, naive vs post-trained library",
         "collect", "needs both arms regenerated from the checkpoints",
         lines=[
             "superposed active-site atoms, two clouds",
             "GAP: per-design naive values do not exist. The training log",
             "keeps only batch mean/min/max, and the folds were cleaned --",
             "62 CIFs survive across 212 steps of refw5.",
             "The checkpoints DO survive, so this is a generate-and-fold",
             "job: ~250 sequences from the stock decoder and ~250 from",
             "*_step_0200.pt, folded best-of-5 and scored.",
         ])

    for letter, (x, y) in zip("abc", [(0.008, 0.975), (0.008, 0.545),
                                      (0.505, 0.545)]):
        fig.text(x, y, f"({letter})", fontsize=fs(9), fontweight="bold",
                 ha="left", va="top")

    save(fig, "fig2_petase_design", dpi=900, tight=False, subdir="main")


if __name__ == "__main__":
    main()
