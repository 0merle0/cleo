#!/usr/bin/env python3
"""Figure 2 --- the PETase, and what post-training did to it.

Four-figure scheme. Figure 1 is the method and the benchmark, this is the
design, Figure 3 the platform, Figure 4 the search. See figure_plan_story.md.

Data panels, built from the runs and libraries the paper shipped:

    (a) every reward term, step 0 vs end, side by side
    (b) the same, as the collapse in batch spread
    (d) the full training trajectories, kept as a separate panel
    (c) each slot's part menu, by distance from the anchor set

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
from palette import (FRAGMENT, INK, PAGE_WIDTH_IN, PALETTE,  # noqa: E402
                     SEQUENTIAL, fs)

# One colour per fragment slot, the same five fig3_momi uses, so a slot
# keeps its identity between the two figures.
SLOTC = list(FRAGMENT) + [PALETTE["purple"]]

NAIVE = INK["muted"]
# Same blue as Figure 1 uses for this method (fig1_method.METHOD).
POST = SEQUENTIAL[4]
LIBC = {PD.LIB2: INK["muted"], PD.LIB3: SEQUENTIAL[4]}
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


def panel_paired(ax, run="refw5", final=20):
    """Every reward term at step 0 and at the end, side by side.

    One column per term, naive on the left of the pair and post-trained on
    the right, so each term is read as a pair and the whole reward is read
    as a row.

    y is the term's REWARD-NORMALISED value: each metric mapped onto [0, 1]
    through the lower and upper bounds in the run config, flipped where the
    mode is min, so that up is always better. That is the only way these
    eight share an axis -- they are angstroms, degrees, a dimensionless
    confidence and a mutation count -- and it is not a plotting convenience:
    it is literally the quantity the reward sums.

    The bar through each point is the batch MIN-MAX, the same range (a)
    draws as a band. So the panel carries both halves of the claim at once:
    the point rising is better positioning, the bar shortening is lower
    variance.

    THIS IS NOT A PER-DESIGN STRIP PLOT, and it cannot be made into one from
    what is on disk. The training log stores batch mean/min/max per step and
    nothing finer, and the folds those came from were cleaned -- 62 CIFs
    survive across 212 steps. What DOES survive is the sequences: the AF3
    input JSONs are intact, 32 at step 0 and 32 at step 200. So a real strip
    plot is a fold-and-score job over 64 sequences (320 predictions at
    best-of-5), not a re-run of training.
    """
    d = PD.load_training(run)
    terms = [t for t in PD.TERMS if f"{t[0]}_batch_mean" in d.columns]

    def norm(v, lo, hi, mode):
        z = (np.asarray(v, float) - lo) / (hi - lo)
        z = np.clip(z, 0.0, 1.0)
        # mode "avg" targets the middle of the window rather than an end, so
        # its score is distance from the centre, not height up the ramp.
        if mode == "min":
            return 1.0 - z
        if mode == "avg":
            return 1.0 - 2.0 * np.abs(z - 0.5)
        return z

    xs = np.arange(len(terms))
    off = 0.17
    for i, (key, label, lo, hi, mode) in enumerate(terms):
        m = d[f"{key}_batch_mean"]
        locol, hicol = f"{key}_batch_min", f"{key}_batch_max"
        has_band = locol in d.columns and hicol in d.columns

        for j, (when, colour) in enumerate(((0, NAIVE), (1, POST))):
            x = xs[i] + (-off if j == 0 else off)
            if when == 0:
                mu = float(m.iloc[0])
                band = ((float(d[locol].iloc[0]), float(d[hicol].iloc[0]))
                        if has_band else None)
            else:
                mu = float(m.tail(final).mean())
                band = ((float(d[locol].tail(final).mean()),
                         float(d[hicol].tail(final).mean()))
                        if has_band else None)
            if band is not None:
                b0, b1 = sorted(norm(band, lo, hi, mode))
                ax.plot([x, x], [b0, b1], color=colour, lw=1.0, zorder=3,
                        solid_capstyle="round", alpha=0.65)
            ax.plot(x, norm(mu, lo, hi, mode), "o", ms=4.6, mfc=colour,
                    mec="white", mew=0.8, zorder=5)

        a = norm(float(m.iloc[0]), lo, hi, mode)
        b = norm(float(m.tail(final).mean()), lo, hi, mode)
        ax.annotate("", xy=(xs[i] + off - 0.045, b),
                    xytext=(xs[i] - off + 0.045, a),
                    arrowprops=dict(arrowstyle="-|>", color=INK["ghost"],
                                    lw=0.9, shrinkA=3, shrinkB=4), zorder=2)

    ax.set_xticks(xs)
    ax.set_xticklabels([t[1] for t in terms], fontsize=fs(6.4), rotation=30,
                       ha="right")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(terms) - 0.4)
    ax.set_ylim(-0.04, 1.08)
    _tidy(ax, ylab="Reward-normalised score")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ms=4.6, mfc=NAIVE, mec="white", mew=0.8,
               ls="none", label="naive (step 0)"),
        Line2D([], [], marker="o", ms=4.6, mfc=POST, mec="white", mew=0.8,
               ls="none", label="post-trained")],
        fontsize=fs(6.5), loc="lower left", frameon=False,
        handlelength=0.8, borderpad=0.0, labelspacing=0.22,
        handletextpad=0.4)


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

def panel_slotdist(fig, gs, menu=None):
    """What each fragment slot's menu looks like, by distance from the anchors.

    One histogram per slot, in the style clean.pdf uses for its per-fragment
    panel: the slot is the unit the library is ordered in -- a well of DNA
    parts -- and how far it was allowed to roam was decided five times, once
    per slot, not once for the design.

    x is the `distN` tag the part was generated and named with: N mutations
    from the corresponding part of the anchor set, the three sequences in
    `config/design/denovo_petase.yaml` that the reward tethers to at weight
    10. The tags are the design record and are taken as authoritative.

    PARTS, not designs, and counts, not density. A part built into 200
    assembled designs is still one part in the menu; counting designs would
    reweight the menu by how often the sampler drew each part, which is a
    fact about the sampler rather than about the library that was ordered.

    THE MENU IS UNIFORM BY DESIGN and the panel shows that: slot 1 carries
    26-27 parts at every distance from 1 to 14, the other four carry 38-50
    at every distance out to 8 or 9. The flat shape is the point. What
    differs between slots is the REACH -- slot 1 runs to 14 mutations while
    slots 3 and 4 stop at 8 -- and that is what to read off this panel.
    """
    if menu is None:
        menu = PD.part_menu()
    slots = sorted(menu.slot.unique())
    grid = gs.subgridspec(1, len(slots), wspace=0.16)
    hi = int(menu.dist.max())
    axes = []
    for j, i in enumerate(slots):
        ax = fig.add_subplot(grid[0, j], sharey=axes[0] if axes else None)
        axes.append(ax)
        v = menu[menu.slot == i].dist.to_numpy()
        ax.hist(v, bins=np.arange(-0.5, hi + 1.5), color=SLOTC[i - 1],
                lw=0, zorder=3)
        ax.set_title(f"slot {i}", fontsize=fs(6.8), pad=3)
        ax.set_xlim(-0.6, hi + 0.6)
        ax.set_xticks([t for t in (0, 5, 10, 15) if t <= hi])
        ax.set_xlabel("mutations", fontsize=fs(6.8))
        ax.tick_params(labelsize=fs(6.4))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for k in ("left", "bottom"):
            ax.spines[k].set_color(INK["muted"])
            ax.spines[k].set_linewidth(0.8)
        if j == 0:
            ax.set_ylabel("Parts in the slot", fontsize=fs(7.5))
        else:
            ax.tick_params(labelleft=False)
    return axes


# ================================================================== assembly

def main():
    menu = PD.part_menu()

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 7.6))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.15, 0.86, 1.00],
                             left=0.085, right=0.985, top=0.955,
                             bottom=0.055, hspace=0.72)

    top = outer[0].subgridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.30)
    panel_paired(fig.add_subplot(top[0, 0]))
    panel_spread(fig.add_subplot(top[0, 1]))

    panel_slotdist(fig, outer[1], menu)

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

    for letter, (x, y) in zip("abc", [(0.008, 0.975), (0.520, 0.975),
                                      (0.008, 0.545)]):
        fig.text(x, y, f"({letter})", fontsize=fs(9), fontweight="bold",
                 ha="left", va="top")

    save(fig, "fig2_petase_design", dpi=900, tight=False, subdir="main")


if __name__ == "__main__":
    main()
