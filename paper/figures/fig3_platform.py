#!/usr/bin/env python3
"""Figure 3 --- the experimental platform, and the three libraries.

Data panels, built from the re-processed raw plate reads
(`library_kinetics.py`) plus the three-library snapshot:

    (b) the filter cascade        (e) one worked kinetic trace
    (c) controls agree per plate  (f) the three libraries, by round
    (d) replicate parity          (g) what was assayed against what exists

(a), the assembly-and-assay pipeline strip, is Jason's `wetlab_pipeline.svg`
and is a schematic, not a data plot.

    uv run python paper/figures/fig3_platform.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from blockout import slot  # noqa: E402
from figio import save  # noqa: E402
from palette import INK, PAGE_WIDTH_IN, PALETTE, SEQUENTIAL, fs  # noqa: E402

DATA = HERE / "data"
LIBC = {"1_original": SEQUENTIAL[1],
        "2_g20_optimisation": SEQUENTIAL[3],
        "3_experiment_guided": SEQUENTIAL[5]}
LIBL = {"1_original": "1 original",
        "2_g20_optimisation": "2 G20 optimisation",
        "3_experiment_guided": "3 experiment-guided"}


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


def load():
    """-> (snapshot, re-processed designs, replicates, traces)."""
    snap = pd.read_csv(DATA / "petase_libraries_all.csv")
    d = pd.read_csv(DATA / "library3_reprocessed_designs.csv")
    r = pd.read_csv(DATA / "library3_reprocessed_replicates.csv")
    t = pd.read_csv(DATA / "library3_reprocessed_traces.csv")
    return snap, d, r, t


# ================================================================ panel (b)

def panel_cascade(ax, designs, reps):
    """What survives each gate, library 3, from the raw reads.

    THE SHIPPED TABLES CANNOT DRAW THIS. jq01's per-round files are
    `processed_filtered_*` -- already past the gates -- so in the snapshot
    library 3 reads as 100% kept and the dropped counts are simply absent.
    These numbers come from re-processing the raw plate reads, which
    reproduces the published values at Spearman 0.990 with a median ratio of
    1.000, so the survivors are the same survivors and only the denominators
    are new.

    Wells, then constructs, then the two gates in the order they are
    applied. The first step is a 3x drop and is not a filter: it is three
    replicate wells becoming one construct.
    """
    s = designs[designs.sample_type == "sample"]
    sr = reps[reps.sample_type == "sample"]
    steps = [
        ("wells\nassayed", len(sr)),
        ("constructs", len(s)),
        ("expressed\n$\\geq$0.1 $\\mu$M", int(s.passes_expression.sum())),
        ("CV < 1", int(s.passes_filter.sum())),
    ]
    xs = np.arange(len(steps))
    vals = [v for _, v in steps]
    cols = [INK["muted"], SEQUENTIAL[2], SEQUENTIAL[3], SEQUENTIAL[5]]
    ax.bar(xs, vals, width=0.62, color=cols, edgecolor="white", linewidth=0.6,
           zorder=2)
    for x, v in zip(xs, vals):
        ax.text(x, v * 1.06, f"{v:,}", fontsize=fs(6.5), ha="center",
                va="bottom", color="black")
    ax.set_xticks(xs)
    ax.set_xticklabels([k for k, _ in steps], fontsize=fs(6.3))
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, max(vals) * 1.22)
    _tidy(ax, ylab="Count")


# ================================================================ panel (c)

def panel_controls(ax, designs):
    """Positive-control rate, per plate, per round.

    RAW rate, not normalised. The normalised control is 1.0 by construction
    -- it is the denominator -- so plotting it would show the definition
    rather than the reproducibility. Raw rate is the quantity that could
    disagree between plates and does not.

    One marker per plate set; plates within a round should sit on top of
    each other, and the round-to-round drift is the real axis of variation.
    """
    c = designs[designs.sample_type == "positive_control"]
    rng = np.random.default_rng(0)
    cvs = []
    for rnd, g in c.groupby("round_number"):
        v = g.bg_sub_conc_norm_rate_mean.to_numpy()
        x = rnd + rng.uniform(-0.17, 0.17, len(v))
        ax.plot(x, v, "o", ms=3.4, mfc=SEQUENTIAL[3], mec="white", mew=0.7,
                ls="none", zorder=4)
        med = float(np.median(v))
        ax.plot([rnd - 0.3, rnd + 0.3], [med, med], color="black", lw=1.4,
                zorder=5)
        cvs.append((rnd, float(np.std(v) / np.mean(v))
                    if np.mean(v) else float("nan"), len(v)))
    # Placed after the loop: inside it, ax.get_ylim() is read before the
    # last round's points exist and the labels land at the wrong height.
    top = ax.get_ylim()[1]
    for rnd, cv, n in cvs:
        ax.text(rnd, top, f"{100*cv:.0f}%", fontsize=fs(5.8), ha="center",
                va="bottom", color=INK["muted"])
    ax.set_xticks(sorted(c.round_number.unique()))
    _tidy(ax, ylab="Control rate\n(bg-subtracted, per $\\mu$M)",
          xlab="Round")


# ================================================================ panel (d)

def panel_parity(ax, reps, designs=None):
    """Replicate against replicate, one point per construct.

    The platform's reproducibility claim, and it needs the individual wells
    -- the shipped tables carry mean, sd and CV but not the three values
    they came from, so this panel is only possible from the raw reads.

    Replicate 1 against replicate 2, not against a mean: a point against the
    mean of a set it belongs to is correlated with itself.

    SPEARMAN, NOT PEARSON, over the full set. Pearson is -0.27 here and that
    number is meaningless: a handful of wells pair a large positive rate
    with a near-zero or negative one (0.730 against -0.021 is the worst) and
    they dominate a product-moment statistic while being exactly what the CV
    gate exists to remove. Rank agreement over everything is 0.978; Pearson
    over the constructs that pass the gate is reported alongside, since that
    is the set any downstream analysis actually sees.

    Failing constructs are drawn in grey rather than dropped. A parity plot
    showing only what survived its own filter is not a reproducibility
    claim.
    """
    s = reps[reps.sample_type == "sample"]
    key = ["round_number", "plate_number", "sample_number", "source_dna_well"]
    piv = s.pivot_table(index=key, columns="replicate_number",
                        values="norm_rate", aggfunc="first")
    if 1 not in piv.columns or 2 not in piv.columns:
        return
    g = piv[[1, 2]].dropna().reset_index()

    keep = None
    if designs is not None:
        ok = designs[designs.passes_filter][key].drop_duplicates()
        ok["_pass"] = True
        g = g.merge(ok, on=key, how="left")
        keep = g["_pass"].notna().to_numpy()

    if keep is not None and (~keep).any():
        ax.plot(g.loc[~keep, 1], g.loc[~keep, 2], "o", ms=2.0,
                mfc=PALETTE["red"], mec="none", alpha=0.55, ls="none",
                rasterized=True, zorder=4, label="fails CV")
    sel = g if keep is None else g[keep]
    ax.plot(sel[1], sel[2], "o", ms=1.8, mfc=SEQUENTIAL[3], mec="none",
            alpha=0.3, ls="none", rasterized=True, zorder=3)

    lim = float(np.nanpercentile(np.r_[g[1], g[2]], 99.0))
    ax.plot([0, lim], [0, lim], color="black", lw=0.8, ls=(0, (3, 2)),
            zorder=5)

    rho = float(g[1].corr(g[2], method="spearman"))
    txt = f"n = {len(g):,}\n$\\rho$ = {rho:.2f}"
    if keep is not None and keep.any():
        rp = float(np.corrcoef(sel[1], sel[2])[0, 1])
        txt += f"\nr = {rp:.2f} (passing)"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=fs(6.3),
            ha="left", va="top", color="black")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    _tidy(ax, ylab="Replicate 2", xlab="Replicate 1")
    if keep is not None and (~keep).any():
        ax.legend(fontsize=fs(6.0), loc="lower right", frameon=False,
                  handlelength=0.8, borderpad=0.0, handletextpad=0.4)


# ================================================================ panel (e)

def panel_trace(ax, traces, reps, n_show=3):
    """One plate's worth of worked traces: 4MU against real time.

    What "a rate" means, drawn once. A high sample, a median sample and the
    plate's negative controls, with the least-squares fit each one's rate
    comes from. Nothing else in the figure shows the measurement itself.
    """
    t = traces[(traces.round_number == 1) & (traces.plate_number == 1)
               & (traces.sample_number == 0)].copy()
    if not len(t):
        return
    # `datetime` is a Unix epoch in SECONDS (1.77e9 = Jan 2026), which is
    # also the x the rate was fitted against. Parsing it as a timestamp
    # reads it as nanoseconds and collapses the axis to ~1e-9 hours.
    t["h"] = (t["datetime"] - t["datetime"].min()) / 3600.0

    r = reps[(reps.round_number == 1) & (reps.plate_number == 1)
             & (reps.sample_number == 0) & (reps.sample_type == "sample")]
    # Picked on RAW rate, because this panel plots raw 4MU. Picking on the
    # normalised rate put a shallow trace under the label "most active": a
    # poorly expressed well divides a small slope by a small concentration
    # and can top the normalised ranking while barely moving on this axis.
    # That is a real property of the measurement, and the place to show it
    # is not the panel whose job is to define what a slope is.
    r = r.sort_values("raw_rate")
    picks = []
    if len(r):
        picks.append((r.iloc[-1].well, SEQUENTIAL[5], "fastest"))
        picks.append((r.iloc[len(r) // 2].well, SEQUENTIAL[2], "median"))
    neg = t[t.sample_type == "negative_control"].well.unique()[:n_show]

    for well in neg:
        g = t[t.well == well].sort_values("h")
        ax.plot(g.h, g["4MU_um_mean"], "-o", ms=2.4, lw=0.9,
                color=INK["muted"], alpha=0.8, zorder=3)
    if len(neg):
        ax.plot([], [], "-o", ms=2.4, lw=0.9, color=INK["muted"],
                label="no enzyme")

    for well, colour, label in picks:
        g = t[t.well == well].sort_values("h")
        ax.plot(g.h, g["4MU_um_mean"], "-o", ms=3.0, lw=1.2, color=colour,
                zorder=4, label=label)

    _tidy(ax, ylab="4MU ($\\mu$M)", xlab="Time (h)")
    ax.legend(fontsize=fs(6.3), loc="upper left", frameon=False,
              handlelength=1.2, borderpad=0.0, labelspacing=0.25,
              handletextpad=0.4)


# ================================================================ panel (f)

def panel_libraries(ax, snap):
    """All three libraries on one momi axis, by round. The headline.

    LOG y: the three libraries span 0.01 to 395 x momi and a linear axis
    renders libraries 1 and 2 as a line on the floor.

    G20 does not land at one height across the libraries -- 17.11 measured
    in library 1 against its own momi control, 19.70 derived in libraries 2
    and 3 -- and that 26% disagreement is the whole uncertainty in the
    factor. It is left visible rather than normalised away; see
    docs/MOMI_NORMALISATION.md.
    """
    s = snap[(snap.sample_type == "sample") & snap.passes_filter].copy()
    s = s[s.momi_norm_rate > 0]
    rng = np.random.default_rng(0)
    pos, labels = [], []
    x = 0
    for lib in ("1_original", "2_g20_optimisation", "3_experiment_guided"):
        g = s[s.library == lib]
        for rnd in sorted(g.round_number.unique()):
            v = g[g.round_number == rnd].momi_norm_rate.to_numpy()
            if not len(v):
                continue
            ax.plot(x + rng.uniform(-0.3, 0.3, len(v)), v, "o", ms=1.5,
                    mfc=LIBC[lib], mec="none", alpha=0.35, ls="none",
                    rasterized=True, zorder=3)
            med = float(np.median(v))
            ax.plot([x - 0.38, x + 0.38], [med, med], color="black", lw=1.5,
                    zorder=5)
            pos.append(x)
            labels.append(str(int(rnd)))
            x += 1
        x += 0.6

    ax.axhline(1, color=PALETTE["gray"], lw=0.9, ls=":", zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=fs(6.3))
    ax.tick_params(axis="x", length=0)
    _tidy(ax, ylab="Activity ($\\times$ momi WT)", xlab="Round")

    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ms=4, mfc=LIBC[k],
                              mec="none", ls="none", label=LIBL[k])
                       for k in LIBC],
              fontsize=fs(6.3), loc="upper left", frameon=False,
              handlelength=0.8, borderpad=0.0, labelspacing=0.22,
              handletextpad=0.4)


# ================================================================ panel (g)

def panel_scale(ax, snap):
    """What was assayed against what the library contains.

    The gap is the point: ~10^4 measurements against libraries of 10^7 and
    10^12 members. Nothing here is a sampling claim -- it is the reason the
    search in Figure 4 has to exist at all.
    """
    rows = [("1 original", 1.1e7, int(((snap.library == "1_original")
                                       & (snap.sample_type == "sample")).sum())),
            ("2 G20 opt.", 1e12, int(((snap.library == "2_g20_optimisation")
                                      & (snap.sample_type == "sample")).sum())),
            ("3 exp-guided", 1e12, int(((snap.library == "3_experiment_guided")
                                        & (snap.sample_type == "sample")).sum()))]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, size, n) in zip(ys, rows):
        ax.plot([n, size], [y, y], color=INK["ghost"], lw=1.2, zorder=2,
                solid_capstyle="round")
        ax.plot(size, y, "o", ms=5.5, mfc=INK["muted"], mec="white", mew=0.8,
                zorder=4)
        ax.plot(n, y, "o", ms=5.5, mfc=SEQUENTIAL[4], mec="white", mew=0.8,
                zorder=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=fs(6.5))
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.set_xlim(1e2, 1e14)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    _tidy(ax, xlab="Library members")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ms=5, mfc=SEQUENTIAL[4], mec="white",
               ls="none", label="assayed"),
        Line2D([], [], marker="o", ms=5, mfc=INK["muted"], mec="white",
               ls="none", label="library size")],
        fontsize=fs(6.3), loc="lower right", frameon=False,
        handlelength=0.8, borderpad=0.0, labelspacing=0.22,
        handletextpad=0.4)


# ================================================================== assembly

def main():
    snap, designs, reps, traces = load()

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 8.4))
    outer = fig.add_gridspec(4, 1, height_ratios=[0.52, 1.0, 1.0, 1.0],
                             left=0.085, right=0.985, top=0.968,
                             bottom=0.045, hspace=0.62)

    slot(fig.add_subplot(outer[0]), "a",
         "The assembly and assay pipeline",
         "built", "svg/wetlab_pipeline.svg -- splice it into this slot",
         lines=[
             "geneblocks with BsaI adapters -> ECHO acoustic dispense ->",
             "Golden Gate -> PCR in plate -> PUREfrex, 3 replicates ->",
             "add PET-trimer probe -> plate reader",
         ])

    r1 = outer[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.95], wspace=0.46)
    panel_cascade(fig.add_subplot(r1[0, 0]), designs, reps)
    panel_controls(fig.add_subplot(r1[0, 1]), designs)
    panel_parity(fig.add_subplot(r1[0, 2]), reps, designs)

    r2 = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.34)
    panel_trace(fig.add_subplot(r2[0, 0]), traces, reps)
    panel_scale(fig.add_subplot(r2[0, 1]), snap)

    panel_libraries(fig.add_subplot(outer[3]), snap)

    for letter, (x, y) in zip("abcdefg", [
            (0.008, 0.975), (0.008, 0.800), (0.345, 0.800), (0.675, 0.800),
            (0.008, 0.535), (0.505, 0.535), (0.008, 0.275)]):
        fig.text(x, y, f"({letter})", fontsize=fs(9), fontweight="bold",
                 ha="left", va="top")

    save(fig, "fig3_platform", dpi=900, tight=False, subdir="main")


if __name__ == "__main__":
    main()
