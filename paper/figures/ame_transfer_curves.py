#!/usr/bin/env python3
"""Held-out AF3 transfer against training level, over training steps.

This is the overfitting panel (plan section 8.3, T1.1). A single plateau number
cannot distinguish a policy that got better from one that learned its
predictor's quirks -- both read as a high training score. Two curves can: when
the training level climbs while held-out AF3 stalls or falls, the gap between
them *is* the overfitting, measured rather than argued.

Layout is a 2x2 of small multiples, rows = backbone, columns = arm:

    columns   rf3-only batch 8 (`fragrl4`)  |  3-oracle pool batch 32 (`fragb32`)
    rows      M0097, M0255

Both columns share one y axis because both series are pass rates on [0, 1] --
there is no second scale here and no reason for a second axis. The shaded band
is the gap, drawn only where training exceeds held-out, since that is the
direction that means the training number overstates.

The training series is the **oracle-balanced** level, not the step's own score.
Under `multi_oracle_from_df` each step is folded by one predictor and the three
disagree by up to 26 points on identical designs, so the raw per-step series
jitters by more than the effect: on fragb32 M0255 step 120 the single draw reads
0.844 against a balanced 0.680. `af3_probe_offline.py` computes the balanced
mean over a sliding gap-free window and stores it as `train_pass`; the raw draw
is kept as `train_pass_step` and is plotted faintly, because showing the
difference is more honest than quietly smoothing it away.

    uv run python paper/figures/ame_transfer_curves.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figio import save          # noqa: E402
from palette import PALETTE     # noqa: E402

AME = Path(__file__).resolve().parents[2] / "experiments" / "ame"

# (family, column title). Order is the argument: single oracle first, so the
# eye reads "gap opens" then "gap closes".
# Columns are ordered so adjacent pairs are one-factor comparisons:
#   col 1 -> col 2  holds batch at 8, varies oracle count   (does the pool help?)
#   col 2 -> col 3  holds the 3-oracle pool, varies batch   (does batch help?)
# fragrl4 and fragmulti are both `batch_size: 8, seed: 1` and differ only in
# the oracle function, so the first pair is a genuine matched-batch contrast
# rather than the batch-and-oracle confound of the original two-column figure.
ARMS = [
    ("fragrl4",   "RF3 only\nbatch 8"),
    ("fragmulti", "3-oracle pool\nbatch 8"),
    ("fragb32",   "3-oracle pool\nbatch 32"),
]
BACKBONES = ["M0097", "M0255", "M0315"]
STRIDES = [0, 30, 60, 90, 120]


def load(family, backbone):
    """-> the probe summary rows for one (family, backbone), step-sorted.

    Stems differ from the family name: the batch-8 single-oracle configs are
    `*_frag4` although they live in `fragrl4/`. Globbing on the backbone rather
    than reconstructing the stem keeps this working when a family is renamed.
    """
    hits = sorted((AME / family / "offline_probe").glob(f"*{backbone}*_summary.csv"))
    if not hits:
        return None
    df = pd.concat([pd.read_csv(h) for h in hits], ignore_index=True)
    df = df[df["af3_pass"].notna()].copy()

    # Drop rows written before the balanced comparator existed. Their
    # `train_pass` is the single oracle draw for that step, which is a different
    # quantity -- on fragb32 M0255 step 146 it reads 0.484 against a balanced
    # 0.706 -- and plotting both on one line would be a series that silently
    # changes definition partway along. Those rows are the 2026-10-13
    # attribution readings, kept in experiments/ame/attribution_2026-10-13/ and
    # reported in plan section 8.1; they are not lost, just not comparable here.
    if "train_pass_step" not in df.columns:
        return None
    stale = df["train_pass_step"].isna()
    if stale.any():
        print(f"  {family}/{backbone}: dropping {int(stale.sum())} pre-balanced "
              f"row(s) at step(s) {sorted(df.loc[stale, 'step'].astype(int))}")
        df = df[~stale]

    # Points whose step label does not mean what it says. A resumed run
    # restarts the step counter at 0 and writes into the same outputs/ tree, so
    # fragb32 M0255's step_0000 was overwritten by job 29184308 before the probe
    # read it: that row is a ~step-150 policy labelled 0, and it reads +0.763
    # against a genuine step-0 level of ~0.08 on the other backbone. Excluded by
    # name rather than by an outlier rule, because it is wrong for a known
    # reason and an outlier rule would also have thrown out real volatility.
    # The original steps 33+ survive in experiments/ame/m0255_original_steps/.
    CONTAMINATED = {("fragb32", "M0255"): {0}}
    bad = CONTAMINATED.get((family, backbone), set())
    if bad:
        drop = df["step"].isin(bad)
        if drop.any():
            print(f"  {family}/{backbone}: excluding step(s) "
                  f"{sorted(df.loc[drop, 'step'])} -- overwritten by a resume")
            df = df[~drop]

    if df.empty:
        return None
    df["step"] = df["step"].astype(int)
    # A step probed more than once (a resubmission after a node failure) keeps
    # its last reading rather than averaging two attempts at the same quantity.
    return df.sort_values("step").drop_duplicates("step", keep="last")


def main():
    fig, axes = plt.subplots(
        len(BACKBONES), len(ARMS), figsize=(9.6, 7.2),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)
    missing = []

    for r, bb in enumerate(BACKBONES):
        for c, (fam, title) in enumerate(ARMS):
            ax = axes[r, c]
            df = load(fam, bb)
            if df is None:
                missing.append(f"{fam}/{bb}")
                ax.text(0.5, 0.5, "no probe data", ha="center", va="center",
                        transform=ax.transAxes, color=PALETTE["gray"], fontsize=8)
                ax.set_xticks([])
                continue

            x = df["step"].to_numpy()
            tr = df["train_pass"].to_numpy(dtype=float)
            af = df["af3_pass"].to_numpy(dtype=float)

            # The gap, shaded only where the training number overstates.
            ax.fill_between(x, af, tr, where=tr > af, interpolate=True,
                            color=PALETTE["red"], alpha=0.13, linewidth=0,
                            label="gap (training overstates)")

            if "train_pass_step" in df:
                ax.plot(x, df["train_pass_step"], color=PALETTE["gray"],
                        lw=1.0, ls=":", alpha=0.7, zorder=2,
                        label="training, single oracle draw")
            ax.plot(x, tr, color=PALETTE["blue"], lw=2.0, marker="o", ms=4,
                    zorder=3, label="training, oracle-balanced")
            ax.plot(x, af, color=PALETTE["orange"], lw=2.0, marker="s", ms=4,
                    zorder=3, label="held-out AF3")

            mean_gap = float(np.mean(tr - af))
            # The labels live in a strip above the data rather than inside the
            # plot. Choosing a corner per panel does not work here: these series
            # end anywhere between 0.03 and 1.0, so every in-axes position
            # collides with some panel's curve -- on M0315/fragrl4 the top-left
            # label sat directly on a line pinned at 0.98. The y limit is
            # extended past 1 and the ticks stop at 1.0, so the strip reads as
            # margin rather than as headroom the data could occupy.
            ax.annotate(f"mean gap {mean_gap:+.3f}", xy=(0.04, 0.94),
                        xycoords="axes fraction", fontsize=8, va="top",
                        color=PALETTE["red"] if mean_gap > 0.05 else PALETTE["gray"])

            # A panel that begins saturated cannot show a gap opening, so its
            # near-zero mean is not evidence that the arm transfers. M0315 under
            # the single-oracle arm starts at 0.98 training / 0.97 AF3: there is
            # no room above for training to pull away. Said on the figure
            # because the number alone reads as a pass.
            if tr[0] >= 0.9 and af[0] >= 0.9:
                ax.annotate("starts at ceiling -- gap untestable",
                            xy=(0.04, 0.84), xycoords="axes fraction",
                            fontsize=7, va="top", color=PALETTE["gray"],
                            style="italic")

            # Say which strides are absent and why. A panel that silently starts
            # at step 60 looks like a shorter run rather than a damaged one.
            probed = set(int(v) for v in x)
            gone = sorted(set(STRIDES) - probed)
            if gone:
                ax.annotate("steps " + ",".join(str(g) for g in gone) + " unusable",
                            xy=(0.04, 0.84), xycoords="axes fraction",
                            fontsize=7, va="top", color=PALETTE["gray"], style="italic")

            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{bb}\nchimera pass rate", fontsize=9)
            ax.set_ylim(0, 1.30)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("training step", fontsize=9)

    # One legend for the figure: identity is never carried by colour alone, and
    # four panels repeating the same four keys is noise.
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Held-out AF3 transfer vs training level, chimera objective",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    save(fig, "ame_transfer_curves")   # figio.save prints the path itself
    if missing:
        # Named, not silent: an absent panel means a probe job has not landed,
        # which is different from a panel whose gap is genuinely zero.
        print("no probe data yet for: " + ", ".join(missing))


if __name__ == "__main__":
    main()
