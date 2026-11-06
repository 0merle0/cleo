#!/usr/bin/env python3
"""Motif-RMSD distributions, rather than the pass rates thresholded out of them.

A success rate is one number read off the far tail of a distribution at a fixed
cutoff. Two arms can share a success rate with completely different distributions
-- one whose whole mode has moved under the cutoff, another still centred at
\\SI{3}{\\angstrom} that occasionally throws an excursion across it. Those are not
the same result, and the E3 replication showed why it matters: of nine designs
that cleared the cutoff during training on the near-miss backbones, five failed
to reproduce even once in 40 refolds, because they were tail draws from a
distribution whose median sat near 3 A.

So plot the distribution and let the cutoff be a line through it.

  Left   M0097, where the mode genuinely moves. Baseline temperature ladder plus
         the four ablation arms.
  Right  the three near-miss backbones, where it does not -- medians at
         2.8-3.1 A with the cutoff far out in the tail.

Encoding note: both families are *ordinal*, not categorical -- the temperature
ladder and the ablation ladder each run in a fixed order -- so each gets a single
hue ramped light to dark rather than a set of arbitrary categorical hues. That
also sidesteps a real failure of the house categorical palette, whose orange and
green sit at OKLab dE 5.7 under protanopia (below the 8 floor). The KL arm keeps
an accent hue because it is the highlighted result, not another rung.

    uv run python paper/figures/ame_distributions.py
"""

import glob
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.titlesize": 15, "axes.linewidth": 1.0,
})
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figio import save  # noqa: E402
from palette import PALETTE  # noqa: E402

AME = HERE.parents[1] / "experiments" / "ame"
CUT = 1.5                      # motif all-atom RMSD cutoff, angstrom
ACCENT = PALETTE["red"]


def ramp(hex_color, n, lo=0.35, hi=1.0):
    """-> n shades of one hue, light to dark. Sequential encoding for an ordinal family."""
    base = np.array(matplotlib.colors.to_rgb(hex_color))
    return [tuple(1 - (1 - base) * f) for f in np.linspace(lo, hi, n)]


def baseline_rmsd(T):
    """Best-of-5 motif RMSD per design, M0097, one temperature."""
    f = AME / "tempsweep" / f"temp_T{T}_bo5_scored.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = d[d.backbone.str.contains("M0097", regex=False)]
    col = "rfd2_motif_rmsd_min" if "rfd2_motif_rmsd_min" in d else "rfd2_motif_rmsd"
    return d[col].dropna().values


def arm_rmsd(run_dir, last=40):
    """Best-of-5 motif RMSD per design over a run's final `last` steps."""
    fs = sorted(glob.glob(f"{run_dir}/**/outputs/step_*/metrics.csv", recursive=True))
    if not fs:
        return None
    d = pd.concat(
        [pd.read_csv(m).assign(step=int(re.search(r"step_(\d+)", m).group(1)))
         for m in fs], ignore_index=True)
    d = d[d.step >= d.step.max() - (last - 1)]
    return d.ame_motif_rmsd.dropna().values


def panel(ax, rows, title, xmax):
    """Horizontal box + jittered strip, one row per arm. -> nothing."""
    rng = np.random.default_rng(0)
    labels = [r[0] for r in rows]
    for i, (lab, v, c) in enumerate(rows):
        y = len(rows) - i
        # Strip first, box on top: the box carries median and quartiles, the
        # strip shows the shape the box hides -- notably whether mass near the
        # cutoff is a shoulder or a lone excursion.
        ax.scatter(np.clip(v, None, xmax), y + rng.uniform(-0.17, 0.17, len(v)),
                   s=4, color=c, alpha=0.22, linewidths=0, rasterized=True, zorder=1)
        bp = ax.boxplot([v], positions=[y], vert=False, widths=0.5,
                        showfliers=False, patch_artist=True, zorder=2)
        for box in bp["boxes"]:
            box.set(facecolor="white", edgecolor=c, linewidth=1.6, alpha=0.9)
        for part in ("whiskers", "caps"):
            for a in bp[part]:
                a.set(color=c, linewidth=1.3)
        for med in bp["medians"]:
            med.set(color=c, linewidth=2.4)
        # Median label rides just past the upper whisker rather than pinned to
        # the axis edge: with a log axis a short box sits far from the edge and
        # an edge-anchored number cannot be matched to its row by eye.
        w = min(np.percentile(v, 75) + 1.5 * (np.percentile(v, 75) - np.percentile(v, 25)),
                v.max())
        ax.text(min(w, xmax) * 1.06, y, f"{np.median(v):.2f}", ha="left",
                va="center", fontsize=9, color="#374151")

    ax.axvline(CUT, color="#111827", lw=1.4, ls="--", zorder=3)
    ax.text(CUT, len(rows) + 0.80, f"cutoff {CUT} Å", ha="center", fontsize=10)
    ax.set_yticks(range(len(rows), 0, -1))
    ax.set_yticklabels(labels)
    # Log scale: RMSD spans 1 to >10 A here, and on a linear axis the single
    # worst arm compresses every distribution that matters into the left fifth.
    ax.set_xscale("log")
    ax.set_xlim(0.85, xmax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([1, 1.5, 2, 3, 5, 8, 12])
    ax.set_ylim(0.4, len(rows) + 1.0)
    ax.set_xlabel("motif all-atom RMSD (Å), best of 5")
    ax.set_title(title, pad=8)
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def main():
    temps = ["0.1", "0.2", "0.3", "0.5", "0.7", "1.0"]
    tcol = ramp(PALETTE["orange"], len(temps))
    arms = [("legacy objective", "conv200_legacy"),
            ("+ surrogate fix", "conv200_surr"),
            ("+ log-prob fix", "conv200"),
            ("+ KL anchor", "conv200_kl")]
    acol = ramp(PALETTE["blue"], len(arms) - 1) + [ACCENT]

    left = []
    for T, c in zip(temps, tcol):
        v = baseline_rmsd(T)
        if v is not None and len(v):
            left.append((f"LigandMPNN $T$={T}", v, c))
    for (lab, stem), c in zip(arms, acol):
        v = arm_rmsd(AME / "centering" / f"run_M0097_1ctt_cond9_14_{stem}")
        if v is not None and len(v):
            left.append((lab, v, c))

    near = [("M0050_1dbt", "run_M0050_1dbt_cond2_6"),
            ("M0365_1pfk", "run_M0365_1pfk_cond20_37"),
            ("M0375_4ts9", "run_M0375_4ts9_cond21_4")]
    ncol = ramp(PALETTE["gray"], len(near))
    right = []
    for (lab, stem), c in zip(near, ncol):
        v = arm_rmsd(AME / "runs9near" / stem, last=75)
        if v is not None and len(v):
            right.append((lab, v, c))

    fig, axes = plt.subplots(
        1, 2, figsize=(15.5, 6.4),
        gridspec_kw={"width_ratios": [len(left), max(len(right), 3) + 1.2]})
    panel(axes[0], left, "M0097 — the mode moves under the cutoff", xmax=7.0)
    panel(axes[1], right,
          "Near-miss backbones — the mode does not; hits are tail draws", xmax=12.0)

    # The replicated hit, marked where it actually sits in its own distribution.
    axes[1].annotate("hit00\n22/40 on refold", xy=(1.27, 2), xytext=(2.6, 2.75),
                     fontsize=10, color=ACCENT, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))

    fig.suptitle("Where the distributions actually sit, rather than the "
                 "success rate thresholded out of them")
    fig.tight_layout()
    save(fig, "ame_distributions")


if __name__ == "__main__":
    main()
