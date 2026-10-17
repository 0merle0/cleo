#!/usr/bin/env python3
"""Zero-shot structure prediction against measured PETase activity.

Three panels, because the headline correlation is not interpretable on its own:

  (a) the best zero-shot metric against activity, with the number a library
      designer actually buys -- the enrichment from selecting the top 5% by
      prediction rather than at random.
  (b) activity by design round, on the single screen assay run. This is the
      "searching the library works" panel.
  (c) activity against distance from the design parent. This is the control
      panel, and it is here because distance alone ranks activity BETTER than
      any AF3 metric does (rho -0.65 vs +0.59). A zero-shot claim that does
      not show this is quoting a confound.

Panel (c) is also where the revalidated block is marked. Those 20 designs come
from a different run of the same assay and are the only points above 100, so
they are drawn separately everywhere and never pooled into a correlation.

    uv run python paper/figures/petase_zeroshot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from figio import save
from palette import PALETTE

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "petase_activity_zeroshot.csv"

METRIC = "acyl_activesite_plddt"
METRIC_LABEL = "AF3 active-site pLDDT (acyl-enzyme intermediate)"
# Activity is a rate spanning -1.5 to 261 with a mode near zero, so every panel
# plots it on a symlog axis. A linear axis puts 90% of the library in one row of
# pixels; a log axis cannot show the negative rates, which are real measurements
# and not errors.
SYMLOG = dict(linthresh=1.0, linscale=0.4)
TOP_FRAC = 0.05


def load():
    df = pd.read_csv(DATA)
    df["screen"] = df.revalidated == 0
    return df


def enrichment(df, metric, top_frac=TOP_FRAC):
    """Hit rate for a top-decile-active design among the top `top_frac` by metric.

    -> (hit_rate, baseline, n_selected). Computed on the screen run only.
    """
    sub = df[df.screen & df[metric].notna()]
    thresh = sub.activity.quantile(0.90)
    k = int(top_frac * len(sub))
    sel = sub.nlargest(k, metric)
    return (sel.activity >= thresh).mean(), (sub.activity >= thresh).mean(), k


def panel_zeroshot(ax, df):
    sub = df[df[METRIC].notna()]
    scr, rev = sub[sub.screen], sub[~sub.screen]
    ax.scatter(scr[METRIC], scr.activity, s=4, alpha=0.25, lw=0,
               color=PALETTE["blue"], rasterized=True, label="screen run")
    ax.scatter(rev[METRIC], rev.activity, s=26, lw=0.8, facecolor="none",
               edgecolor=PALETTE["red"], label="revalidated (separate run)")
    rho = spearmanr(scr[METRIC], scr.activity).statistic
    hit, base, k = enrichment(df, METRIC)
    ax.set_yscale("symlog", **SYMLOG)
    ax.set_xlabel(METRIC_LABEL)
    ax.set_ylabel("MoMI normalized rate")
    ax.set_title(f"(a) lightly predictive: $\\rho$ = {rho:+.2f} (n = {len(scr):,})",
                 loc="left", fontsize=9)
    ax.text(0.03, 0.97,
            f"top {int(TOP_FRAC * 100)}% by prediction (n = {k}):\n"
            f"{hit:.0%} are top-decile active\nvs {base:.0%} at random "
            f"({hit / base:.1f}$\\times$)",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            color=PALETTE["gray"])
    ax.legend(loc="lower right", fontsize=7, frameon=False, markerscale=1.5)


def panel_rounds(ax, df):
    scr = df[df.screen & df.momi_round.notna()]
    rounds = sorted(scr.momi_round.unique())
    data = [scr.loc[scr.momi_round == r, "activity"].values for r in rounds]
    bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color="white", lw=1.4))
    for box in bp["boxes"]:
        box.set(facecolor=PALETTE["blue"], edgecolor=PALETTE["blue"], lw=0.8)
    for r_i, vals in enumerate(data, start=1):
        ax.scatter([r_i], [vals.max()], marker="_", s=120, lw=1.4,
                   color=PALETTE["orange"], zorder=3)
    ax.set_yscale("symlog", **SYMLOG)
    ax.set_xticks(range(1, len(rounds) + 1))
    ax.set_xticklabels([f"{int(r)}\nn={len(v)}" for r, v in zip(rounds, data)])
    ax.set_xlabel("design round")
    ax.set_ylabel("MoMI normalized rate")
    med = [float(np.median(v)) for v in data]
    ax.set_title(f"(b) median {med[0]:.1f} $\\to$ {med[-1]:.1f} across rounds "
                 f"(orange: best)", loc="left", fontsize=9)


def panel_distance(ax, df):
    scr, rev = df[df.screen], df[~df.screen]
    ax.scatter(scr.n_mut_from_parent, scr.activity, s=4, alpha=0.25, lw=0,
               color=PALETTE["gray"], rasterized=True, label="screen run")
    ax.scatter(rev.n_mut_from_parent, rev.activity, s=26, lw=0.8,
               facecolor="none", edgecolor=PALETTE["red"],
               label="revalidated (separate run)")
    rho = spearmanr(scr.n_mut_from_parent, scr.activity).statistic
    ax.set_yscale("symlog", **SYMLOG)
    ax.set_xlabel("mutations from design parent")
    ax.set_ylabel("MoMI normalized rate")
    ax.set_title(f"(c) the control: distance alone gives $\\rho$ = {rho:+.2f}",
                 loc="left", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, frameon=False, markerscale=1.5)


def main():
    df = load()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    panel_zeroshot(axes[0], df)
    panel_rounds(axes[1], df)
    panel_distance(axes[2], df)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.4, color="#E5E7EB", zorder=0)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "petase_zeroshot")

    # Print what the section quotes, so the prose and the figure cannot drift.
    scr = df[df.screen]
    print(f"\nscreen designs: {len(scr):,}   revalidated: {(~df.screen).sum()}")
    for m in [c for c in df.columns if c.startswith(("acyl_", "substrate_"))]:
        s = scr[scr[m].notna()]
        print(f"  {m:30s} rho={spearmanr(s[m], s.activity).statistic:+.3f}  n={len(s):,}")
    far = scr[(scr.n_mut_from_parent >= 20) & scr[METRIC].notna()]
    print(f"  {'(>=20 mut only)':30s} rho="
          f"{spearmanr(far[METRIC], far.activity).statistic:+.3f}  n={len(far):,}")
    print(f"  {'n_mut_from_parent':30s} rho="
          f"{spearmanr(scr.n_mut_from_parent, scr.activity).statistic:+.3f}  n={len(scr):,}")
    hit, base, k = enrichment(df, METRIC)
    print(f"\ntop {TOP_FRAC:.0%} enrichment: {hit:.3f} vs {base:.3f} "
          f"= {hit / base:.2f}x  (n={k})")
    cv = pd.read_csv(HERE / "data" / "petase_model_cv_scores.csv")
    print("\nheld-out RidgeCV, 200 splits:")
    print(cv.groupby("feature_set")[["test_spearman", "test_r2"]]
            .agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
