#!/usr/bin/env python3
"""E4: how many folds does a library of N passing designs cost?

Folding is the entire cost of this pipeline -- MPNN sampling is microseconds --
so both arms are charged in *designs folded*, and the policy carries its
training as debt: batch_size x N_steps designs folded before it yields anything.

Two endpoints, because they do not give the same answer:
  * passing designs -- usable library members
  * distinct (position, residue) substitutions among them -- the "unique
    mutations" a library is being built to explore

**Reaching 500-1000 passing designs cannot be run for the baseline.** At
T=1.0 that is 24,000 folds on the easy backbone, and on the other two the
observed rate is 0/96, so no finite budget is demonstrated. Both curves are
therefore projections, and the figure marks where measurement stops.

Pass rate extrapolates as a binomial: folds = target / rate, with Clopper-
Pearson bounds. A 0/96 arm has no rate, only the one-sided bound (rule of
three), so its projection is a lower bound on cost and is drawn as an arrow.

Substitution count does NOT extrapolate linearly -- discovery saturates as the
easily-reached substitutions are exhausted. Measured on M0097, the marginal
return per design falls ~5x between 2 and 238 designs. This uses the standard
incidence-based (Chao2) rarefaction/extrapolation of Colwell et al. (2012),
which is the same estimator ecology uses for "how many more species if I keep
sampling". Extrapolation is capped at 3x the observed passing-design count and simply not
drawn past it. That cap is load-bearing rather than decorative: T=1.0 yields 4
passing designs on the easy backbone and 0 on the other two, so its Q1/Q2 are
pure noise and an uncapped Chao2 curve rises *above* the trained policy's --
an artefact of estimating undetected richness from four observations. Where a
projection cannot be made, the panel says so instead of drawing a line.

    uv run python paper/figures/ame_efficiency.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ame_diversity import AME, BACKBONES, baseline, cleo, passing, reference, subs
from figio import save
from palette import PALETTE

C_BASE, C_CLEO = PALETTE["gray"], PALETTE["blue"]
C_TEXT = "#4B5563"
TIERS = ["easy", "medium", "hard"]
# Designs folded before a trained policy yields anything: batch_size x N_steps.
# Read from the configs rather than assumed -- backbone19 ran 200 steps, the
# E21 replicates 150.
TRAIN_FOLDS = {"run_M0097_1ctt_cond9_14": 16 * 150,
               "run_M0904_1qgx_cond39_95": 16 * 200,
               "run_M0907_1rbl_cond40_74": 16 * 200}


def incidence_matrix(keep, ref):
    """-> boolean (designs x substitutions) incidence, for Chao2."""
    from analyze_selection2 import as_matrix
    if not len(keep):
        return np.zeros((0, 0), bool)
    M = as_matrix(keep.sequence.tolist())
    cols = sorted({(j, M[i, j]) for i in range(M.shape[0])
                   for j in range(M.shape[1]) if M[i, j] != ref[j]})
    idx = {c: k for k, c in enumerate(cols)}
    I = np.zeros((len(M), len(cols)), bool)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] != ref[j]:
                I[i, idx[(j, M[i, j])]] = True
    return I


def chao2_extrapolate(I, targets):
    """Expected distinct substitutions when `targets` designs are sampled.

    Incidence-based extrapolation (Colwell et al. 2012, eq. 9). Interpolation
    below the observed count uses exact rarefaction instead, so the curve is
    measured where data exists and modelled only past it.
    """
    T, S_obs = I.shape[0], int(I.any(0).sum())
    if T == 0:
        return {t: 0.0 for t in targets}
    freq = I.sum(0)
    freq = freq[freq > 0]
    Q1, Q2 = int((freq == 1).sum()), int((freq == 2).sum())
    # Undetected richness. The Q2==0 branch is the standard bias-corrected form.
    if Q2 > 0:
        Q0 = ((T - 1) / T) * Q1 ** 2 / (2 * Q2)
    else:
        Q0 = ((T - 1) / T) * Q1 * (Q1 - 1) / 2
    Q0 = max(Q0, 0.0)

    out = {}
    for t in targets:
        if t <= T:                                    # exact rarefaction
            with np.errstate(over="ignore"):
                miss = np.array([np.exp(stats.hypergeom.logsf(-1, T, T - f, t))
                                 if False else 0.0 for f in freq])
            # P(substitution absent from t of T designs) = C(T-f, t)/C(T, t)
            lp = np.array([
                (stats.binom.logpmf(0, 0, 0) if False else
                 (np.sum(np.log(np.arange(T - f, T - f - t, -1)))
                  - np.sum(np.log(np.arange(T, T - t, -1)))) if (T - f) >= t else -np.inf)
                for f in freq])
            out[t] = float(S_obs - np.exp(lp).sum())
        else:                                         # extrapolation
            tstar = t - T
            if Q0 <= 0 or Q1 <= 0:
                out[t] = float(S_obs)
            else:
                out[t] = float(S_obs + Q0 * (1 - (1 - Q1 / (T * Q0 + Q1)) ** tstar))
    return out


def measured_trajectory(bb):
    """The T=1.0 fold-until-N-pass run, if it has produced anything yet.

    Returns (folds, distinct_passing, frame) or None. Prefer this over the
    96-design pilot wherever it exists: the pilot put M0097 at 4/96 = 4.17%
    with a 95% CI of 1.15-10.33%, so a projection anchored on it inherits that
    whole range. The trajectory measures the rate on thousands of designs
    instead of ninety-six.
    """
    f = AME / "t1_yield" / bb / "cumulative_scored.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if not len(d):
        return None
    return len(d), int(d[d.rfd2_any_pass].sequence.nunique()), d


def collect(targets):
    rows = []
    for bb, tier in zip(BACKBONES, TIERS):
        ref = reference(bb)
        b = baseline(bb)
        b1 = b[b.temperature == 1.0]
        traj = measured_trajectory(bb)
        for label, df, train in (("T=1.0", b1, 0), ("this paper", cleo(bb, "random"), TRAIN_FOLDS[bb])):
            if label == "T=1.0" and traj is not None:
                # Measured run supersedes the 96-design pilot for this arm.
                _, _, df = traj
            keep = passing(df)
            k, n = int(df.rfd2_any_pass.sum()), len(df)
            rate = k / n
            hi = stats.beta.ppf(0.975, k + 1, n - k)          # upper bound on rate
            I = incidence_matrix(keep, ref)
            ext = chao2_extrapolate(I, targets)
            for t in targets:
                folds = (train + t / rate) if rate > 0 else np.nan
                folds_lb = train + t / hi                      # cheapest consistent with data
                rows.append(dict(backbone=bb, tier=tier, arm=label, target=t,
                                 n_obs=n, k_obs=k, rate=rate, train=train,
                                 folds=folds, folds_lb=folds_lb,
                                 U=ext[t], n_pass_obs=len(keep),
                                 extrap=t > max(len(keep), 1)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="1,2,5,10,20,50,100,200,500,1000")
    a = ap.parse_args()
    targets = [int(x) for x in a.targets.split(",")]
    df = collect(targets)

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.0), sharex=True)
    for col, (bb, tier) in enumerate(zip(BACKBONES, TIERS)):
        d = df[df.backbone == bb]
        axT, axU = axes[0][col], axes[1][col]

        for arm, col_ in (("T=1.0", C_BASE), ("this paper", C_CLEO)):
            g = d[d.arm == arm].sort_values("target")
            meas, ext = g[~g.extrap], g[g.extrap]
            solid = g[g.target <= 3 * g.n_pass_obs.iloc[0]]

            # -- top: folds required --------------------------------------
            y = g.folds if g.folds.notna().any() else g.folds_lb
            style = dict(color=col_, lw=2)
            axT.plot(solid.target, (solid.folds if solid.folds.notna().any()
                                    else solid.folds_lb), "-", **style)
            axT.plot(g.target, y, "--", alpha=0.6, **style)
            if g.folds.isna().all():
                # 0 observed: only a lower bound on cost exists.
                axT.annotate("", xy=(g.target.iloc[-1], y.iloc[-1] * 3.2),
                             xytext=(g.target.iloc[-1], y.iloc[-1]),
                             arrowprops=dict(arrowstyle="-|>", color=col_, lw=1.6))

            # -- bottom: substitutions among those designs ----------------
            # Drawn only out to 3x the observed passing designs. Beyond that
            # Chao2 is extrapolating undetected richness from too little.
            npass = g.n_pass_obs.iloc[0]
            if npass >= 2:
                axU.plot(solid.target, solid.U, "--", alpha=0.6, **style)
                axU.plot(meas.target, meas.U, "-", lw=3.2, color=col_, alpha=0.95)
                last = solid.iloc[-1]
                axU.annotate(f"n={npass}", (last.target, last.U),
                             textcoords="offset points", xytext=(5, -2),
                             fontsize=7, color=col_, va="center")

        # Name what could not be projected, rather than leaving a flat zero
        # line that reads as a measured result.
        bl = d[d.arm == "T=1.0"]
        if bl.n_pass_obs.iloc[0] == 0:
            axU.annotate("$T$=1.0: 0/96 passing\nno library at any budget",
                         (0.5, 0.55), xycoords="axes fraction", ha="center",
                         fontsize=8.5, color=C_BASE, style="italic")
        for ax in (axT, axU):
            ax.set_xscale("log")
            ax.grid(alpha=0.22, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for t in (500, 1000):
                ax.axvline(t, color="#D1D5DB", lw=0.9, zorder=0)
        axT.set_yscale("log")
        axT.set_title(f"{tier}\n{bb.replace('run_', '').split('_cond')[0]}",
                      fontsize=10, linespacing=1.5)
        axU.set_xlabel("passing designs in library", fontsize=9)

    axes[0][0].set_ylabel("designs folded", fontsize=9.5)
    axes[1][0].set_ylabel("distinct substitutions", fontsize=9.5)
    axes[0][0].plot([], [], "-", color=C_BASE, lw=2, label="LigandMPNN $T$=1.0")
    axes[0][0].plot([], [], "-", color=C_CLEO, lw=2, label="this paper")
    axes[0][0].plot([], [], "-", color=C_TEXT, lw=3.2, label="measured")
    axes[0][0].plot([], [], "--", color=C_TEXT, lw=2, alpha=0.6, label="projected")
    axes[0][0].legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.tight_layout()
    save(fig, "ame_efficiency")

    pd.set_option("display.width", 220, "display.float_format", "{:,.4g}".format)
    print(df[df.target.isin([500, 1000])][
        ["tier", "arm", "target", "k_obs", "n_obs", "train", "folds", "folds_lb", "U"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
