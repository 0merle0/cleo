#!/usr/bin/env python3
"""Aligning a model to a function nobody can see.

The argument in four panels:

    (a) there is a true sequence-function landscape, and it is not observable
    (b) an assay samples it at a few thousand points
    (c) a model fitted to those points reconstructs it
    (d) how well, as a function of how many points you bought

THE SURFACE IN (c) IS FITTED, NOT DRAWN. A Gaussian process is trained on the
sampled points alone and its posterior mean is contoured on the same scale as
the truth. Panel (d) is the same fit repeated over a sweep of sample sizes
with the agreement measured against the held-out truth. So the figure makes a
claim that could have come out wrong -- if sparse sampling did not recover
this landscape, the panels would show that.

It uses the SAME toy landscape as Figure 1(a), imported rather than
redefined, so a reader who has seen that figure is looking at the same
terrain and the two arguments compose.

Spearman, not R^2, in (d): the thing a design campaign needs from a model is
the ability to RANK candidates, and a model can rank perfectly while being
badly calibrated in absolute units.

    uv run python paper/figures/concept_alignment.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from concept_library import fitness  # noqa: E402  -- the same landscape as Fig 1a
from figio import save  # noqa: E402
from palette import INK, PAGE_WIDTH_IN, PALETTE, SEQUENTIAL, fs  # noqa: E402

GRID = 220          # contour resolution
N_SHOW = 90         # points drawn in (b); the sample the model in (c) sees
NOISE = 0.05        # assay noise, as a fraction of the landscape's range
SWEEP = [10, 20, 40, 80, 160, 320, 640]
REPEATS = 12        # independent draws per sample size in (d)

TERRAIN = LinearSegmentedColormap.from_list(
    "cleo_terrain", ["#FFFFFF", "#FFFFFF", "#DCEBFB", SEQUENTIAL[0],
                     SEQUENTIAL[1], SEQUENTIAL[2], SEQUENTIAL[3]])
DOT = "#1F2937"


def _grid():
    g = np.linspace(0, 1, GRID)
    X, Y = np.meshgrid(g, g)
    return X, Y, fitness(X, Y)


def _frame(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(INK["grid"])
        sp.set_linewidth(0.9)
    ax.set_title(title, fontsize=fs(8), pad=5, color="black")


def _contour(ax, X, Y, Z, levels, alpha=1.0):
    ax.contourf(X, Y, Z, levels=levels, cmap=TERRAIN, zorder=0, alpha=alpha)
    ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.5,
               alpha=0.85 * alpha, zorder=1)


def sample(rng, n, noise=NOISE, span=None):
    """-> (points, measured values). The assay: sparse, and noisy."""
    P = rng.uniform(0.02, 0.98, size=(n, 2))
    true = fitness(P[:, 0], P[:, 1])
    span = span if span is not None else (true.max() - true.min())
    return P, true + rng.normal(0, noise * span, n)


def fit_model(P, y):
    """-> a GP trained on the sampled points only.

    The kernel carries a WhiteKernel because the measurements are noisy and
    a noiseless GP would interpolate that noise exactly, producing a surface
    pocked with spikes at every sample. That is not a modelling nicety: it
    is the difference between a model that generalises and one that has
    memorised its training set.
    """
    k = (ConstantKernel(1.0, (1e-2, 1e2))
         * RBF(0.12, (2e-2, 1.0))
         + WhiteKernel(1e-2, (1e-6, 1e0)))
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True,
                                  n_restarts_optimizer=2, random_state=0)
    gp.fit(P, y)
    return gp


# ================================================================== panels

def panel_truth(ax, X, Y, Z, levels):
    """The landscape as it is. Nobody gets to see this one."""
    _contour(ax, X, Y, Z, levels)
    _frame(ax, "Ground truth")
    ax.set_ylabel("Sequence space", fontsize=fs(7.5), labelpad=3)


def panel_data(ax, X, Y, Z, levels, P, y):
    """What an assay returns: a few points, with error, and nothing else.

    The landscape is drawn at low opacity rather than omitted. Omitted, the
    panel is a scatter of dots in a void and the reader loses the thread;
    ghosted, it is clear that the surface is there and simply not visible to
    the experiment.
    """
    _contour(ax, X, Y, Z, levels, alpha=0.18)
    sc = ax.scatter(P[:, 0], P[:, 1], c=y, cmap=TERRAIN, s=15,
                    edgecolors=DOT, linewidths=0.45, zorder=4,
                    vmin=Z.min(), vmax=Z.max())
    _frame(ax, f"Measured ({len(P)} assayed)")
    return sc


def panel_model(ax, X, Y, Z, levels, gp):
    """The model's landscape: the GP posterior mean, on the truth's scale.

    Same contour levels as (a), so the two panels are directly comparable --
    re-levelling each to its own range would make any model look like the
    truth.
    """
    pred = gp.predict(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    _contour(ax, X, Y, pred, levels)

    # The truth's optima, ringed on the model's surface. This is the panel's
    # actual claim: not that the surfaces look alike, but that the model puts
    # its maxima where the real ones are.
    for cx, cy in _optima(Z, X, Y):
        ax.plot(cx, cy, "o", ms=9, mfc="none", mec=PALETTE["red"], mew=1.2,
                zorder=6)
    _frame(ax, "Model, fitted to those points")


def _optima(Z, X, Y, k=4, min_sep=0.12):
    """-> the k highest, mutually separated maxima of the true landscape."""
    order = np.argsort(Z.ravel())[::-1]
    xs, ys = X.ravel(), Y.ravel()
    out = []
    for i in order:
        p = (xs[i], ys[i])
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > min_sep ** 2
               for q in out):
            out.append(p)
        if len(out) == k:
            break
    return out


def panel_sweep(ax, rng, X, Y, Z):
    """Agreement with the truth against the number of measurements.

    Each point is `REPEATS` independent draws at that sample size, scored by
    Spearman against the true landscape on a held-out grid. The band is the
    full range over those draws, not a standard error: at n = 10 the spread
    is the result -- whether the model is any good depends on which ten
    points you happened to measure -- and a standard error would hide it.
    """
    Q = np.column_stack([X.ravel(), Y.ravel()])
    truth = Z.ravel()
    sub = rng.choice(len(Q), 4000, replace=False)   # held-out scoring grid
    span = Z.max() - Z.min()

    med, lo, hi = [], [], []
    for n in SWEEP:
        rs = []
        for _ in range(REPEATS):
            P, y = sample(rng, n, span=span)
            pred = fit_model(P, y).predict(Q[sub])
            rs.append(spearmanr(pred, truth[sub]).statistic)
        rs = np.array(rs)
        med.append(np.median(rs))
        lo.append(rs.min())
        hi.append(rs.max())

    ax.fill_between(SWEEP, lo, hi, color=SEQUENTIAL[2], alpha=0.20, lw=0,
                    zorder=2)
    ax.plot(SWEEP, med, "-o", color=SEQUENTIAL[4], lw=1.5, ms=4, mfc="white",
            mew=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(SWEEP)
    ax.set_xticklabels([str(n) for n in SWEEP], fontsize=fs(6.5))
    ax.set_ylim(-0.05, 1.03)
    ax.axhline(0, color=INK["muted"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_xlabel("Variants measured", fontsize=fs(7.5))
    ax.set_ylabel("Agreement with ground truth\n(Spearman)", fontsize=fs(7.5))
    ax.tick_params(labelsize=fs(7), colors="black", width=0.8, length=2.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK["muted"])
        ax.spines[sp].set_linewidth(0.8)
    return med


def main():
    rng = np.random.default_rng(3)
    X, Y, Z = _grid()
    levels = np.linspace(Z.min(), Z.max(), 13)

    P, y = sample(rng, N_SHOW)
    gp = fit_model(P, y)

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 4.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], left=0.085,
                             right=0.985, top=0.935, bottom=0.105, hspace=0.42)

    top = outer[0].subgridspec(1, 3, wspace=0.10)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(top[0, 2])
    panel_truth(ax_a, X, Y, Z, levels)
    panel_data(ax_b, X, Y, Z, levels, P, y)
    panel_model(ax_c, X, Y, Z, levels, gp)

    bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30)
    med = panel_sweep(fig.add_subplot(bot[0, 0]), rng, X, Y, Z)

    ax_t = fig.add_subplot(bot[0, 1])
    ax_t.axis("off")
    ax_t.text(0.0, 0.95,
              "The landscape is real and unobservable.\n"
              "The assay returns a sparse, noisy sample of it.\n"
              "A model fitted to that sample recovers where the\n"
              "optima are, and the recovery is a function of how\n"
              "many variants were bought.",
              fontsize=fs(7.5), color="black", ha="left", va="top",
              linespacing=1.65)
    ax_t.text(0.0, 0.30,
              f"red rings in (c): the true optima\n"
              f"band in (d): full range over {REPEATS} independent draws",
              fontsize=fs(6.5), color=INK["muted"], ha="left", va="top",
              linespacing=1.5)

    # Placed against the gridspec columns, not guessed: the top row is
    # aspect-equal, so each axes is a square fitted and CENTRED inside its
    # cell and its real left edge is not the cell's left edge. Letters sit
    # just outside the squares rather than on their titles.
    for letter, (x, yy) in zip("abcd", [(0.012, 0.985), (0.345, 0.985),
                                        (0.660, 0.985), (0.012, 0.470)]):
        fig.text(x, yy, f"({letter})", fontsize=fs(9), fontweight="bold",
                 ha="left", va="top")

    save(fig, "concept_alignment", dpi=900, tight=False)

    print("agreement with ground truth, median over "
          f"{REPEATS} draws:")
    for n, m in zip(SWEEP, med):
        print(f"  {n:4d} measured  rho = {m:.3f}")


if __name__ == "__main__":
    main()
