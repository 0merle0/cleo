#!/usr/bin/env python3
"""Figure 1 --- the method, and the benchmark that says it works.

Four-figure scheme, matching the story draft: this figure carries the
concept, the method and the AME benchmark together, which is what every
"figure 1?" in the draft points at.

    (a) three landscapes      (e) recombinants fold
    (b) fragment shuffling    (f) diversity over passing members
    (c) the GRPO objective    (g) where in sequence space
    (d) what it costs         (h) how fine it can be cut

Panels (e), (f) and (h) are POOLED over the three backbones with the
per-backbone points drawn on top. M0255 is materially harder than the other
two and drives every tail in this benchmark; a bare mean hides the one cell
a reader should ask about.

Eight panels makes this a tall figure -- 9.4 in against a 6.5 in page. If it
has to come down, (d) is the panel whose argument the text can carry alone.

    uv run python paper/figures/fig1_method.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ame_arms as AA  # noqa: E402
import landscape3d as L3  # noqa: E402
import panel_cost as COST  # noqa: E402
from concept_library import N_VARIANTS, _clip  # noqa: E402
from figio import save  # noqa: E402
from palette import (FRAGMENT, INK, PAGE_WIDTH_IN, PALETTE,  # noqa: E402
                     SEQUENTIAL, fs)

# --- the toy landscape ---------------------------------------------------
#
# THREE optima, not the four the 2D version carries. At the size this panel
# prints, a fourth peak is a fourth thing to track in a drawing whose whole
# argument is "the parent sits on one hill and there are others" -- three
# says it and leaves the mesh legible. concept_library keeps the four-peak
# 2D version; this landscape is defined here so changing one cannot silently
# change the other.
#
# Placed by their PROJECTED x (px = x + SHEAR*y), not their x. Two peaks a
# long way apart in the plane can land on top of each other in an oblique
# view: (0.73, 0.70) and (0.79, 0.25) project to 1.02 and 0.90 and read as
# one ridge. These three project to 0.39 / 0.66 / 1.03 and stay separate.
#
#        cx,   cy,   width, height
PEAKS = [
    (0.25, 0.34, 0.094, 0.74),   # the parent's own optimum -- good, not best
    (0.72, 0.74, 0.104, 1.00),   # the global optimum
    (0.61, 0.11, 0.086, 0.86),
]
START = (0.25, 0.34)

# Arm colours, used in (e), (f) and (g) so one arm is one colour throughout.
# Stock MPNN is grey: it is the control and should sit back rather than
# compete with the trained arms for weight.
ARMC = {"sampled": INK["muted"],
        "plain RL": PALETTE["yellow"],
        "chimera": SEQUENTIAL[3]}
MARK = {"M0097": "o", "M0255": "s", "M0315": "^"}


def fitness(x, y):
    """-> the toy landscape: three gaussian optima plus gentle ruggedness."""
    z = sum(a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
            for cx, cy, s, a in PEAKS)
    return z + 0.02 * np.sin(4 * np.pi * x) * np.sin(3.5 * np.pi * y)


def _zmax():
    """One zmax for all three landscapes.

    Normalising each panel to its own maximum would make three vertical
    scales out of one landscape, and the panel's entire job is that the
    landscape is the same in all three.
    """
    g = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(g, g)
    return float(np.max(fitness(X, Y)))


ZMAX = _zmax()


def _tidy(ax, ylab=None, xlab=None):
    """House axis treatment: two spines, recessive, labels at the floor size."""
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

def _surface(ax):
    L3.frame(ax)
    L3.draw_surface(ax, fitness, zmax=ZMAX)


def panel_epcr(ax, rng):
    """Mutagenesis walks: dense, and it never leaves the parent's basin."""
    _surface(ax)
    L3.title(ax, "Error-prone PCR")
    pts = _clip(rng.normal(START, 0.055, size=(N_VARIANTS, 2)))
    L3.draw_points(ax, fitness, pts, ZMAX)
    L3.draw_start(ax, fitness, START, ZMAX)
    L3.axis_cue(ax)


def panel_naive(ax, rng):
    """Combinatorial assembly with parts chosen without regard to fitness.

    Drawn as a PRODUCT SET rather than a cloud, because that is what assembly
    builds: 6 variants for block 1 crossed with 6 for block 2 is their 36
    combinations, not 36 free points.
    """
    _surface(ax)
    L3.title(ax, "Naive combinatorial")
    r = np.random.default_rng(50)
    xs = np.sort(r.uniform(0.05, 0.95, 6))
    ys = np.sort(r.uniform(0.05, 0.95, 6))
    pts = _clip(np.array([(x, y) for x in xs for y in ys], float)
                + rng.normal(0, 0.012, (36, 2)))
    L3.draw_points(ax, fitness, pts, ZMAX)
    L3.draw_start(ax, fitness, START, ZMAX)


def panel_ours(ax, rng):
    """Generate the same reach, then filter hard before building anything.

    The filter IS the search. The library spans the space as widely as the
    naive one -- the faint open candidates -- but only what survives folding
    and scoring is kept, and the survivors sit on the optima. The filled
    points number N_VARIANTS, identical to the other two panels, so the three
    differ in WHERE the budget goes and not in how much of it there is.
    """
    _surface(ax)
    L3.title(ax, "Filtered combinatorial")
    xs = rng.uniform(0.05, 0.95, 14)
    ys = rng.uniform(0.05, 0.95, 14)
    cand = _clip(np.array([(x, y) for x in xs for y in ys], float)
                 + rng.normal(0, 0.010, (196, 2)))
    keep = np.argsort(fitness(cand[:, 0], cand[:, 1]))[-N_VARIANTS:]
    mask = np.zeros(len(cand), bool)
    mask[keep] = True

    rej = cand[~mask]
    z = fitness(rej[:, 0], rej[:, 1]) / ZMAX + 0.012
    px, py = L3.iso(rej[:, 0], rej[:, 1], z)
    ax.plot(px, py, "o", ms=2.3, mfc="none", mec=INK["muted"], mew=0.5,
            ls="none", zorder=5)

    L3.draw_points(ax, fitness, cand[mask], ZMAX)
    L3.draw_start(ax, fitness, START, ZMAX)


# ================================================================ panel (b)

def panel_shuffle(ax):
    """The shuffling step: B parents, cut at k, recombined into chimeras.

    Drawn at B = 4 and k = 4 rather than the B = 16, N = 64 actually used.
    Sixteen four-block rows is a texture, not a diagram, and the thing the
    panel has to make legible is the ROTATION RULE -- chimera (rho, i) takes
    its j-th fragment from parent (i + rho*j) mod B. Four rows shows the rule
    exactly; the caption carries the real numbers.

    The rho = 0 row is drawn and labelled. A quarter of every chimera batch
    is the unrecombined parent, which inflates the in-loop pass rate and is
    the single most common misreading of these runs -- so it is on the
    diagram rather than in a footnote.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    B = K = 4
    bw, bh = 0.050, 0.060
    gap = 0.005

    def row(x0, y0, parents, boxed=False):
        for j, p in enumerate(parents):
            ax.add_patch(plt.Rectangle(
                (x0 + j * (bw + gap), y0), bw, bh,
                facecolor=FRAGMENT[p], edgecolor="white", linewidth=0.8,
                zorder=3))
        if boxed:
            ax.add_patch(plt.Rectangle(
                (x0 - 0.006, y0 - 0.006), K * (bw + gap) - gap + 0.012,
                bh + 0.012, facecolor="none", edgecolor=PALETTE["red"],
                linewidth=1.0, zorder=4))

    # --- left: the decoded batch -----------------------------------------
    ax.text(0.000, 0.98, "cut $B$ parents into $k$", fontsize=fs(7.5),
            color="black", ha="left", va="top")
    for i in range(B):
        y = 0.76 - i * (bh + 0.038)
        row(0.030, y, [i] * K)
        ax.text(0.020, y + bh / 2, f"{i}", fontsize=fs(6.5),
                color=INK["muted"], ha="right", va="center")

    # --- arrow ------------------------------------------------------------
    ax.annotate("", xy=(0.340, 0.655), xytext=(0.272, 0.655),
                arrowprops=dict(arrowstyle="-|>", color=INK["muted"], lw=1.1))

    # --- right: the chimeras ---------------------------------------------
    ax.text(0.365, 0.98, "recombine across parents", fontsize=fs(7.5),
            color="black", ha="left", va="top")
    ax.text(0.365, 0.893,
            r"fragment $j$ from parent $(i+\rho j)\ \mathrm{mod}\ B$",
            fontsize=fs(6.5), color=INK["muted"], ha="left", va="top")

    for r, rho in enumerate((0, 1, 2)):
        y = 0.76 - r * (bh + 0.038)
        for c, i in enumerate((0, 1)):
            x0 = 0.395 + c * (K * (bw + gap) + 0.050)
            row(x0, y, [(i + rho * j) % B for j in range(K)], boxed=rho == 0)
        ax.text(0.385, y + bh / 2, rf"$\rho={rho}$", fontsize=fs(6.5),
                color=INK["muted"], ha="right", va="center")

    ax.text(0.395, 0.76 - 2 * (bh + 0.038) - 0.045,
            r"$\rho=0$ returns the parent:"  "\n"
            r"a quarter of every batch",
            fontsize=fs(6.5), color=PALETTE["red"], ha="left", va="top")

    # --- the credit assignment -------------------------------------------
    ax.plot([0.000, 1.000], [0.300, 0.300], color=INK["grid"], lw=0.9)
    ax.text(0.000, 0.240,
            "fold best-of-5, score, then average each fragment's\n"
            "reward over every chimera that contains it",
            fontsize=fs(7), color="black", ha="left", va="top",
            linespacing=1.5)
    ax.text(0.000, 0.055,
            r"in training $B=16$, $N_{\rm chimera}=64$: every fragment "
            r"scored 4 times",
            fontsize=fs(6.5), color=INK["muted"], ha="left", va="top")


# ================================================================ panel (c)

def panel_objective(ax):
    """The GRPO objective, written over fragments rather than sequences.

    A typeset block, not a plot. It sits beside (b) on purpose: every symbol
    in it is a step of the diagram to its left, and separating the two is
    what made the draft's reference to it unreadable.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.00, 0.045), 1.00, 0.955,
                               facecolor="#FAFBFC", edgecolor=INK["grid"],
                               linewidth=0.8, zorder=0))

    ax.text(0.5, 0.905,
            r"$J(\theta)=E\,[\,\frac{1}{kB}"
            r"\sum_{i,j}\min(\rho_{ij}A_{ij},\;"
            r"\mathrm{clip}(\rho_{ij})\,A_{ij})\,]$",
            fontsize=fs(8), color="black", ha="center", va="center")

    ax.text(0.5, 0.715,
            r"$\rho_{ij}=\frac{\pi_\theta(f_{ij}\,|\,\mathrm{str})}"
            r"{\pi_{\theta_{\mathrm{old}}}(f_{ij}\,|\,\mathrm{str})}$",
            fontsize=fs(7.5), color="black", ha="center", va="center")

    ax.text(0.5, 0.545,
            r"$A_{ij}=\frac{R(f_{ij})-\mu}{\sigma+\epsilon}$",
            fontsize=fs(7.5), color="black", ha="center", va="center")

    # Two lines, because Arial has no U+2208 and the spelled-out version is
    # wider than the panel. Splitting is better than shrinking: the font
    # floor is 7 pt and this is already at it.
    ax.text(0.5, 0.425, r"$R(f_{ij})=\mathrm{mean}\ R(S)$",
            fontsize=fs(7.5), color="black", ha="center", va="center")
    ax.text(0.5, 0.350, r"over every chimera $S$ containing $f_{ij}$",
            fontsize=fs(7), color="black", ha="center", va="center")

    ax.text(0.045, 0.300,
            "likelihood is over FRAGMENTS,\nnot whole sequences",
            fontsize=fs(6.5), color=INK["muted"], ha="left", va="top",
            linespacing=1.45)
    ax.text(0.045, 0.172,
            r"$\mu,\sigma$ over the rollout, so a"
            "\nwhole-batch oracle offset cancels",
            fontsize=fs(6.5), color=INK["muted"], ha="left", va="top",
            linespacing=1.45)


# ================================================================ panel (d)

def panel_cost(ax):
    """Folds to a library of size S: brute-force filtering against training.

    The y axis is FOLDS, not steps and not dollars. Training is flat in S
    because the enrichment lives in the policy rather than in the members --
    once trained, every design drawn from it inherits the pass rate without
    being folded. Brute force is linear in S because every candidate has to
    be folded to find out whether it passes.

    Below the crossover brute force wins, and the panel says so by drawing
    the crossing rather than starting the axis past it.
    """
    S = np.logspace(2, 12, 200)
    for p in (0.258, 0.496, 0.598):
        ax.plot(S, COST.brute_force_folds(S, p), lw=1.0,
                color=INK["muted"], alpha=0.8, zorder=2)
    ax.plot([], [], lw=1.0, color=INK["muted"], label="fold every candidate")
    ax.axhline(COST.TRAIN_FOLDS, color=SEQUENTIAL[3], lw=1.6, zorder=3,
               label="post-train once")

    # A band, not a point: the three backbones' vanilla pass rates put the
    # crossover in three different places.
    lo, hi = (COST.crossover(p) for p in (0.258, 0.598))
    ax.axvspan(lo, hi, color=PALETTE["yellow"], alpha=0.22, lw=0, zorder=1)
    ax.text(np.sqrt(lo * hi), 4e3, "break-even", fontsize=fs(6),
            color="black", ha="center", va="bottom", rotation=90)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e2, 1e12)
    ax.set_ylim(1e3, 1e13)
    ax.set_xticks([1e2, 1e6, 1e10])
    _tidy(ax, ylab="Structure predictions", xlab="Library members")
    ax.legend(fontsize=fs(6), loc="upper left", frameon=False,
              handlelength=1.2, borderpad=0.0, labelspacing=0.25,
              handletextpad=0.4)


# ================================================================ panels e/f

def _arm_panel(ax, d, col, ylab):
    """Pooled bar per arm, with every backbone drawn on top of it.

    The bar is the mean over backbones (and over seeds, where an arm has
    them). The points are what the mean is made of, and they stay because
    M0255 is materially harder than the other two and drives every tail in
    this benchmark -- a bare mean hides the one cell a reader should ask
    about.
    """
    arms = list(AA.ARMS)
    for x, arm in enumerate(arms):
        g = d[d.arm == arm]
        ax.bar(x, g[col].mean(), width=0.64, color=ARMC[arm], alpha=0.28,
               edgecolor=ARMC[arm], linewidth=1.1, zorder=2)
        jitter = np.linspace(-0.18, 0.18, len(g))
        for (_, r), dx in zip(g.sort_values("site").iterrows(), jitter):
            ax.plot(x + dx, r[col], MARK[r.site], ms=3.4, mfc="white",
                    mec=ARMC[arm], mew=1.0, zorder=4)

    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(["stock\nMPNN", "plain\nRL", "chimera\n+pool"],
                       fontsize=fs(6.5))
    _tidy(ax, ylab=ylab)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.62, len(arms) - 0.38)


def panel_pass(ax, d):
    """Recombinant pass rate. The headline, and it is unambiguous."""
    _arm_panel(ax, d, "pass_rate", "Recombinants passing (%)")
    ax.set_ylim(0, 115)
    for x, arm in enumerate(AA.ARMS):
        m = d[d.arm == arm].pass_rate.mean()
        ax.text(x, 106, f"{m:.0f}", fontsize=fs(7), ha="center", va="center",
                color=ARMC[arm], fontweight="bold")


def panel_diversity(ax, d):
    """Rarefied distinct substitutions over PASSING members.

    THIS PANEL DOES NOT SAY WHAT THE DRAFT ASSUMES, and that is the reason to
    draw it. The chimera arm carries more diversity than stock MPNN on two
    backbones and LESS on M0315 (1054 against 1237), and plain RL carries
    more than either on all three. Post-training for recombination buys pass
    rate; it does not also win diversity, and the text should claim the first
    and not the second.

    Rarefied because an arm that passes more often has more survivors to find
    substitutions in. Every arm on a backbone is subsampled to a common
    survivor count first, so a difference is coverage and not sample size.
    """
    _arm_panel(ax, d, "subs", "Distinct substitutions\n(rarefied)")
    ax.set_ylim(0, 2100)
    for x, arm in enumerate(AA.ARMS):
        m = d[d.arm == arm].subs.mean()
        ax.text(x, 1960, f"{m:.0f}", fontsize=fs(7), ha="center", va="center",
                color=ARMC[arm], fontweight="bold")


# ================================================================ panel (g)

def _mutant_library(parent, rng, n_each=400):
    """-> single and double mutants of `parent`: the error-prone PCR arm.

    Sampled rather than enumerated. All 180x19 = 3,420 singles would swamp
    every other arm in the scatter, and the panel is about WHERE each library
    sits rather than how many members it has.
    """
    L = len(parent)
    out = []
    for n_mut in (1, 2):
        for _ in range(n_each):
            s = list(parent)
            for p in rng.choice(L, n_mut, replace=False):
                s[p] = AA.AA[rng.integers(20)]
            out.append("".join(s))
    return out


def collect_space(site="M0097", seed=0):
    """-> (PC1/PC2 by arm, explained variance, the parent's coordinates).

    One PCA per backbone, never one across all three: they are different
    proteins, and a shared embedding would resolve backbone identity and
    nothing else. M0097 is shown; the other two belong in supplementary.

    The PARENT is the highest-ranked PASSING stock-MPNN recombinant -- the
    hit a directed-evolution campaign would actually start from. Mutating a
    consensus instead would anchor the mutagenesis arm on a sequence that was
    never built or folded.
    """
    rng = np.random.default_rng(seed)

    samp = AA.load(site, "sampled")[0][1]
    post = AA.load(site, "chimera")[0][1]
    sp = samp[samp.rfd2_any_pass].drop_duplicates("sequence")
    pp = post[post.rfd2_any_pass].drop_duplicates("sequence")
    parent = (sp.sort_values("af3_ranking_score", ascending=False)
              .sequence.iloc[0])

    groups = [("single + double", _mutant_library(parent, rng)),
              ("stock MPNN", sp.sequence.tolist()),
              ("chimera + pool", pp.sequence.tolist())]

    # ONE matrix, centred and decomposed ONCE. That is what puts the three
    # libraries in the same space -- three PCAs fitted separately would give
    # three sets of axes that happen to be drawn on top of each other.
    X = AA.onehot([s for _, g in groups for s in g] + [parent])
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    Z = U[:, :2] * S[:2]
    var = 100 * S[:2] ** 2 / (S ** 2).sum()

    rows, i = [], 0
    for name, g in groups:
        for z in Z[i:i + len(g)]:
            rows.append(dict(arm=name, pc1=z[0], pc2=z[1]))
        i += len(g)
    return pd.DataFrame(rows), var, Z[-1]


def panel_space(ax, d, var, pz):
    """All three libraries projected into one PCA of their one-hot sequences.

    PC1 carries ~48% and is very nearly distance from the parent, so this
    panel largely restates the Hamming numbers rather than adding independent
    evidence -- say so in the caption. What it adds is the SHAPE: at this
    scale the mutagenesis library is a point, stock MPNN is a loose cloud,
    and the post-trained library is a tight cloud somewhere else entirely.
    """
    colour = {"single + double": PALETTE["red"],
              "stock MPNN": INK["muted"],
              "chimera + pool": SEQUENTIAL[3]}
    for arm in ("single + double", "stock MPNN", "chimera + pool"):
        g = d[d.arm == arm]
        ax.plot(g.pc1, g.pc2, "o", ms=2.4, mfc=colour[arm], mec="none",
                alpha=0.55, ls="none", rasterized=True, zorder=3, label=arm)

    ax.plot(pz[0], pz[1], marker="*", ms=8, mfc="white", mec="black",
            mew=1.0, ls="none", zorder=6)
    ax.annotate("parent", xy=(pz[0], pz[1]), xytext=(5, -10),
                textcoords="offset points", fontsize=fs(6.5), color="black")

    _tidy(ax, ylab=f"PC2 ({var[1]:.0f}%)", xlab=f"PC1 ({var[0]:.0f}%)")
    ax.legend(fontsize=fs(6.5), loc="upper left", frameon=False,
              handlelength=0.7, borderpad=0.0, labelspacing=0.2,
              handletextpad=0.35, markerscale=2.0)


# ================================================================ panel (h)

def panel_k(ax):
    """Retention against k, pooled over the three backbones.

    Two conditions: a policy TRAINED at each k, and the single k=4 policy
    re-cut at each k post hoc. The gap between them is the result -- it is
    not that fine cuts are impossible, it is that a policy has to be trained
    at the granularity it will be cut at.

    The band is MIN-MAX over the three backbones, not a confidence interval.
    Every cell is one seed, against a documented 21-point run-to-run swing on
    pass rate, so the band is spread across backbones and says nothing about
    sampling error.
    """
    d = pd.read_csv(HERE / "data" / "ame_fragk.csv")
    for cond, lab, c in (("trained", "trained at $k$", SEQUENTIAL[3]),
                         ("posthoc", r"$k\!=\!4$ policy, cut at $k$",
                          INK["muted"])):
        g = (d[d.cond == cond].groupby("k").retention
             .agg(["mean", "min", "max"]).reset_index())
        ax.fill_between(g.k, 100 * g["min"], 100 * g["max"], color=c,
                        alpha=0.16, lw=0, zorder=2)
        ax.plot(g.k, 100 * g["mean"], "-o", color=c, lw=1.4, ms=3.2,
                mfc="white", mew=1.1, zorder=3, label=lab)

    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 36, 64])
    ax.set_xticklabels(["2", "4", "8", "16", "36", "64"])
    ax.set_ylim(0, 112)
    _tidy(ax, ylab="Recombinants passing (%)",
          xlab="Fragments per design, $k$")
    ax.legend(fontsize=fs(6.5), loc="lower left", frameon=False,
              handlelength=1.3, borderpad=0.0, labelspacing=0.25,
              handletextpad=0.4)


# ================================================================== assembly

def main():
    arms = AA.collect()
    space, var, pz = collect_space()

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 9.4))
    outer = fig.add_gridspec(
        4, 1, height_ratios=[0.74, 1.00, 1.02, 1.02],
        left=0.085, right=0.985, top=0.968, bottom=0.048, hspace=0.52)

    # (a) three landscapes, one budget spent three ways.
    row = outer[0].subgridspec(1, 3, wspace=0.07)
    rng = np.random.default_rng(7)
    for i, fn in enumerate((panel_epcr, panel_naive, panel_ours)):
        fn(fig.add_subplot(row[0, i]), rng)

    # (b) the shuffling scheme, (c) the objective it feeds.
    mid = outer[1].subgridspec(1, 2, width_ratios=[1.70, 1.0], wspace=0.10)
    panel_shuffle(fig.add_subplot(mid[0, 0]))
    panel_objective(fig.add_subplot(mid[0, 1]))

    # (d) cost, (e) pass rate, (f) diversity.
    r3 = outer[2].subgridspec(1, 3, width_ratios=[1.25, 1.0, 1.0],
                              wspace=0.52)
    panel_cost(fig.add_subplot(r3[0, 0]))
    panel_pass(fig.add_subplot(r3[0, 1]), arms)
    panel_diversity(fig.add_subplot(r3[0, 2]), arms)

    # (g) sequence space, (h) granularity.
    r4 = outer[3].subgridspec(1, 2, wspace=0.28)
    panel_space(fig.add_subplot(r4[0, 0]), space, var, pz)
    panel_k(fig.add_subplot(r4[0, 1]))

    # Panel letters in FIGURE coordinates. ax.get_tightbbox under-measures
    # rotated text, which is how panel letters end up sitting on y-labels --
    # so these are placed explicitly against the gridspec rows rather than
    # measured off each axes.
    for letter, (x, y) in zip("abcdefgh", [
            (0.008, 0.980), (0.008, 0.805), (0.632, 0.805),
            (0.008, 0.545), (0.345, 0.545), (0.668, 0.545),
            (0.008, 0.272), (0.510, 0.272)]):
        fig.text(x, y, f"({letter})", fontsize=fs(9), fontweight="bold",
                 ha="left", va="top")

    save(fig, "fig1_method", dpi=900, tight=False, subdir="main")


if __name__ == "__main__":
    main()
