#!/usr/bin/env python3
"""Figure 1 --- the method, and the benchmark that says it works.

Four-figure scheme, matching the story draft: this figure carries the
concept, the method and the AME benchmark together, which is what every
"figure 1?" in the draft points at.

    (a) three landscapes      (e) diversity vs pass rate
    (b) fragment shuffling    (f) oracle calls vs library size
    (c) the GRPO objective    (g) where in sequence space
    (d) DNA cost at several k (h) how fine it can be cut

(e) and (f) are both pooled, each with its three backbones drawn on the
bars. On pass rate the three agree and the points corroborate; on diversity
they do NOT -- M0315 falls while the mean rises -- so on (f) the points are
the result and the panel must never be shown without them. (j) is pooled
with a min-max band.

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
import concept_library as CL  # noqa: E402
import motif_variance as MV  # noqa: E402
import panel_cost as COST  # noqa: E402
from figio import save  # noqa: E402
from palette import (FRAGMENT, INK, PAGE_WIDTH_IN, PALETTE,  # noqa: E402
                     SEQUENTIAL, fs)

# Arm colours, used in (e) and (g) so one arm is one colour throughout.
# Stock MPNN is grey: it is the control and should sit back rather than
# compete with the trained arms for weight.
# ONE blue for this method, a step darker than the mid ramp, used in every
# panel where the method appears -- the bars, the trained-at-k curve, the
# sequence-space cloud, the motif boxes, the post-train line in the cost
# panel. A method that is two different blues across one figure reads as two
# different things.
METHOD = SEQUENTIAL[4]

ARMC = {"sampled": INK["muted"],
        "plain RL": PALETTE["yellow"],
        "chimera": METHOD}
MARK = {"M0097": "o", "M0255": "s", "M0315": "^"}


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
#
# THE 2D LANDSCAPES FROM clean.pdf, not a new drawing. `concept_library` has
# carried this panel since the first version of the figure -- filled contour
# seen from above, four optima, a fitness key and a shared legend -- and it
# is what the argument has always been made with.
#
# A 3D version was built (soft shaded surface, three peaks, a Function axis;
# see landscape3d.py) and is NOT used here. It is kept in the tree rather
# than deleted: swapping back is changing which three functions panel_a
# calls. The 2D version reads better at this size, which is the whole reason
# it was 2D to begin with -- an oblique view spends vertical space on depth
# that a contour spends on the landscape itself.

def panel_landscapes(fig, gs, rng=None):
    """The three-arm landscape row, exactly as clean.pdf composes it.

    Four columns -- three landscapes and the fitness key -- over one legend
    strip. The key and the legend are not decoration: without the key the
    contour shading has no referent, and without the legend the open circles
    in the third panel (candidates filtered OUT) read as a third class of
    design rather than as the filter doing its job.
    """
    rng = rng or np.random.default_rng(7)
    top = gs.subgridspec(2, 1, height_ratios=[1.0, 0.13], hspace=0.06)
    land = top[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.17], wspace=0.11)
    CL.panel_epcr(fig.add_subplot(land[0, 0]), rng)
    CL.panel_naive(fig.add_subplot(land[0, 1]), rng)
    CL.panel_ours(fig.add_subplot(land[0, 2]), rng)
    CL.fitness_key(fig.add_subplot(land[0, 3]))
    CL.landscape_legend(fig.add_subplot(top[1]))


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
    """What a given spend buys, as a library size, at several k.

    This is `concept_library.panel_cost`, which has been the cost panel all
    along -- whole gene, k=2, k=4, k=8 -- and is the one the argument needs.
    It was briefly replaced here by a folds-vs-library-size panel; that is a
    different and also-true argument about ORACLE cost, and it now lives in
    `panel_folds` below rather than displacing this one.

    Axes are cost on x and library size on y: the question is what you GET
    for what you PAY. One hue dark to light, because k is ordered and whole
    genes are just k=1 on the same ramp.
    """
    CL.panel_cost(ax)
    ax.tick_params(labelsize=fs(7))
    ax.xaxis.label.set_size(fs(7.5))
    ax.yaxis.label.set_size(fs(7.5))


def panel_folds(ax):
    """Structure predictions to reach a library of size S: filter vs train.

    Kept as a separate panel rather than merged into (d). (d) is about DNA
    dollars and (c) about oracle calls -- same shape of argument, different
    currency, and one axis cannot carry both.

    Training is flat in S because the enrichment lives in the policy rather
    than in the members: once trained, every design drawn from it inherits
    the pass rate without being folded. Brute force is linear because every
    candidate has to be folded to find out whether it passes. Below the
    crossover brute force wins, and the panel draws the crossing rather than
    starting the axis past it.
    """
    S = np.logspace(2, 12, 200)
    for p in (0.258, 0.496, 0.598):
        ax.plot(S, COST.brute_force_folds(S, p), lw=1.0,
                color=INK["muted"], alpha=0.8, zorder=2)
    ax.plot([], [], lw=1.0, color=INK["muted"], label="fold every candidate")
    ax.axhline(COST.TRAIN_FOLDS, color=METHOD, lw=1.6, zorder=3,
               label="post-train once")

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

# "This work" rather than "post-trained", to match the third landscape in
# panel (a). One name for one thing across the whole figure.
ARM_LABEL = {"sampled": "stock ProteinMPNN", "plain RL": "plain RL",
             "chimera": "This work"}

SITES = ("M0097", "M0255", "M0315")


def _grouped_bars(ax, d, col, ylab, arms, legend=False, fmt="{:.0f}"):
    """Grouped bars: one group per backbone, one bar per arm.

    Per backbone rather than pooled. The move from stock to post-trained is
    not the same move on all three -- M0315 reverses on diversity -- and a
    pooled bar with the three points scattered on it leaves the reader to
    pair them up by eye. Three groups of two says it directly.

    Arms with replicate seeds (plain RL has two) are averaged within the
    backbone, since the bar is that backbone's value for that arm.
    """
    n = len(arms)
    width = 0.78 / n
    for j, arm in enumerate(arms):
        xs, ys = [], []
        for i, site in enumerate(SITES):
            g = d[(d.site == site) & (d.arm == arm)]
            if not len(g):
                continue
            xs.append(i + (j - (n - 1) / 2) * width)
            ys.append(g[col].mean())
        ax.bar(xs, ys, width=width * 0.88, color=ARMC[arm], alpha=0.85,
               edgecolor="white", linewidth=0.6, zorder=2,
               label=ARM_LABEL[arm] if legend else None)
        for x, y in zip(xs, ys):
            ax.text(x, y, " " + fmt.format(y), rotation=90, ha="center",
                    va="bottom", fontsize=fs(6), color=ARMC[arm], zorder=4)

    ax.set_xticks(range(len(SITES)))
    ax.set_xticklabels(SITES, fontsize=fs(6.5))
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(SITES) - 0.4)
    _tidy(ax, ylab=ylab)
    if legend:
        ax.legend(fontsize=fs(6.5), loc="upper left", frameon=False,
                  handlelength=1.0, borderpad=0.0, labelspacing=0.25,
                  handletextpad=0.4)


def _pooled_bars(ax, d, col, ylab, arms, legend=False):
    """One bar per arm, pooled over backbones, with the backbones on it.

    The bar is the mean over the three backbones; the points are the three
    values it is made of. Both are needed: the bar is the claim and the
    points are what stops it being an average rescued by one backbone.

    No value labels. They were tried inside the bars in white and above
    them in colour; the first fails when a figure is printed or downsampled,
    the second lands in the points. The axis carries the numbers.
    """
    xs = np.arange(len(arms))
    for x, arm in zip(xs, arms):
        g = d[d.arm == arm]
        m = g[col].mean()
        # Solid fill, no alpha: alpha over white makes a FOURTH blue that is
        # not in the ramp, which is how a figure ends up with the cost plot's
        # k=4 line and this bar looking like different colours.
        ax.bar(x, m, width=0.56, color=ARMC[arm], edgecolor="white",
               linewidth=0.6, zorder=2)

        pts = []
        for i, site in enumerate(SITES):
            h = g[g.site == site]
            if not len(h):
                continue
            v = h[col].mean()
            pts.append(v)
            ax.plot(x + np.linspace(-0.16, 0.16, len(SITES))[i], v,
                    MARK[site], ms=4.2, mfc="white", mec=INK["ink"],
                    mew=1.0, zorder=4)

        _ = pts

    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABEL[a].replace(" ", "\n", 1) for a in arms],
                       fontsize=fs(6.5))
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    _tidy(ax, ylab=ylab)

    if legend:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker=MARK[s_], ms=4.2,
                                  mfc="white", mec=INK["ink"], mew=1.0,
                                  ls="none", label=s_) for s_ in SITES],
                  fontsize=fs(6), loc="upper left", frameon=False,
                  handlelength=0.8, borderpad=0.0, labelspacing=0.2,
                  handletextpad=0.4)


def panel_pass(ax, d, arms=("sampled", "chimera")):
    """Recombinant pass rate, pooled over the three backbones.

    All three gain, and the smallest gain is +26 points, so the bar is the
    headline and the points corroborate it.
    """
    ax.set_ylim(0, 108)
    _pooled_bars(ax, d, "pass_rate", "Recombinants passing (%)", arms,
                 legend=True)


def panel_diversity(ax, d, arms=("sampled", "chimera")):
    """Rarefied distinct substitutions over PASSING members, pooled.

    THE POINTS ARE LOAD-BEARING HERE IN A WAY THEY ARE NOT IN (e). The bars
    say diversity rises, 900 -> 981. The points say that is two backbones
    rising and ONE FALLING: M0315 goes 1238 -> 1045 while its pass rate goes
    to 98%. The pooled bar alone would support a claim the data does not --
    see figure_plan_story.md section 7 -- so this panel must never be shown
    with the points stripped off.

    Rarefied because an arm that passes more often has more survivors to find
    substitutions in. Every arm on a backbone is subsampled to a common
    survivor count first, so a difference is coverage and not sample size.
    """
    ax.set_ylim(0, 1450)
    _pooled_bars(ax, d, "subs", "Distinct substitutions\n(rarefied)", arms)


def panel_motif_var(ax, md, arms=("sampled", "chimera")):
    """How tightly each library holds its catalytic atoms.

    y is `rfd2_motif_rmsd` -- the predicted motif's RMSD to the theozyme it
    was scaffolded around -- one point per DESIGN, averaged over its five
    diffusion samples. Its SPREAD is the quantity: a library whose members
    all place the catalytic atoms in the same place has a tight box,
    whatever its median.

    LOG y, because the stock arm has a tail the method does not. M0315's
    worst stock design sits at 53 A against a 1.83 A median; on a linear
    axis that one design sets the scale and flattens every box to a line.

    Box is median and IQR, whiskers 5th-95th percentile. Over EVERY design,
    not only passing ones -- conditioning on passing truncates at the pass
    threshold and manufactures the very reduction being claimed, since the
    method passes more often and so keeps more of its own tail.
    """
    width = 0.34
    for j, arm in enumerate(arms):
        for i, site in enumerate(SITES):
            v = md[(md.site == site) & (md.arm == arm)].rmsd.to_numpy()
            if not len(v):
                continue
            x = i + (j - (len(arms) - 1) / 2) * width
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            lo, hi = np.percentile(v, [5, 95])
            c = ARMC[arm]
            ax.plot([x, x], [lo, hi], color=c, lw=0.9, zorder=3,
                    solid_capstyle="butt")
            ax.add_patch(plt.Rectangle(
                (x - width * 0.36, q1), width * 0.72, q3 - q1,
                facecolor=c, alpha=0.55, edgecolor=c, linewidth=0.8,
                zorder=4))
            ax.plot([x - width * 0.36, x + width * 0.36], [med, med],
                    color="white", lw=1.2, zorder=5, solid_capstyle="butt")

    # The benchmark's own criterion: 1.5 A is where AME calls a motif
    # recovered, so the line is the reason the spread matters.
    ax.axhline(1.5, color=PALETTE["red"], lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.text(len(SITES) - 0.45, 1.56, "1.5 A", fontsize=fs(6),
            color=PALETTE["red"], ha="right", va="bottom")

    ax.set_yscale("log")
    ax.set_ylim(0.3, 60)
    ax.set_yticks([0.5, 1, 2, 5, 10, 50])
    ax.set_yticklabels(["0.5", "1", "2", "5", "10", "50"])
    ax.set_xticks(range(len(SITES)))
    ax.set_xticklabels(SITES, fontsize=fs(6.5))
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(SITES) - 0.4)
    _tidy(ax, ylab="Motif RMSD to theozyme (A)")

    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="s", ms=5, mfc=ARMC[a],
                              mec=ARMC[a], ls="none", label=ARM_LABEL[a])
                       for a in arms],
              fontsize=fs(6.5), loc="upper left", frameon=False,
              handlelength=0.8, borderpad=0.0, labelspacing=0.22,
              handletextpad=0.4)


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
              ("stock ProteinMPNN", sp.sequence.tolist()),
              ("This work", pp.sequence.tolist())]

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
              "stock ProteinMPNN": INK["muted"],
              "This work": METHOD}
    for arm in ("single + double", "stock ProteinMPNN", "This work"):
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

    ONE condition: a policy trained at each k. The post-hoc arm -- the
    single k=4 policy re-cut at every k -- is a control about train/test
    mismatch, not part of the claim this panel makes, and it is kept in
    `figS_posthoc` rather than drawn here.

    The band is MIN-MAX over the three backbones, not a confidence interval.
    Every cell is one seed, against a documented 21-point run-to-run swing on
    pass rate, so the band is spread across backbones and says nothing about
    sampling error.
    """
    d = pd.read_csv(HERE / "data" / "ame_fragk.csv")
    c = METHOD
    g = (d[d.cond == "trained"].groupby("k").retention
         .agg(["mean", "min", "max"]).reset_index())
    ax.fill_between(g.k, 100 * g["min"], 100 * g["max"], color=c,
                    alpha=0.16, lw=0, zorder=2)
    ax.plot(g.k, 100 * g["mean"], "-o", color=c, lw=1.6, ms=3.6,
            mfc="white", mew=1.2, zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 36, 64])
    ax.set_xticklabels(["2", "4", "8", "16", "36", "64"])
    ax.set_ylim(0, 112)
    _tidy(ax, ylab="Recombinants passing (%)",
          xlab="Fragments per design, $k$")


# ================================================================== assembly

def main():
    arms = AA.collect()
    md = MV.design_means()
    space, var, pz = collect_space()

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 9.4))
    outer = fig.add_gridspec(
        4, 1, height_ratios=[0.74, 1.00, 1.02, 1.02],
        left=0.085, right=0.985, top=0.968, bottom=0.048, hspace=0.52)

    # (a) three landscapes, one budget spent three ways.
    panel_landscapes(fig, outer[0])

    # (b) the shuffling scheme, (c) the objective it feeds.
    mid = outer[1].subgridspec(1, 2, width_ratios=[1.70, 1.0], wspace=0.10)
    panel_shuffle(fig.add_subplot(mid[0, 0]))
    panel_objective(fig.add_subplot(mid[0, 1]))

    # (d) DNA cost, (e) the trade-off. (d) takes the wider cell: its four
    # price labels are spread along the 10^12 reference line, so its aspect
    # is load-bearing rather than cosmetic -- squeezed, they collide.
    r3 = outer[2].subgridspec(1, 3, width_ratios=[1.75, 1.0, 1.0],
                              wspace=0.46)
    panel_cost(fig.add_subplot(r3[0, 0]))
    panel_pass(fig.add_subplot(r3[0, 1]), arms)
    panel_diversity(fig.add_subplot(r3[0, 2]), arms)

    # (g) catalytic-atom spread, (h) fold cost, (i) sequence space,
    # (j) granularity.
    r4 = outer[3].subgridspec(1, 4, width_ratios=[1.0, 0.9, 1.05, 1.05],
                              wspace=0.58)
    panel_motif_var(fig.add_subplot(r4[0, 0]), md)
    panel_folds(fig.add_subplot(r4[0, 1]))
    panel_space(fig.add_subplot(r4[0, 2]), space, var, pz)
    panel_k(fig.add_subplot(r4[0, 3]))

    # Panel letters in FIGURE coordinates. ax.get_tightbbox under-measures
    # rotated text, which is how panel letters end up sitting on y-labels --
    # so these are placed explicitly against the gridspec rows rather than
    # measured off each axes.
    for letter, (x, y) in zip("abcdefghij", [
            (0.008, 0.980), (0.008, 0.805), (0.632, 0.805),
            (0.008, 0.545), (0.430, 0.545), (0.705, 0.545),
            (0.008, 0.272), (0.265, 0.272), (0.505, 0.272),
            (0.755, 0.272)]):
        fig.text(x, y, f"({letter})", fontsize=fs(9), fontweight="bold",
                 ha="left", va="top")

    save(fig, "fig1_method", dpi=900, tight=False, subdir="main")


if __name__ == "__main__":
    main()
