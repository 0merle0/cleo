#!/usr/bin/env python3
"""One panel per experimental campaign, for the narrative deck (campaigns.tex).

Every campaign in the project measures the same two quantities -- how much of a
library folds, and how much sequence space it covers -- and each puts a
different dial on the trade. This script draws each campaign once, on matched
axes where the axes are genuinely matched, so the narrative can be chosen rather
than assembled after the fact.

The numbers are constants here *on purpose*, with the command that produced each
block named above it. Three of the source trees no longer exist (`backbone19`,
`selection2`, pruned and not archived) and two of the campaigns are wet-lab
snapshots, so a script that recomputed everything would not run. What it must
not do is disagree with the paper: the klsweep block below is the output of
`experiments/ame/klsweep_table.py`, which is the single definition of those
cells, and the library block is `ame_library_table.py --draws 60`. Re-run those
two and paste, rather than editing a number in place.

    uv run python paper/figures/campaign_panels.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figio import save
from palette import PALETTE

DATA = Path(__file__).resolve().parent / "data"

C_SAMP, C_RL = PALETTE["gray"], PALETTE["blue"]
C_CHIM4, C_CHIM16 = PALETTE["orange"], PALETTE["green"]
C_WARN, C_INK = PALETTE["red"], "#374151"
# Temperature is a magnitude, so it gets one hue stepped light -> dark. The hue
# is deliberately not the policy's blue: a second blue would put the baseline in
# the same visual family as the thing it is being compared against.
T_RAMP = ["#E8B680", "#D9922F", "#B86A14", "#8E4A0B", "#5E3005"]

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9CA3AF", "axes.labelcolor": C_INK,
    "xtick.color": "#6B7280", "ytick.color": "#6B7280",
    "text.color": C_INK, "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# C1  experiments/ame/tempsweep, M0097_1ctt, LigandMPNN, 96 designs per T.
# T=1.0 has no U_15: at 4.2% pass too few designs survive to rarefy to 15.
TEMP = [(0.1, 58.3, 161, 40.6), (0.2, 66.7, 205, 47.8), (0.3, 58.3, 273, 57.9),
        (0.5, 34.4, 437, 80.5), (0.7, 26.0, 607, 100.5)]
TEMP_DEAD = (1.0, 4.2, None, 126.3)

# C2/C4  experiments/ame/klsweep_table.py -- the single definition of these cells.
KL = [  # kl_weight, pass%, U_15, U, mean Hamming, entropy, n_pass of 640
    (0.0,   83.12, 505.92, 2110, 76.62,  0.99, 532),
    (0.005, 82.19, 554.49, 2251, 87.51,  1.15, 526),
    (0.01,  57.50, 764.82, 2499, 109.19, 1.54, 368),
    (0.02,  56.72, 802.65, 2501, 114.41, 1.65, 363),
    (0.05,  40.47, 849.87, 2480, 121.85, 1.78, 259),
    (0.10,   9.84, 878.37, 1731, 124.12, 1.85,  63)]
LADDER = [("legacy objective", 78.0, 468.0), ("+ surrogate fix", 88.8, 570.5),
          ("+ log-prob fix", 83.1, 505.9), ("+ KL anchor", 56.7, 802.6)]

# C5/C6  paper/figures/ame_library_table.py --draws 60
# site, % of benchmark backbones with 0 passes, arm, library pass%, diversity
LIB = [
    ("M0664_2dhn", 16, "sampled", 90.6, 1753.0), ("M0664_2dhn", 16, "RL", 94.9, 1689.6),
    ("M0097_1ctt", 55, "sampled", 59.8, 836.0),  ("M0097_1ctt", 55, "RL", 72.7, 1259.5),
    ("M0097_1ctt", 55, "chim4", 85.5, 1129.5),   ("M0097_1ctt", 55, "chim16", 91.0, 984.4),
    ("M0255_1mg5", 77, "sampled", 25.8, 584.0),  ("M0255_1mg5", 77, "RL", 79.3, 875.0),
    ("M0255_1mg5", 77, "chim4", 97.3, 469.4),    ("M0255_1mg5", 77, "chim16", 87.1, 786.9),
    ("M0315_1ey3", 85, "sampled", 49.6, 1222.0), ("M0315_1ey3", 85, "RL", 78.1, 1283.6),
    ("M0315_1ey3", 85, "chim4", 97.3, 1087.5),   ("M0315_1ey3", 85, "chim16", 98.4, 1083.5),
    ("M0500_1e3i", 86, "sampled", 10.9, 399.0),  ("M0500_1e3i", 86, "RL", 53.5, 563.9),
    ("M0058_1cju", 95, "sampled", 12.9, 394.0),  ("M0058_1cju", 95, "RL", 44.9, 531.5),
    ("M0050_1dbt", 97, "sampled", 11.7, 229.6),  ("M0050_1dbt", 97, "RL", 5.5, 212.0),
    ("M0904_1qgx", 97, "sampled", 12.9, 353.0),  ("M0904_1qgx", 97, "RL", 47.3, 462.2),
    ("M0365_1pfk", 99, "sampled", 0.4, None),    ("M0365_1pfk", 99, "RL", 5.5, None)]

# C7  experiments/ame/{r1rep,fragb32}, T2 library eval, 256 parents + 256 recombinants
T2 = [("M0097", (68.0, 71.1), 96.9, 25.8), ("M0255", (29.3, 30.5), 82.4, 52.0),
      ("M0315", (65.2, 57.0), 100.0, 34.8)]

# C8  experiments/ame/{fragbsweep,fragb16,fragb32}/offline_probe, 38 probe points
GAP = [(8, 5, 0.085, 0.312), (16, 15, -0.001, 0.151),
       (32, 14, -0.005, 0.111), (64, 4, 0.054, 0.133)]
ORACLE = [("pool (fragb32)", -0.005, 14, True),
          ("RF3 only (fragb32rf3)", 0.041, 10, True),
          ("RF3 only (fragrl4)", 0.279, 15, False)]

# C9  adaptyv_library, de novo PETase, 3,135 wells assayed
RHO = [("acyl his_catN pLDDT", 0.5996), ("acyl activesite pLDDT", 0.5865),
       ("acyl his pLDDT", 0.5727), ("acyl ser pLDDT", 0.5422),
       ("ptm (global)", 0.3478), ("iptm (global)", 0.3089),
       ("mutation count", -0.65)]
DEPTH = [("0\u20139", 218.0), ("20\u201329", 6.9), ("50+", 0.2)]

ARMS = ["sampled", "RL", "chim4", "chim16"]
ARMC = {"sampled": C_SAMP, "RL": C_RL, "chim4": C_CHIM4, "chim16": C_CHIM16}
ARMN = {"sampled": "sampled $T{=}1.0$", "RL": "RL policy",
        "chim4": "RL+chimera $k{=}4$", "chim16": "RL+chimera $k{=}16$"}


def grid(ax, axis="y"):
    ax.grid(axis=axis, color="#E5E7EB", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def c1():
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    grid(ax, "both")
    for i, (T, p, u, _) in enumerate(TEMP):
        ax.scatter([u], [p], s=70, color=T_RAMP[i], ec="white", lw=1.2, zorder=3)
        ax.annotate(f"$T{{=}}{T}$", (u, p), xytext=(8, -3), textcoords="offset points",
                    fontsize=8, color=C_INK)
    ax.annotate("best folding point", (607, 26.0), xytext=(-10, 16),
                textcoords="offset points", ha="right", fontsize=8, color="#6B7280")
    ax.set_xlim(120, 700); ax.set_ylim(0, 75)
    ax.set_xlabel("$U_{15}$ (distinct substitutions, rarefied to 15 passing designs)")
    ax.set_ylabel("oracle pass rate (%)")
    save(fig, "panel_c1"); plt.close(fig)


def c2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    grid(a, "both")
    w, p, u15, u = (np.array([r[i] for r in KL]) for i in (0, 1, 2, 3))
    a.plot(u15, p, color="#D1D5DB", lw=1.2, ls="--", zorder=1)
    a.scatter(u15, p, s=55, color=C_RL, ec="white", lw=1.2, zorder=3)
    for wi, ui, pi in zip(w, u15, p):
        off = (-8, 8) if wi == 0.10 else (8, -3)
        a.annotate(f"$w{{=}}{wi:g}$", (ui, pi), xytext=off, textcoords="offset points",
                   ha="right" if wi == 0.10 else "left", fontsize=7.5, color=C_INK)
    a.set_xlim(460, 950); a.set_ylim(0, 95)
    a.set_xlabel("$U_{15}$ (rarefied coverage)")
    a.set_ylabel("oracle pass rate (%)")
    a.set_title("the KL anchor trades pass rate for coverage", loc="left")

    grid(b)
    x = np.arange(len(KL))
    b.plot(x, u, "-o", ms=4, color=C_CHIM4, lw=1.6, label="$U$ (raw)")
    b.plot(x, u15, "-o", ms=4, color=C_RL, lw=1.6, label="$U_{15}$ (rarefied)")
    b.set_xticks(x); b.set_xticklabels([f"{v:g}" for v in w])
    b.set_xlabel("kl_weight"); b.set_ylabel("distinct substitutions")
    b.set_ylim(0, 2750); b.legend(frameon=False, loc="center left")
    b.set_title("raw $U$ collapses where $U_{15}$ still climbs", loc="left")
    fig.tight_layout()
    save(fig, "panel_c2"); plt.close(fig)


def c3():
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    grid(ax, "both")
    ax.axhspan(26.0, 95, xmin=0, xmax=1, color="none")
    ax.add_patch(plt.Rectangle((607, 26.0), 950 - 607, 95 - 26.0,
                               color=C_RL, alpha=0.07, zorder=1, lw=0))
    for i, (T, p, u, _) in enumerate(TEMP):
        ax.scatter([u], [p], s=65, color=T_RAMP[i], ec="white", lw=1.2, zorder=3)
    for _, p, u, *_ in KL:
        ax.scatter([u], [p], s=65, color=C_RL, ec="white", lw=1.2, zorder=3)
    ax.annotate("LigandMPNN, $T{=}0.1$\u2013$0.7$", (273, 58.3), xytext=(0, 14),
                textcoords="offset points", fontsize=8, color="#8E4A0B")
    ax.annotate("this paper, $w{=}0.02$", (802.65, 56.72), xytext=(-12, 14),
                textcoords="offset points", ha="right", fontsize=8, color=C_RL)
    ax.annotate("dominates the baseline's\nbest folding point", (945, 88),
                ha="right", va="top", fontsize=8, color="#6B7280")
    ax.set_xlim(120, 950); ax.set_ylim(0, 95)
    ax.set_xlabel("$U_{15}$ (rarefied coverage)")
    ax.set_ylabel("oracle pass rate (%)")
    save(fig, "panel_c3"); plt.close(fig)


def c4():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    y = np.arange(len(LADDER))[::-1]
    names = [n.replace("+ ", "$+$ ") for n, _, _ in LADDER]
    for ax, vals, col, lab, xmax, fmt in (
            (a, [v for _, v, _ in LADDER], C_RL, "oracle pass rate (%)", 104, "{:.1f}"),
            (b, [v for _, _, v in LADDER], C_CHIM4, "$U_{15}$", 950, "{:.0f}")):
        grid(ax, "x")
        ax.barh(y, vals, height=0.6, color=col, zorder=2)
        for yi, v in zip(y, vals):
            ax.annotate(fmt.format(v), (v, yi), xytext=(5, -3),
                        textcoords="offset points", fontsize=8, color=C_INK)
        ax.set_xlim(0, xmax); ax.set_xlabel(lab)
    a.set_yticks(y); a.set_yticklabels(names)
    fig.tight_layout()
    save(fig, "panel_c4"); plt.close(fig)


def c5():
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    grid(ax)
    by = {}
    for site, dead, arm, p, _ in LIB:
        if arm in ("sampled", "RL"):
            by.setdefault(site, {"dead": dead})[arm] = p
    sites = sorted((s for s in by if {"sampled", "RL"} <= by[s].keys()),
                   key=lambda s: by[s]["dead"])
    used = []
    for s in sites:
        d = by[s]
        up = d["RL"] > d["sampled"]
        col = C_RL if up else C_WARN
        ax.plot([0, 1], [d["sampled"], d["RL"]], color=col, lw=1.4, zorder=2)
        ax.scatter([0], [d["sampled"]], s=38, color=C_SAMP, ec="white", lw=1, zorder=3)
        ax.scatter([1], [d["RL"]], s=38, color=col, ec="white", lw=1, zorder=3)
        ypos = d["RL"]                 # dodge labels that would print on top of
        while any(abs(ypos - v) < 3.4 for v in used):   # each other (two arms
            ypos += 3.4                                  # both land on 5.5)
        used.append(ypos)
        ax.annotate(f"{s}  {d['dead']}% dead", (1, ypos), xytext=(8, -3),
                    textcoords="offset points", fontsize=7.5, color=col)
    for x, m in ((0, 30.5), (1, 53.5)):
        ax.plot([x - 0.06, x + 0.06], [m, m], color="#6B7280", lw=1.4, ls=(0, (2, 2)))
        ax.annotate(f"mean {m}", (x, m), xytext=(-9, -3), textcoords="offset points",
                    ha="right", fontsize=7.5, color="#6B7280",
                    bbox=dict(fc="white", ec="none", pad=1.2))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["sampled $T{=}1.0$", "RL policy"])
    ax.set_xlim(-0.18, 1.62); ax.set_ylim(0, 100)
    ax.set_ylabel("recombined library pass rate (%)")
    save(fig, "panel_c5"); plt.close(fig)


def c6():
    """Faceted by backbone. One panel per backbone rather than three
    trajectories on one axis: the arms cross each other in diversity, so a
    single panel draws four long segments over one another and the shape of any
    one backbone becomes unreadable. Axes are shared, so the panels stay
    comparable."""
    sites = ["M0097_1ctt", "M0255_1mg5", "M0315_1ey3"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), sharex=True, sharey=True)
    for ax, s in zip(axes, sites):
        grid(ax, "both")
        pts = [(a, p, d) for site, _, a, p, d in LIB if site == s]
        pts = [next(q for q in pts if q[0] == a) for a in ARMS
               if any(q[0] == a for q in pts)]
        ax.plot([q[2] for q in pts], [q[1] for q in pts], color="#D1D5DB",
                lw=1.2, zorder=1)
        for a, p, d in pts:
            ax.scatter([d], [p], s=80, color=ARMC[a], ec="white", lw=1.3, zorder=3)
        ax.set_title(s, loc="left", fontsize=8.5, color="#6B7280")
        ax.set_xlim(380, 1420); ax.set_ylim(0, 108)
        ax.set_xticks([500, 900, 1300])
    axes[0].set_ylabel("recombined library pass rate (%)")
    fig.supxlabel("library diversity (distinct substitutions over passing members, "
                  "rarefied)", fontsize=8.5, color=C_INK, y=0.10)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=7, mec="white",
                               color=ARMC[a], label=ARMN[a]) for a in ARMS],
               frameon=False, loc="lower center", ncol=4, fontsize=8,
               handletextpad=0.3, columnspacing=1.4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, "panel_c6"); plt.close(fig)


def c7():
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    grid(ax)
    for i, (site, seeds, chim, delta) in enumerate(T2):
        lo, hi = min(seeds), max(seeds)
        ax.bar(i, hi - lo, bottom=lo, width=0.3, color=C_SAMP, zorder=2)
        ax.annotate("2 seeds", (i - 0.17, (lo + hi) / 2), xytext=(-5, -3),
                    textcoords="offset points", ha="right", fontsize=7.5, color="#6B7280")
        ax.scatter([i], [chim], s=95, color=C_CHIM4, ec="white", lw=1.3, zorder=4)
        ax.annotate(f"$+{delta}$ pt", (i, chim), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=8, color=C_CHIM4)
    ax.set_xticks(range(len(T2))); ax.set_xticklabels([s for s, _, _, _ in T2])
    ax.set_ylim(0, 115); ax.set_ylabel("recombinant pass rate (%)")
    ax.legend(handles=[Patch(color=C_SAMP, label="plain RL, two seeds"),
                       Line2D([], [], marker="o", ls="", ms=7, mec="white",
                              color=C_CHIM4, label="chimera objective, oracle pool")],
              frameon=False, loc="lower center", ncol=2, handletextpad=0.4)
    save(fig, "panel_c7"); plt.close(fig)


def c8():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    grid(a)
    for i, (batch, n, mean, worst) in enumerate(GAP):
        a.bar(i, mean, width=0.5, color=C_CHIM4 if mean > 0 else C_RL, zorder=2)
        a.plot([i, i], [mean, worst], color="#6B7280", lw=1.2, zorder=3)
        a.plot([i - 0.12, i + 0.12], [worst, worst], color="#6B7280", lw=1.2, zorder=3)
        a.annotate("worst", (i, worst), xytext=(7, -3), textcoords="offset points",
                   fontsize=7.5, color="#6B7280")
        a.annotate(f"$n{{=}}{n}$", (i, mean), xytext=(0, 9 if mean > 0 else -14),
                   textcoords="offset points", ha="center", fontsize=7.5, color="#6B7280")
    a.axhline(0, color="#6B7280", lw=1.1)
    a.set_xticks(range(len(GAP))); a.set_xticklabels([str(g[0]) for g in GAP])
    a.set_xlabel("rollout batch size")
    a.set_ylabel("in-loop $-$ held-out AF3")
    a.set_ylim(-0.06, 0.37)
    a.set_title("the in-loop number overstates", loc="left")

    grid(b, "x")
    y = np.arange(len(ORACLE))[::-1]
    for yi, (name, gap, n, ok) in zip(y, ORACLE):
        b.barh(yi, gap, height=0.5, zorder=2,
               color=(C_CHIM4 if gap > 0 else C_RL) if ok else "white",
               ec=C_WARN if not ok else "none", hatch="///" if not ok else None, lw=1.0)
        b.annotate(f"${gap:+.3f}$", (gap, yi), xytext=(6, -3),
                   textcoords="offset points", fontsize=8, color=C_INK)
    b.axvline(0, color="#6B7280", lw=1.1)
    b.set_yticks(y)
    b.set_yticklabels([n for n, _, _, _ in ORACLE], fontsize=7.5)
    b.set_xlim(-0.05, 0.36); b.set_xlabel("in-loop $-$ held-out AF3")
    b.set_title("hatched arm also varies initialisation", loc="left")
    fig.tight_layout()
    save(fig, "panel_c8"); plt.close(fig)


def c9():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                               gridspec_kw={"width_ratios": [1.6, 1]})
    grid(a, "x")
    y = np.arange(len(RHO))[::-1]
    for yi, (name, rho) in zip(y, RHO):
        dist = name == "mutation count"
        col = C_WARN if dist else (C_SAMP if "global" in name else C_RL)
        a.barh(yi, rho, height=0.58, color=col, zorder=2)
        a.annotate(f"${rho:+.2f}$", (rho, yi),
                   xytext=(6 if rho > 0 else -6, -3), textcoords="offset points",
                   ha="left" if rho > 0 else "right", fontsize=8, color=C_INK)
    a.axvline(0, color="#6B7280", lw=1.1)
    a.set_yticks(y)
    a.set_yticklabels([n for n, _ in RHO], fontsize=7.5)
    a.set_xlim(-0.95, 0.82)
    a.set_xlabel("Spearman $\\rho$ against measured activity")
    a.set_title("zero-shot predictors, PETase", loc="left")

    grid(b)
    for i, (lab, med) in enumerate(DEPTH):
        b.bar(i, med, width=0.55, color=C_CHIM4, zorder=2)
        b.annotate(f"{med:g}", (i, med), xytext=(0, 4), textcoords="offset points",
                   ha="center", fontsize=8, color=C_INK)
    b.set_yscale("log"); b.set_ylim(0.08, 900)
    b.set_xticks(range(len(DEPTH))); b.set_xticklabels([d[0] for d in DEPTH])
    b.set_xlabel("mutations from the parent")
    b.set_ylabel("median activity")
    b.set_title("activity falls with depth", loc="left")
    fig.tight_layout()
    save(fig, "panel_c9"); plt.close(fig)


def c10():
    def load(name):
        with open(DATA / f"petase_super_{name}_transfer.csv") as fh:
            return [(r["concept"], float(r["spearman_petase"]),
                     float(r["spearman_super"])) for r in csv.DictReader(fh)]
    af3, ros = load("af3"), load("rosetta")
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    grid(ax, "both")
    ax.axhline(0, color="#D1D5DB", lw=1.0)
    ax.axvline(0, color="#D1D5DB", lw=1.0)
    ax.plot([-0.3, 0.3], [-0.3, 0.3], color="#9CA3AF", lw=1.1, ls=(0, (4, 3)),
            label="$y = x$, perfect transfer")
    ax.scatter([p for _, p, _ in ros], [s for _, _, s in ros], s=26, color=C_CHIM4,
               alpha=0.8, ec="white", lw=0.6, label="Rosetta concepts (66)", zorder=3)
    ax.scatter([p for _, p, _ in af3], [s for _, _, s in af3], s=34, color=C_RL,
               alpha=0.85, ec="white", lw=0.6, label="AF3 concepts (49)", zorder=4)
    ax.annotate("acyl_his_catN_plddt", (0.5996, -0.1475),
                xytext=(-8, -12), textcoords="offset points", ha="right",
                fontsize=7.5, color=C_WARN)
    ax.annotate("His\u2013Asp distance", (-0.5016, -0.0716), xytext=(10, 8),
                textcoords="offset points", fontsize=7.5, color=C_CHIM4)
    ax.set_xlim(-0.62, 0.72); ax.set_ylim(-0.26, 0.26)
    ax.set_xlabel("Spearman $\\rho$ on PETase (3,135 assayed)")
    ax.set_ylabel("$\\rho$ on super (2,509 purified)")
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    save(fig, "panel_c10"); plt.close(fig)




# ---------------------------------------------------------------------------
# Cost. Not a campaign -- no data, a design calculation -- but it is the
# question every one of the campaigns above is eventually answerable to.
#
# The model, stated so it can be argued with:
#
#   A protein of L amino acids is 3L bp. Split it into k equal fragments and
#   each fragment is 3L/k bp. With a budget B and synthesis priced at c per
#   base, you can buy B/c bases, which is B/(3Lc) full-length-equivalents of
#   DNA. Spread over k slots, the number of variants you can stock per slot is
#
#       v = B / (3 L c)          <- note: no k in it
#
#   because cutting into k pieces buys k times as many fragments, each 1/k as
#   long. The two cancel exactly. Recombining across slots then gives
#
#       S = v^k = (B / (3 L c))^k
#
#   So in the idealised model k is a pure exponent on a number set entirely by
#   budget, and doubling the budget multiplies library size by 2^k.
#
# Two corrections that are not optional in practice:
#
#   adapters  Each fragment needs its own Golden Gate sites and overhangs,
#             h bp that are paid for and are not payload. Cost per fragment is
#             c(3L/k + h), so the fragment count no longer scales as k.
#   minimum   Synthesis is sold with a minimum billable length m. Below it you
#             pay for m and receive 3L/k. On a 200-aa
#             protein that floor binds from k = 3 onward, and by k = 8 most of
#             every base bought is padding.
#
# Both are vendor terms rather than physics, so they are parameters here.
# COST_BP is Twist eBlock pricing, confirmed. MIN_BP and ADAPTER_BP are
# order-of-magnitude placeholders and are the two numbers to check against a
# live quote before anyone plans a spend from this figure.
AA = 200             # enzyme length in amino acids
COST_BP = 0.07       # $/base, Twist eBlocks
ADAPTER_BP = 40      # Golden Gate sites + overhangs per fragment (assumed)
MIN_BP = 300         # minimum billable fragment length (assumed)
PARENTS = 256          # parents per cell in the T2 library evaluations (C6, C7)
# The PETase library as actually ordered: 5 slots, 25-26 variants each = 128
# distinct protein parts spanning 1.1e7 recombinants, but 1,920 DNA parts were
# ordered to realise them. The gap is what redundancy costs.
PETASE = dict(aa=209, k=5, per_slot=25.6, size=1.1e7, protein_parts=128,
              dna_parts=1920)


def variants_naive(budget, aa=AA, c=COST_BP):
    return budget / (3 * aa * c)


def billable_bp(k, aa=AA, h=ADAPTER_BP, m=MIN_BP):
    """-> bp charged for one fragment, after adapters and the order minimum."""
    return max(m, 3 * aa / k + h)


def variants_real(budget, k, aa=AA, c=COST_BP, h=ADAPTER_BP, m=MIN_BP):
    return budget / (c * k * billable_bp(k, aa, h, m))


def cost_size():
    """How library size scales with cost -- the scaling law itself.

    Taking logs of S = (B/B1)^k, where B1 = 3Lc is the cost of one full-length
    gene:

        log S = k (log B - log B1)

    which on log-log axes is a straight line of slope k through the single
    point (B1, 1). So the whole model is a fan: every k is a line, k is the
    slope, and they all pivot on the cost of one gene. k = 1 is the line you
    get without recombination at all -- buy v genes, have v sequences -- and
    the distance from it to any other line is what cutting and reassembling
    buys for the same money.

    The right panel inverts it to the question anyone actually asks: what does
    a target library size cost, and how much of that is the vendor's order
    minimum rather than the model.
    """
    B1 = 3 * AA * COST_BP                       # one full-length gene
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.6))

    grid(a, "both")
    TOP, XMAX = 1e18, 1e6
    B = np.logspace(np.log10(B1), np.log10(XMAX), 400)
    ks = [1, 2, 3, 4, 6, 8, 12, 16]
    cols = plt.cm.YlGnBu(np.linspace(0.32, 0.95, len(ks)))
    for k, col in zip(ks, cols):
        y = (B / B1) ** k
        a.plot(B, y, color=col, lw=1.7, zorder=3)
        if y[-1] > TOP:                          # label where it leaves the box
            xb = B[np.argmax(y > TOP)]
            a.annotate(f"{k}", (xb, TOP), xytext=(1, -10),
                       textcoords="offset points", fontsize=8, color=col,
                       ha="center")
        else:
            a.annotate(f"$k{{=}}{k}$", (B[-1], y[-1]), xytext=(4, -3),
                       textcoords="offset points", fontsize=8, color=col)
    a.annotate("$k =$", (2.2e2, TOP), xytext=(0, -10), ha="right",
               textcoords="offset points", fontsize=8, color="#6B7280")
    a.scatter([B1], [1], s=42, color=C_WARN, ec="white", lw=1.2, zorder=5)
    a.annotate(f"one gene, \\${B1:.0f}: every line pivots here",
               (B1, 1), xytext=(9, -11), textcoords="offset points",
               fontsize=7.5, color=C_WARN)
    a.annotate("$S = (B/B_1)^{k}$\n$\\log S = k\\,(\\log B - \\log B_1)$",
               (0.97, 0.11), xycoords="axes fraction", ha="right",
               fontsize=9, color=C_INK)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(B1 * 0.5, XMAX); a.set_ylim(0.25, TOP)
    a.set_xlabel("synthesis budget (\\$)")
    a.set_ylabel("library size (recombinants)")
    a.set_title("slope is $k$; $k = 1$ is no recombination", loc="left",
                fontsize=8.5)

    grid(b, "both")
    kk = np.arange(2, 17)
    targets = [(1e4, "$10^{4}$"), (1e9, "$10^{9}$"), (1e12, "$10^{12}$")]
    for (S, lab), col in zip(targets, ["#9FB6D1", C_RL, "#0F2440"]):
        ideal = [B1 * S ** (1 / k) for k in kk]
        real = [COST_BP * k * billable_bp(k) * S ** (1 / k) for k in kk]
        b.plot(kk, ideal, color=col, lw=1.2, ls=(0, (4, 3)), zorder=2)
        b.plot(kk, real, color=col, lw=2.0, zorder=3)
        b.annotate(lab, (kk[0], real[0]), xytext=(6, 2),
                   textcoords="offset points", fontsize=8, color=col)
    b.set_yscale("log"); b.set_xlim(1.4, 16.8); b.set_ylim(50, 3e8)
    b.set_xticks([2, 4, 6, 8, 10, 12, 14, 16])
    b.set_xlabel("$k$, fragments per construct")
    b.set_ylabel("budget to reach that library size (\\$)")
    b.set_title("dashed: model. solid: after the order minimum", loc="left",
                fontsize=8.5)
    fig.tight_layout()
    save(fig, "panel_cost_size"); plt.close(fig)


def cost_k():
    """What the order minimum does to the idealised answer."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1),
                               gridspec_kw={"width_ratios": [1.15, 1]})
    grid(a, "both")
    ks = np.arange(2, 25)
    for B, col in ((2000, "#9FB6D1"), (10000, C_RL), (50000, "#0F2440")):
        ideal = [variants_naive(B) ** k for k in ks]
        real = [max(variants_real(B, k), 1e-9) ** k for k in ks]
        a.plot(ks, ideal, color=col, lw=1.3, ls=(0, (4, 3)), zorder=2)
        a.plot(ks, real, color=col, lw=2.0, zorder=3)
        a.annotate(f"\\${B:,}", (ks[-1], min(real[-1], 6e24)), xytext=(5, -3),
                   textcoords="offset points", fontsize=8, color=col)
    a.set_yscale("log"); a.set_ylim(1e2, 1e26); a.set_xlim(1.5, 27)
    a.set_xlabel("$k$, fragments per construct")
    a.set_ylabel("library size (recombinants)")
    a.set_title("dashed: idealised. solid: after adapters and the order minimum",
                loc="left", fontsize=8)

    grid(b)
    kk = [2, 3, 4, 6, 8, 12, 16]
    payload = np.array([3 * AA / k for k in kk])
    adapter = np.full(len(kk), float(ADAPTER_BP))
    padding = np.array([max(0.0, MIN_BP - 3 * AA / k - ADAPTER_BP) for k in kk])
    b.bar(range(len(kk)), payload, 0.62, color=C_RL, label="payload", zorder=2)
    b.bar(range(len(kk)), adapter, 0.62, bottom=payload, color=C_SAMP,
          label="adapters", zorder=2)
    b.bar(range(len(kk)), padding, 0.62, bottom=payload + adapter, color="white",
          ec=C_WARN, hatch="///", lw=0.9, label="paid, not received", zorder=2)
    b.axhline(MIN_BP, color=C_WARN, lw=1.0, ls=(0, (3, 3)), zorder=3)
    b.annotate(f"{MIN_BP} bp order minimum", (len(kk) - 0.5, MIN_BP),
               xytext=(0, 4), textcoords="offset points", ha="right",
               fontsize=7.5, color=C_WARN)
    b.set_xticks(range(len(kk))); b.set_xticklabels([str(k) for k in kk])
    b.set_xlabel("$k$, fragments per construct")
    b.set_ylabel("bp billed per fragment")
    b.set_ylim(0, 380)
    b.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=1)
    b.set_title("what a base buys", loc="left", fontsize=8)
    fig.tight_layout()
    save(fig, "panel_cost_k"); plt.close(fig)


if __name__ == "__main__":
    plt.rcParams["text.usetex"] = False
    for fn in (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10,
               cost_size, cost_k):
        fn()
