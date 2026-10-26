#!/usr/bin/env python3
"""Does training at k rescue recombination at k?

`ame_granularity.py` sweeps k from 2 to 36 and shows retention falling as
fragments shorten. Every point in it past k=4 comes from a policy TRAINED at
k=4, so the decline has two candidate causes it cannot separate:

  train/test mismatch   the policy never saw an 11-residue fragment boundary,
                        and a policy trained at that k would hold retention;
  physical              11 residues cannot carry their own junction context,
                        so the decline survives training at that k.

E18 trains at k in {2, 8, 16} on all three backbones and recombines each policy
at its OWN k, which separates them. This script puts the two conditions on one
axis. The answer is the gap between the curves:

  gap closes   -> the earlier decline was mismatch, and the method scales
                  further than `ame_granularity.py` implies;
  gap persists -> the decline is physical, and that is the method's boundary.

Three things to know about what is being compared.

**Retention is the recombinant pass rate, undivided.** Recombinants are built
only from parents that already passed, so the denominator of
recombinant-over-parent is 1 by construction and retention is the recombinant
pass rate itself. This is the definition `make_recombinants.py` states and
`ame_granularity.py` plots, and it is kept here so the two figures measure the
same thing. An earlier draft of this script divided by the pass rate of the
whole unselected 256-design sample, which is a different quantity and gave
retentions above 100%. That sample rate is still reported, as `sample_pass`,
because it says how good the policy's raw output is -- but it is context, not
a denominator.

**Exact parents are excluded from the libraries.** `make_recombinants.py` drops
them and all duplicates before sampling, so these numbers do not carry the 25%
unrecombined-parent inflation that the in-loop `build_chimeras` chimera pass
rate does. An in-loop number and a number here are not comparable.

**The k=4 cell is the ragged one.** Its `_last.pt` checkpoints sit at steps 125
(M0097), 299 (M0255) and 150 (M0315), against 199 for eight of the nine fragk
arms and 150 for M0315/k=16. Step 125 is the only checkpoint present in all
twelve runs; sampling there would make the sweep uniform at the price of
rerunning cells already folded. So k=4 is drawn hollow, as an anchor whose
training length is not matched.

    uv run python paper/figures/ame_fragk.py [--draws 200]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLEO = HERE.parents[1]
AME = CLEO / "experiments" / "ame"
sys.path.insert(0, str(HERE))
from figio import save  # noqa: E402
from palette import PALETTE  # noqa: E402
sys.path.insert(0, str(AME))
from analyze_selection2 import as_matrix  # noqa: E402

BB = {"M0097": "run_M0097_1ctt_cond9_14",
      "M0255": "run_M0255_1mg5_cond17_66",
      "M0315": "run_M0315_1ey3_cond18_25"}
# k=4 anchor lives in a different family per backbone; first hit wins.
ANCHOR = {"M0097": ("fragbsweep", "bs16"),
          "M0255": ("fragb16", "bs16"),
          "M0315": ("fragb16", "bs16")}
TRAINED_K = (2, 8, 16)
C = {"trained": PALETTE["blue"], "posthoc": PALETTE["gray"]}
MARK = {"M0097": "o", "M0255": "s", "M0315": "^"}


def pass_rate(path):
    """-> (passing deduplicated sequences, pass fraction) for one scored CSV."""
    d = pd.read_csv(path)
    keep = d[d.rfd2_any_pass].drop_duplicates("sequence")
    return keep.sequence.tolist(), float(d.rfd2_any_pass.mean())


def subs(seqs, ref):
    """-> distinct (position, residue) substitutions away from `ref`."""
    M = as_matrix(list(seqs))
    return len({(j, M[i, j]) for i in range(M.shape[0])
                for j in range(M.shape[1]) if M[i, j] != ref[j]})


def cell(site, k):
    """-> (eval parents csv, recomb chimeras csv) for one trained-at-k cell."""
    if k == 4:
        fam, suf = ANCHOR[site]
    else:
        fam, suf = "fragk", f"fragk{k}"
    stem = f"{BB[site]}_{suf}"
    return (AME / fam / "eval" / stem / f"{stem}_af3scored.csv",
            AME / fam / "recomb" / f"{stem}_scored.csv")


def collect_trained():
    """-> rows for policies recombined at the k they were trained at."""
    rows = []
    for site in BB:
        for k in (4,) + TRAINED_K:
            ev, rc = cell(site, k)
            if not (ev.exists() and rc.exists()):
                continue
            par_seqs, par = pass_rate(ev)
            chi_seqs, chi = pass_rate(rc)
            rows.append(dict(site=site, k=k, cond="trained",
                             sample_pass=par, retention=chi,
                             n_pass=len(chi_seqs), seqs=chi_seqs))
    return rows


def collect_posthoc():
    """-> rows for the k=4-trained policy recombined at every k."""
    rows = []
    for f in sorted(AME.glob("slotsweep/*_policy_k*_scored.csv")):
        if "_samples" in f.name:
            continue
        m = re.match(r"(M\d+)_policy_k(\d+)_scored\.csv", f.name)
        if not m or m.group(1) not in BB:
            continue
        site, k = m.group(1), int(m.group(2))
        ev = sorted((AME / "r1panel" / "eval").glob(f"run_{site}*/*_af3scored.csv"))
        if not ev:
            continue
        _, par = pass_rate(ev[0])
        chi_seqs, chi = pass_rate(f)
        rows.append(dict(site=site, k=k, cond="posthoc",
                         sample_pass=par, retention=chi,
                         n_pass=len(chi_seqs), seqs=chi_seqs))
    return rows


def add_diversity(d, draws, rng):
    """Rarefied distinct substitutions, matched within (site, cond).

    Matched to the smallest passing count across the k levels of that series,
    so a level cannot win on diversity merely by having more of its library
    fold -- the same rule `ame_library_table.py` applies across arms.
    """
    d["diversity"] = np.nan
    for (site, cond), g in d.groupby(["site", "cond"]):
        pools = {r.k: r.seqs for r in g.itertuples() if len(r.seqs) >= 2}
        if len(pools) < 2:
            continue
        n = min(len(v) for v in pools.values())
        if n < 2:
            continue
        for k, pool in pools.items():
            ref = np.array([max(set(c), key=list(c).count)
                            for c in as_matrix(pool).T], dtype="S1")
            val = float(np.mean([subs(list(rng.choice(pool, n, replace=False)), ref)
                                 for _ in range(draws)]))
            d.loc[(d.site == site) & (d.cond == cond) & (d.k == k),
                  "diversity"] = val
        d.loc[(d.site == site) & (d.cond == cond), "rarefied_to"] = n
    return d


def panel_retention(d, fig, gs):
    """Retention against k, one facet per backbone, both conditions."""
    axes = []
    for i, site in enumerate(sorted(BB)):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2],
                             sharey=axes[0] if axes else None)
        axes.append(ax)
        for cond, lbl in (("posthoc", "trained at $k=4$, cut at $k$"),
                          ("trained", "trained at $k$")):
            g = d[(d.site == site) & (d.cond == cond)].sort_values("k")
            if not len(g):
                continue
            ax.plot(g.k, 100 * g.retention, "-", color=C[cond], lw=2, zorder=2,
                    label=lbl if i == 0 else None)
            # k=4 is the anchor whose training length is not matched: hollow.
            for r in g.itertuples():
                hollow = (cond == "trained" and r.k == 4)
                ax.plot(r.k, 100 * r.retention, MARK[site], ms=8, zorder=3,
                        mfc="white" if hollow else C[cond],
                        mec=C[cond], mew=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 36])
        ax.set_xticklabels(["2", "4", "8", "16", "36"])
        ax.set_xlabel("fragments per design, $k$")
        ax.set_title(site, fontsize=10)
        ax.axhline(50, color=PALETTE["red"], lw=1, ls=":", zorder=1)
        grid(ax)
        if i == 0:
            ax.set_ylabel("retention (% of recombinants passing)")
            ax.legend(frameon=False, fontsize=8, loc="lower left")
        else:
            ax.tick_params(labelleft=False)
    return axes


def panel_rates(d, ax):
    """Recombinant rate against the unselected sample rate it came from.

    Not a ratio -- the two are plotted side by side because the sample rate is
    a property of the policy and the recombinant rate is a property of its
    fragments, and k moves only the second.
    """
    for site in sorted(BB):
        g = d[(d.site == site) & (d.cond == "trained")].sort_values("k")
        if not len(g):
            continue
        ax.plot(g.k, 100 * g.retention, "-", color=C["trained"], lw=2,
                marker=MARK[site], ms=7, mec=C["trained"], mfc=C["trained"],
                label=None)
        ax.plot(g.k, 100 * g.sample_pass, "--", color=PALETTE["gray"], lw=1.5,
                marker=MARK[site], ms=6, mec=PALETTE["gray"], mfc="white",
                label=None)
    # Identity is the marker, the measured quantity is the colour+dash; a
    # legend keyed on one backbone's two lines would claim the other two are
    # something else.
    h = [Line2D([], [], color=C["trained"], lw=2, label="recombinants"),
         Line2D([], [], color=PALETTE["gray"], lw=1.5, ls="--",
                label="unselected sample")]
    h += [Line2D([], [], color=PALETTE["gray"], lw=0, marker=MARK[s_], ms=6,
                 mfc="white", mec=PALETTE["gray"], label=s_)
          for s_ in sorted(BB)]
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16])
    ax.set_xticklabels(["2", "4", "8", "16"])
    ax.set_xlabel("fragments per design, $k$")
    ax.set_ylabel("pass rate (%)")
    ax.set_title("recombinants vs the sample they came from", fontsize=10)
    ax.legend(handles=h, frameon=False, fontsize=7, loc="lower left", ncol=2)
    grid(ax)


def panel_trade(d, ax):
    """Library pass against library diversity, k labelled along each path."""
    for site in sorted(BB):
        g = d[(d.site == site) & (d.cond == "trained")
              ].dropna(subset=["diversity"]).sort_values("k")
        if not len(g):
            continue
        ax.plot(g.diversity, 100 * g.retention, "-", color=C["trained"],
                lw=1.5, alpha=0.6, zorder=2)
        ax.plot(g.diversity, 100 * g.retention, MARK[site], ms=8,
                color=C["trained"], mec="white", mew=1, zorder=3,
                label=site)
        for r in g.itertuples():
            ax.annotate(f"$k$={r.k}", (r.diversity, 100 * r.retention),
                        textcoords="offset points", xytext=(6, 5),
                        fontsize=7, color=PALETTE["gray"])
    ax.margins(x=0.12)
    ax.set_xlabel("distinct substitutions in passing library (rarefied)")
    ax.set_ylabel("library pass rate (%)")
    ax.set_title("up and to the right is better", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    grid(ax, axis="both")


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color="#E5E7EB", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def report(d):
    """Print every number the figure draws, so prose cannot drift from it."""
    t = d.drop(columns=["seqs"]).copy()
    for c in ("sample_pass", "retention"):
        t[c] = 100 * t[c]
    pd.set_option("display.width", 200)
    print("\nFRAGMENT-COUNT SWEEP -- AF3 best-of-5, n=256 per cell, "
          "exact parents excluded.\n")
    print(t.sort_values(["site", "cond", "k"]).to_string(
        index=False, float_format=lambda v: f"{v:.1f}"))

    print("\nretention, trained at k vs trained at k=4 and cut at k:")
    for k in TRAINED_K:
        a = d[(d.cond == "trained") & (d.k == k)].set_index("site").retention
        b = d[(d.cond == "posthoc") & (d.k == k)].set_index("site").retention
        both = a.index.intersection(b.index)
        if not len(both):
            continue
        gap = 100 * (a[both] - b[both])
        print(f"  k={k:<3d} trained {100 * a[both].mean():5.1f}%   "
              f"post-hoc {100 * b[both].mean():5.1f}%   "
              f"gap {gap.mean():+5.1f} pts   "
              f"higher on {(gap > 0).sum()}/{len(both)} backbones")

    miss = [(s, k) for s in BB for k in (4,) + TRAINED_K
            if not len(d[(d.site == s) & (d.cond == 'trained') & (d.k == k)])]
    if miss:
        print("\nMISSING trained-at-k cells (jobs still running or failed):")
        for s, k in miss:
            print(f"  {s} k={k}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    d = pd.DataFrame(collect_trained() + collect_posthoc())
    if not len(d):
        sys.exit("no cells found")
    d = add_diversity(d, a.draws, rng)
    report(d)

    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.45, wspace=0.55)
    panel_retention(d, fig, gs)
    panel_rates(d, fig.add_subplot(gs[1, 0:2]))
    panel_trade(d, fig.add_subplot(gs[1, 3:6]))
    save(fig, "ame_fragk")
    d.drop(columns=["seqs"]).to_csv(HERE / "data" / "ame_fragk.csv", index=False)
    print(f"wrote {HERE / 'data' / 'ame_fragk.csv'}")


if __name__ == "__main__":
    main()
