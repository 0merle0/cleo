#!/usr/bin/env python3
"""What cutting into more fragments does to library diversity.

`ame_fragk.py` answers whether a library still folds as k rises. This answers
what it costs in variety, which is a different question with a different sign,
and the two together are the trade the paper argues over.

"Diversity" is not one number. Three senses pull apart as k rises, so all three
are measured rather than one being chosen:

  reach        log10 of the product over slots of distinct fragments per slot.
               The size of the combinatorial space the ordered parts span. More
               slots multiply, shorter fragments collide, and the product is
               where those two fight.
  collision    distinct fragments per slot as a fraction of parents. At coarse
               cuts every parent gives a unique fragment to every slot; as
               fragments shorten they start repeating. This is degradation
               visible without folding anything.
  substitution distinct (position, residue) substitutions carried by library
               members that PASS, rarefied to a matched count. What variety
               actually survives into usable constructs.

Reach is counted as the product of ACTUAL distinct fragment counts, never P^k.
Fragments collide as they shorten, so P^k overstates the space by orders of
magnitude at high k -- the same correction `ame_granularity.py` makes.

Reach and substitution diversity are not redundant. Reach counts what could be
assembled from the parts; substitution diversity counts what the folding
oracle still accepts. A k that multiplies the first while shrinking the second
is buying library size with dead wells, and only plotting both shows it.

    uv run python paper/figures/ame_fragk_diversity.py [--draws 200]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
from make_recombinants import equal_bounds  # noqa: E402

BB = {"M0097": "run_M0097_1ctt_cond9_14",
      "M0255": "run_M0255_1mg5_cond17_66",
      "M0315": "run_M0315_1ey3_cond18_25"}
ANCHOR = {"M0097": ("fragbsweep", "bs16"),
          "M0255": ("fragb16", "bs16"),
          "M0315": ("fragb16", "bs16")}
KS = (2, 4, 8, 16, 36, 64)
# Training length confounds diversity: converging narrows a policy, so a cell
# that got more gradient steps looks less diverse for a reason that has nothing
# to do with k. M0255/k=4 is the demonstration -- it has the MOST passing
# parents of its backbone's cells and the narrowest (pairwise 42.4 against
# 58-81), and that is measured on passing sequences only, so gating on pass
# does not remove the effect.
#
# The fix is matched training, not converged training. Every `fragk` arm ran
# exactly 200 steps (M0315/k=16 is the lone exception at 166); the k=4 anchor
# is borrowed from another family at 150/300/152. So k=4 is excluded from the
# trend and drawn hollow, and the comparison that carries a claim is over
# MATCHED. `steps` is carried in the output so this stays checkable rather than
# resting on this comment.
MATCHED_STEPS = 200.0
MARK = {"M0097": "o", "M0255": "s", "M0315": "^"}
BLUE = PALETTE["blue"]


def candidates(site, k):
    """-> (family, stem) to try, best first.

    E18b trained k=4 in-family and reran M0315/k=16 as `fragk16b` after its
    SIGTERM at step 166, so every level is a matched 200 steps. The borrowed
    anchor stays last as a fallback for a tree where E18b has not run.
    """
    c = [("fragk", f"{BB[site]}_fragk{k}")]
    if k == 16:
        c.insert(0, ("fragk", f"{BB[site]}_fragk16b"))
    if k == 4:
        fam, suf = ANCHOR[site]
        c.append((fam, f"{BB[site]}_{suf}"))
    return c


def resolve(site, k):
    """-> (eval csv, recomb csv, family, stem) for the best cell on disk."""
    for fam, stem in candidates(site, k):
        ev = AME / fam / "eval" / stem / f"{stem}_af3scored.csv"
        rc = AME / fam / "recomb" / f"{stem}_scored.csv"
        if ev.exists() and rc.exists():
            return ev, rc, fam, stem
    fam, stem = candidates(site, k)[0]
    return (AME / fam / "eval" / stem / f"{stem}_af3scored.csv",
            AME / fam / "recomb" / f"{stem}_scored.csv", fam, stem)


def cell(site, k):
    ev, rc, *_ = resolve(site, k)
    return ev, rc


def train_steps(site, k):
    """-> how many steps this cell actually trained for, or nan."""
    _, _, fam, stem = resolve(site, k)
    p = AME / fam / stem / stem / f"{stem}_train_metrics.csv"
    if not p.exists():
        return float("nan")
    return float(pd.read_csv(p, usecols=["step"]).step.max() + 1)


def passing(path):
    d = pd.read_csv(path)
    return d[d.rfd2_any_pass].drop_duplicates("sequence").sequence.tolist()


def slot_counts(seqs, k):
    """-> distinct fragments in each of the k slots, over the parent pool."""
    L = len(seqs[0])
    e = [round(i * L / k) for i in range(k + 1)]
    return [len({q[e[i]:e[i + 1]] for q in seqs}) for i in range(k)]


def mean_pairwise(seqs, rng, cap=200):
    """-> mean pairwise Hamming distance, on a capped random subsample."""
    if len(seqs) > cap:
        seqs = list(rng.choice(seqs, cap, replace=False))
    M = as_matrix(list(seqs))
    n = len(M)
    if n < 2:
        return np.nan
    tot = sum((M[i] != M[j]).sum() for i in range(n) for j in range(i + 1, n))
    return float(tot / (n * (n - 1) / 2))


def subs(seqs, ref):
    M = as_matrix(list(seqs))
    return len({(j, M[i, j]) for i in range(M.shape[0])
                for j in range(M.shape[1]) if M[i, j] != ref[j]})


def collect(draws, rng):
    rows, pools = [], {}
    for site in BB:
        for k in KS:
            ev, rc = cell(site, k)
            if not (ev.exists() and rc.exists()):
                continue
            par, chi = passing(ev), passing(rc)
            if len(par) < 2 or len(chi) < 2:
                continue
            counts = slot_counts(par, k)
            pools[(site, k)] = chi
            rows.append(dict(
                site=site, k=k, matched=train_steps(site, k) == MATCHED_STEPS,
                steps=train_steps(site, k),
                parents=len(par), n_pass=len(chi),
                reach=float(np.sum(np.log10(counts))),
                frags_per_slot=float(np.mean(counts)),
                collision=float(np.mean(counts) / len(par)),
                pairwise=mean_pairwise(chi, rng)))
    d = pd.DataFrame(rows)
    if not len(d):
        return d
    # Rarefy substitution diversity within each backbone, so levels are
    # comparable down a column and a level cannot win by having more of its
    # library fold.
    # Rarefaction depth is set by the MATCHED cells only. Letting the k=4
    # anchor set it would drag every matched level down to the depth of a run
    # that is not part of the comparison. A cell whose pool is shallower than
    # that depth (k=4 on two backbones) gets NaN rather than a point rarefied
    # to a different depth and plotted as if comparable.
    d["substitution"] = np.nan
    for site in BB:
        g = {k: v for (s_, k), v in pools.items() if s_ == site}
        ok = set(d[(d.site == site) & d.matched].k)
        m = {k: v for k, v in g.items() if k in ok}
        if len(m) < 2:
            continue
        n = min(len(v) for v in m.values())
        for k, pool in g.items():
            if len(pool) < n:
                continue
            ref = np.array([max(set(c), key=list(c).count)
                            for c in as_matrix(pool).T], dtype="S1")
            d.loc[(d.site == site) & (d.k == k), "substitution"] = float(np.mean(
                [subs(list(rng.choice(pool, n, replace=False)), ref)
                 for _ in range(draws)]))
        d.loc[d.site == site, "rarefied_to"] = n
    return d


def line(ax, d, col, title, ylabel, logy=False):
    for site in sorted(BB):
        g = d[d.site == site].dropna(subset=[col]).sort_values("k")
        if not len(g):
            continue
        # The trend is drawn over matched-length cells only; the k=4 anchor is
        # plotted hollow and off the line so it cannot be read as part of it.
        m = g[g.matched]
        ax.plot(m.k, m[col], "-", color=BLUE, lw=1.5, marker=MARK[site],
                ms=8, mec="white", mew=1, label=site)
        a = g[~g.matched]
        ax.plot(a.k, a[col], MARK[site], ms=8, mfc="white", mec=BLUE, mew=1.5,
                ls="none", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(KS))
    ax.set_xticklabels([str(k) for k in KS])
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("fragments per design, $k$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, color="#E5E7EB", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def report(d):
    pd.set_option("display.width", 220)
    print("\nDIVERSITY ACROSS THE FRAGMENT-COUNT SWEEP "
          "(trained at k, recombined at the same k)\n")
    print(d.sort_values(["site", "k"]).to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))
    ragged = d[~d.matched]
    if len(ragged):
        print("\nEXCLUDED from the trend (training length not matched): "
              + ", ".join(f"{r.site} k={int(r.k)} at {int(r.steps)} steps"
                          for r in ragged.itertuples()))
    print("\ndirection of each measure, matched-length cells only:")
    for site in sorted(BB):
        g = d[(d.site == site) & d.matched].sort_values("k")
        if len(g) < 2:
            continue
        a, b = g.iloc[0], g.iloc[-1]
        print(f"  {site}  k={int(a.k)} -> {int(b.k)}:  "
              f"reach {a.reach:.1f} -> {b.reach:.1f} log10   "
              f"collision {a.collision:.3f} -> {b.collision:.3f}   "
              f"substitution {a.substitution:.0f} -> {b.substitution:.0f}   "
              f"pairwise {a.pairwise:.1f} -> {b.pairwise:.1f}")
    miss = [(s_, k) for s_ in BB for k in KS
            if not len(d[(d.site == s_) & (d.k == k)])]
    if miss:
        print("\nMISSING cells (not yet folded):")
        print("  " + ", ".join(f"{s_} k={k}" for s_, k in miss))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    d = collect(a.draws, rng)
    if not len(d):
        sys.exit("no cells found")
    report(d)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    line(axes[0], d, "reach", "combinatorial reach grows with $k$",
         "log$_{10}$ distinct recombinants")
    line(axes[1], d, "collision", "but fragments start repeating",
         "distinct fragments per slot / parents")
    line(axes[2], d, "substitution", "variety that survives folding",
         "distinct substitutions (rarefied)")
    line(axes[3], d, "pairwise", "and how far apart members sit",
         "mean pairwise Hamming")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "ame_fragk_diversity")
    d.to_csv(HERE / "data" / "ame_fragk_diversity.csv", index=False)
    print(f"wrote {HERE / 'data' / 'ame_fragk_diversity.csv'}")


if __name__ == "__main__":
    main()
