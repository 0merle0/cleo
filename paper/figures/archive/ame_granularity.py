#!/usr/bin/env python3
"""Library success rate against how finely a design is cut.

Two panels over the same x-axis, the number of slots a design is split into:

  top     retention -- the fraction of recombinants that still fold. Falls with
          k, because shorter fragments carry less of their own context.
  bottom  fragment collision -- distinct fragments per slot as a fraction of
          parents. At coarse granularity every parent contributes a unique
          fragment to every slot; as fragments shorten the designs start sharing
          them, which is degradation visible without folding anything.

Library size is deliberately NOT plotted. At a fixed synthesis budget both arms
build the same-size library -- each slot takes budget/k fragments regardless of
which arm produced them -- so the curves lie on top of each other and the panel
shows nothing. That is itself the finding: the arms do not differ in how large a
library they can build, only in what fraction of it folds.

Includes k=4 from the main recombination panel, so the curve is anchored on the
measurement everything else in the paper rests on.

    uv run python paper/figures/ame_granularity.py
"""

import argparse
import re
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

PARTS = 1000          # synthesised fragments the whole library may use
RETENTION_FLOOR = 0.5  # below this you fold two constructs per usable one

C = {"policy": PALETTE["blue"], "baseline": PALETTE["gray"]}
LABEL = {"policy": "RL-trained", "baseline": "LigandMPNN"}
MARK = {"M0097": "o", "M0255": "s", "M0315": "^"}


def library_size(site, arm, k, parts_budget=None):
    """log10 distinct recombinants: product over slots of distinct fragments.

    With `parts_budget` set, each slot is capped at `budget / k` fragments. That
    is the comparison a synthesis budget actually imposes, and it is the only
    fair one between arms: the policy has 199 passing parents on M0315 and the
    baseline 926, so an uncapped product rewards the arm that happened to
    produce more parents rather than the one whose parts recombine.

    NOT P^k. Fragments collide as they shorten -- at 7 residues one M0097 slot
    holds 21 distinct variants from 178 parents -- and P^k overstates the
    library by 10^15 at k=36. The product of actual distinct counts is the
    number of constructs that exist.
    """
    ev = sorted((AME / "r1panel" / "eval").glob(f"run_{site}*/*_af3scored.csv"))
    if not ev:
        return np.nan
    bb = ev[0].parent.name[:-3]
    p = ev[0] if arm == "policy" else AME / "r1panel" / f"{bb}_baseline" / "baseline_scored.csv"
    if not Path(p).exists():
        return np.nan
    d = pd.read_csv(p)
    seqs = d[d.rfd2_any_pass].drop_duplicates("sequence").sequence.tolist()
    if len(seqs) < 2:
        return np.nan
    L = len(seqs[0])
    e = [round(i * L / k) for i in range(k + 1)]
    counts = [len({q[e[i]:e[i + 1]] for q in seqs}) for i in range(k)]
    if parts_budget:
        counts = [min(c, max(parts_budget // k, 1)) for c in counts]
    return float(np.sum(np.log10(counts)))


def parents_for(site, arm):
    """Passing-parent count."""
    ev = sorted((AME / "r1panel" / "eval").glob(f"run_{site}*/*_af3scored.csv"))
    if not ev:
        return np.nan
    bb = ev[0].parent.name[:-3]
    p = ev[0] if arm == "policy" else AME / "r1panel" / f"{bb}_baseline" / "baseline_scored.csv"
    if not Path(p).exists():
        return np.nan
    d = pd.read_csv(p)
    return int(d[d.rfd2_any_pass].sequence.nunique())


def collision(site, arm, k):
    """Mean distinct fragments per slot, as a fraction of parents."""
    ev = sorted((AME / "r1panel" / "eval").glob(f"run_{site}*/*_af3scored.csv"))
    if not ev:
        return np.nan
    bb = ev[0].parent.name[:-3]
    p = ev[0] if arm == "policy" else AME / "r1panel" / f"{bb}_baseline" / "baseline_scored.csv"
    if not Path(p).exists():
        return np.nan
    d = pd.read_csv(p)
    seqs = d[d.rfd2_any_pass].drop_duplicates("sequence").sequence.tolist()
    if len(seqs) < 2:
        return np.nan
    L = len(seqs[0])
    e = [round(i * L / k) for i in range(k + 1)]
    counts = [len({q[e[i]:e[i + 1]] for q in seqs}) for i in range(k)]
    return float(np.mean(counts) / len(seqs))


def collect():
    rows = []
    # k != 4 from the sweep
    for f in sorted((AME / "slotsweep").glob("*_scored.csv")):
        if "_samples" in f.name:
            continue
        m = re.match(r"(M\d+)_(policy|baseline)_k(\d+)_scored\.csv", f.name)
        if not m:
            continue
        site, arm, k = m.group(1), m.group(2), int(m.group(3))
        d = pd.read_csv(f)
        rows.append(dict(site=site, arm=arm, k=k, retention=d.rfd2_any_pass.mean()))
    # k = 4 from the main panel
    for f in sorted((AME / "r1panel" / "recomb").glob("*_scored.csv")):
        if "_samples" in f.name:
            continue
        b = f.name.replace("_scored.csv", "")
        arm = "policy" if b.endswith("_policy") else "baseline"
        site = "_".join(re.sub(r"_(policy|baseline)$", "", b).replace("run_", "").split("_")[:1])
        if site not in MARK:
            continue
        d = pd.read_csv(f)
        rows.append(dict(site=site, arm=arm, k=4, retention=d.rfd2_any_pass.mean()))
    df = pd.DataFrame(rows).drop_duplicates(["site", "arm", "k"])
    df["parents"] = [parents_for(r.site, r.arm) for r in df.itertuples()]
    df["log10_lib"] = [library_size(r.site, r.arm, r.k) for r in df.itertuples()]
    df["log10_lib_budget"] = [library_size(r.site, r.arm, r.k, PARTS) for r in df.itertuples()]
    # Expected working constructs at a fixed synthesis budget, in log space.
    df["log10_working"] = df.log10_lib_budget + np.log10(df.retention.clip(lower=1e-9))
    df["collision"] = [collision(r.site, r.arm, r.k) for r in df.itertuples()]
    return df.sort_values(["site", "arm", "k"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    d = collect()
    if d.empty:
        sys.exit("no slot-sweep data yet")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)
    for (site, arm), g in d.groupby(["site", "arm"]):
        g = g.sort_values("k")
        style = dict(color=C[arm], marker=MARK.get(site, "o"), ms=6, lw=1.8,
                     mec="white", mew=1.0,
                     ls="-" if arm == "policy" else "--",
                     alpha=1.0 if arm == "policy" else 0.75)
        ax1.plot(g.k, 100 * g.retention, **style)
        ax2.plot(g.k, 100 * g.collision, **style)

    ax1.axhline(100 * RETENTION_FLOOR, color="#B23A3A", lw=1.0, ls=":", zorder=0)
    ax1.annotate(f"{RETENTION_FLOOR:.0%} floor", (0.99, 100 * RETENTION_FLOOR),
                 xycoords=("axes fraction", "data"), ha="right", va="bottom",
                 fontsize=7.5, color="#B23A3A")
    ax1.set_ylabel("recombinants that fold (%)", fontsize=9.5)
    ax2.set_ylabel("distinct fragments per slot\n(% of parents)", fontsize=9.5)
    ax2.set_xlabel("fragments per design ($k$)", fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.22, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks(sorted(d.k.unique()))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(0, 105)

    h = [plt.Line2D([], [], color=C[a], ls="-" if a == "policy" else "--", lw=1.8,
                    label=LABEL[a]) for a in ("policy", "baseline")]
    h += [plt.Line2D([], [], color="#4B5563", marker=MARK[s], ls="none", ms=6, label=s)
          for s in MARK if s in set(d.site)]
    ax1.legend(handles=h, fontsize=8, frameon=False, ncol=2, loc="lower left")

    fig.tight_layout()
    save(fig, "ame_granularity")

    pd.set_option("display.width", 200, "display.float_format", "{:.3g}".format)
    print(d.to_string(index=False))
    for (site, arm), g in d.groupby(["site", "arm"]):
        if len(g) < 2:
            continue
        best = g.loc[g.log10_working.idxmax()]
        print(f"  {site} {arm:8s}: best k = {best.k:.0f}  "
              f"({10**best.log10_working:.3g} working constructs)")


if __name__ == "__main__":
    main()
