#!/usr/bin/env python3
"""The library table: what fraction of the library folds, and how diverse it is.

Two numbers per arm, because they are the two things a combinatorial library is
bought on:

  pass%       of the recombined library, the fraction that clears the motif
              criterion. If this is low the library is mostly dead wells.
  diversity   distinct (position, residue) substitutions carried by the library
              members that PASS, rarefied to a matched count of passing members.

              Counted over passing chimeras directly rather than over fragments.
              Counting every fragment used would credit variety that only ever
              appeared in constructs that failed -- it would measure what was
              synthesised, not what works. Attributing to fragments instead and
              gating on "appeared in a passing chimera" reintroduces the
              attempt-count confound that killed the survival metric, since
              appearances per fragment depend on parent count. Scoring the
              passing members themselves avoids both.

Deliberately only these two. Earlier versions carried a per-fragment "survival"
rate, which measured whether *any* chimera containing a fragment passed -- and
that is confounded by parent count, since appearances per fragment are
(chimeras x slots) / n_fragments. An arm with 28 parents gave each fragment 9.1
attempts and scored 0.92 while its library folded 30% of the time. The column
was dropped rather than repaired: it answered a question nobody building a
library asks.

    uv run python paper/figures/ame_library_table.py [--draws 200]
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLEO = HERE.parents[1]
AME = CLEO / "experiments" / "ame"
sys.path.insert(0, str(AME))
from analyze_selection2 import as_matrix  # noqa: E402

ARMS = ("sampled", "RL", "RL+frag")


def passing_library(scored_csv):
    """-> (sequences of passing library members, pass % of the library)."""
    d = pd.read_csv(scored_csv)
    keep = d[d.rfd2_any_pass].drop_duplicates("sequence")
    return keep.sequence.tolist(), 100 * d.rfd2_any_pass.mean()


def subs(seqs, ref):
    M = as_matrix(list(seqs))
    return len({(j, M[i, j]) for i in range(M.shape[0])
                for j in range(M.shape[1]) if M[i, j] != ref[j]})


def locate(bb, arm):
    if arm == "sampled":
        return AME / "r1panel" / "recomb" / f"{bb}_baseline_scored.csv"
    if arm == "RL":
        return AME / "r1panel" / "recomb" / f"{bb}_policy_scored.csv"
    return AME / "fragrl2" / "recomb" / f"{bb}_frag2_scored.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    pub = pd.read_csv(CLEO / "paper/figures/data/rfd2_AME_per_site.csv")

    bbs = sorted({p.name[:-3] for p in (AME / "r1panel" / "eval").glob("run_*_r1")})
    rows, frag_store = [], {}
    for bb in bbs:
        site = "_".join(bb.replace("run_", "").split("_")[:2])
        for arm in ARMS:
            f = locate(bb, arm)
            if not Path(f).exists():
                continue
            seqs, passpct = passing_library(f)
            frag_store[(site, arm)] = seqs
            dead = pub.loc[pub.benchmark == site, "pct_backbones_0pass"]
            rows.append(dict(site=site, dead=dead.iloc[0] if len(dead) else np.nan,
                             arm=arm, n_pass=len(seqs), library_pass=passpct))

    # Rarefied to a matched count of PASSING members, so an arm cannot win on
    # diversity simply by having more of its library fold.
    div = {}
    for site in {s for s, _ in frag_store}:
        arms = {ar: frag_store[(site, ar)] for s, ar in frag_store if s == site}
        if len(arms) < 2:
            continue
        n = min(len(v) for v in arms.values())
        if n < 2:
            continue
        for arm, pool in arms.items():
            ref = np.array([max(set(c), key=list(c).count)
                            for c in as_matrix(pool).T], dtype="S1")
            div[(site, arm)] = float(np.mean(
                [subs(list(rng.choice(pool, n, replace=False)), ref)
                 for _ in range(a.draws)]))

    d = pd.DataFrame(rows)
    d["diversity"] = [div.get((r.site, r.arm), np.nan) for r in d.itertuples()]
    d = d.sort_values(["dead", "site", "arm"])
    pd.set_option("display.width", 200, "display.float_format", "{:.0f}".format)
    print("LIBRARY TABLE -- recombined libraries, 4 slots, AF3 best-of-5, n=256\n")
    print(d.to_string(index=False,
                      header=["site", "dead%", "arm", "n_pass", "pass%", "novel_muts"],
                      float_format=lambda v: f"{v:.1f}"))

    for metric in ("library_pass", "diversity"):
        p = d.pivot(index="site", columns="arm", values=metric)
        if "RL" not in p or "sampled" not in p:
            continue
        b = p.dropna(subset=["RL", "sampled"])
        print(f"\n{metric}: RL > sampled on {(b.RL > b.sampled).sum()}/{len(b)}"
              f"   mean {b.RL.mean():.1f} vs {b.sampled.mean():.1f}")
        if "RL+frag" in p:
            c = p.dropna(subset=["RL", "RL+frag"])
            if len(c):
                print(f"{' ' * len(metric)}  RL+frag > RL on "
                      f"{(c['RL+frag'] > c.RL).sum()}/{len(c)}"
                      f"   mean {c['RL+frag'].mean():.1f} vs {c.RL.mean():.1f}")


if __name__ == "__main__":
    main()
