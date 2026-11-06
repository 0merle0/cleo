#!/usr/bin/env python3
"""Diversity of the fragments that survive recombination.

A combinatorial library is built from the parts that still work after shuffling,
not from every part you could cut. So the diversity that matters is over
*surviving* fragments -- those appearing in at least one passing chimera -- and
not over all designs.

Reported per (backbone, arm, slot):

  n_frag       fragments entering recombination (= parents, since at 45 residues
               no two passing designs share one)
  n_surv       fragments appearing in >= 1 passing chimera
  surv_frac    n_surv / n_frag -- CONFOUNDED by parent count, kept only for
               continuity. A fragment survives if ANY of its chimeras passes,
               and appearances per fragment = (chimeras x slots) / n_fragments,
               so an arm with fewer parents gives each fragment more attempts.
               M0255's chimera-objective arm had 28 parents, 9.1 attempts per
               fragment, and scored 0.92 at 30% retention -- 1-0.7^9 = 0.96,
               which is the number, not modularity.
  frag_rate    mean per-fragment pass RATE. Attempt-count independent, matches
               a one-attempt resampling of the same data, and is the quantity
               the fragment-level reward itself computes. Use this one.
  U_surv       distinct (position, residue) substitutions across surviving
               fragments, vs the slot consensus
  ham_surv     mean pairwise Hamming among surviving fragments

The two diversity columns answer different questions and the parent-count
confound hits them differently. `U_surv` grows with how many fragments survived,
so it is reported both raw and rarefied to a matched count; `ham_surv` is a mean
over pairs and is already count-independent.

    uv run python paper/figures/ame_surviving_fragments.py [--draws 200]
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLEO = HERE.parents[1]
PANEL = CLEO / "experiments" / "ame" / "r1panel"
sys.path.insert(0, str(CLEO / "experiments" / "ame"))
from analyze_selection2 import as_matrix  # noqa: E402


def fragment_table(scored_csv):
    """-> {slot: {frag_seq: [pass flags]}} from a scored chimera set."""
    df = pd.read_csv(scored_csv)
    if "parent_fragments" not in df.columns:
        src = Path(str(scored_csv).replace("_scored.csv", ".csv"))
        df = df.merge(pd.read_csv(src)[["name", "parent_fragments"]], on="name", how="left")
    df = df.dropna(subset=["parent_fragments"])
    out = defaultdict(lambda: defaultdict(list))
    for _, r in df.iterrows():
        names = str(r.parent_fragments).split("___")
        k, L = len(names), len(r.sequence)
        edges = [round(i * L / k) for i in range(k + 1)]
        for i, nm in enumerate(names):
            out[int(nm.split(".")[0])][r.sequence[edges[i]:edges[i + 1]]].append(
                bool(r.rfd2_any_pass))
    return out


def subs_count(seqs, ref):
    if not seqs:
        return 0
    M = as_matrix(list(seqs))
    return len({(j, M[i, j]) for i in range(M.shape[0])
                for j in range(M.shape[1]) if M[i, j] != ref[j]})


def mean_pairwise(seqs):
    if len(seqs) < 2:
        return 0.0
    M = as_matrix(list(seqs))
    n = len(M)
    tot = sum((M[i] != M[j]).sum() for i in range(n) for j in range(i + 1, n))
    return 2.0 * tot / (n * (n - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    pub = pd.read_csv(CLEO / "paper/figures/data/rfd2_AME_per_site.csv")

    cells = {}
    for f in sorted(PANEL.glob("recomb/*_scored.csv")):
        if "_samples" in f.name:
            continue
        b = f.name.replace("_scored.csv", "")
        arm = "policy" if b.endswith("_policy") else "baseline"
        bb = re.sub(r"_(policy|baseline)$", "", b)
        site = "_".join(bb.replace("run_", "").split("_")[:2])
        cells.setdefault(site, {})[arm] = fragment_table(f)

    rows = []
    for site, arms in sorted(cells.items()):
        dead = pub.loc[pub.benchmark == site, "pct_backbones_0pass"]
        dead = dead.iloc[0] if len(dead) else np.nan
        for arm, tbl in arms.items():
            for slot, frags in sorted(tbl.items()):
                surv = [s for s, flags in frags.items() if any(flags)]
                allf = list(frags)
                if not allf:
                    continue
                ref = np.array([max(set(c), key=list(c).count)
                                for c in as_matrix(allf).T], dtype="S1")
                rates = [float(np.mean(fl)) for fl in frags.values()]
                rows.append(dict(site=site, dead=dead, arm=arm, slot=slot,
                                 n_frag=len(allf), n_surv=len(surv),
                                 appear=float(np.mean([len(fl) for fl in frags.values()])),
                                 frag_rate=float(np.mean(rates)),
                                 surv_frac=len(surv) / len(allf),
                                 U_surv=subs_count(surv, ref),
                                 ham_surv=mean_pairwise(surv)))
    d = pd.DataFrame(rows)
    if not len(d):
        sys.exit("no recombination cells found")

    # Rarefy U to a matched surviving-fragment count, per site, so the two arms
    # are compared at equal depth rather than at whatever each happened to yield.
    rare = []
    for site, arms in sorted(cells.items()):
        if len(arms) < 2:
            continue
        for slot in sorted(arms["policy"]):
            sets = {}
            for arm, tbl in arms.items():
                frags = tbl.get(slot, {})
                sets[arm] = [s for s, fl in frags.items() if any(fl)]
            n = min(len(v) for v in sets.values())
            if n < 2:
                continue
            for arm, surv in sets.items():
                allf = list(arms[arm][slot])
                ref = np.array([max(set(c), key=list(c).count)
                                for c in as_matrix(allf).T], dtype="S1")
                vals = [subs_count(list(rng.choice(surv, n, replace=False)), ref)
                        for _ in range(a.draws)]
                rare.append(dict(site=site, arm=arm, slot=slot, n=n,
                                 U_rare=float(np.mean(vals))))
    r = pd.DataFrame(rare)

    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format)
    print("SURVIVING FRAGMENTS -- those appearing in >= 1 passing chimera\n")
    agg = d.groupby(["dead", "site", "arm"]).agg(
        n_frag=("n_frag", "sum"), appear=("appear", "mean"),
        frag_rate=("frag_rate", "mean"), surv_frac=("surv_frac", "mean"),
        U_surv=("U_surv", "sum"), ham=("ham_surv", "mean")
    ).reset_index().sort_values(["dead", "site", "arm"])
    print(agg.to_string(index=False))

    piv = agg.pivot(index=["dead", "site"], columns="arm", values="frag_rate")
    both = piv.dropna()
    if len(both):
        print(f"\nper-fragment pass rate: policy > baseline on "
              f"{(both.policy > both.baseline).sum()}/{len(both)} backbones")
        print(f"  mean  policy {both.policy.mean():.2f}   baseline {both.baseline.mean():.2f}")

    if len(r):
        rp = r.groupby(["site", "arm"]).U_rare.sum().unstack()
        rp = rp.dropna()
        print("\nU over surviving fragments, RAREFIED to matched count (summed over slots)")
        print(rp.to_string())
        print(f"\n  policy > baseline on {(rp.policy > rp.baseline).sum()}/{len(rp)}")


if __name__ == "__main__":
    main()
