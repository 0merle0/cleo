#!/usr/bin/env python3
"""R4: what the fragment libraries look like after recombination.

Takes the passing designs each arm produced at a matched fold budget, splits
them into 4 equal fragments with the pipeline's own ``split_into_fragments``,
and asks what recombining those fragments buys.

Two comparisons, because they answer different questions and disagree:

**As sampled** -- each arm keeps the parents its own matched budget produced.
This is the practical question: given 2,656 folds, whose library is bigger?
It is confounded with pass rate, deliberately, because pass rate is what you
actually pay for.

**Matched parents** -- both arms subsampled to the same parent count. This
isolates whether the *fragments themselves* differ, independent of how many
parents each arm managed to produce.

The library size is the product over slots of distinct fragments, and the
synthesis cost is the sum. That asymmetry (product vs sum) is the whole reason
fragment recombination is worth doing, and it means parent count matters far
more than per-parent diversity.

    uv run python paper/figures/ame_fragments.py [--slots 4] [--sites ...]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLEO = HERE.parents[1]
sys.path.insert(0, str(CLEO / "src"))
from cleo.design.sample_from_policy import split_into_fragments  # noqa: E402

PANEL = CLEO / "experiments" / "ame" / "r1panel"


def equal_bounds(length, k):
    """k equal [start, end] inclusive slots covering 0..length-1."""
    edges = [round(i * length / k) for i in range(k + 1)]
    return [[edges[i], edges[i + 1] - 1] for i in range(k)]


def frag_stats(seqs, k):
    """-> (per-slot distinct counts, product, sum)."""
    if not seqs:
        return [0] * k, 0, 0
    bounds = equal_bounds(len(seqs[0]), k)
    d = split_into_fragments(list(seqs), bounds)
    counts = [len(d[str(i + 1)]) for i in range(k)]
    return counts, math.prod(counts), sum(counts)


def passing_seqs(path):
    if not Path(path).exists():
        return []
    df = pd.read_csv(path)
    return df[df.rfd2_any_pass].drop_duplicates("sequence").sequence.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--draws", type=int, default=50,
                    help="resamples for the matched-parent comparison")
    ap.add_argument("--sites", default="M0664,M0097,M0255,M0315",
                    help="site prefixes, easiest first")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    rows = []
    for site in a.sites.split(","):
        evals = sorted(PANEL.glob(f"eval/run_{site}*/*_af3scored.csv"))
        if not evals:
            continue
        stem = evals[0].parent.name
        bb = stem[:-3]
        pol = passing_seqs(evals[0])
        base = passing_seqs(PANEL / f"{bb}_baseline" / "baseline_scored.csv")
        if not pol or not base:
            continue

        pc, pprod, psum = frag_stats(pol, a.slots)
        bc, bprod, bsum = frag_stats(base, a.slots)

        # Matched parents: subsample both arms to the smaller count.
        n = min(len(pol), len(base))
        pm, bm = [], []
        for _ in range(a.draws):
            pm.append(frag_stats(list(rng.choice(pol, n, replace=False)), a.slots)[1])
            bm.append(frag_stats(list(rng.choice(base, n, replace=False)), a.slots)[1])

        # Do the two arms reach the same fragments at all?
        bounds = equal_bounds(len(pol[0]), a.slots)
        shared = []
        for i, (s, e) in enumerate(bounds):
            fp = {q[s:e + 1] for q in pol}
            fb = {q[s:e + 1] for q in base}
            shared.append(len(fp & fb))

        rows.append(dict(site=site, n_pol=len(pol), n_base=len(base),
                         pol_slots=pc, base_slots=bc,
                         pol_lib=pprod, base_lib=bprod,
                         pol_parts=psum, base_parts=bsum,
                         matched_n=n, pol_lib_m=np.mean(pm), base_lib_m=np.mean(bm),
                         shared_frags=sum(shared)))

    d = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"FRAGMENT RECOMBINATION, {a.slots} equal slots\n")
    print("AS SAMPLED (each arm keeps what its matched budget produced)")
    for _, r in d.iterrows():
        print(f"  {r.site}  policy {r.n_pol:5d} parents -> {r.pol_lib:.3g} recombinants "
              f"from {r.pol_parts:5d} parts")
        print(f"  {' '*len(r.site)}  base   {r.n_base:5d} parents -> {r.base_lib:.3g} recombinants "
              f"from {r.base_parts:5d} parts")
    print("\nMATCHED PARENTS (both subsampled to the smaller count)")
    for _, r in d.iterrows():
        ratio = r.pol_lib_m / r.base_lib_m if r.base_lib_m else float("nan")
        print(f"  {r.site}  n={r.matched_n:5d}   policy {r.pol_lib_m:.3g}   "
              f"base {r.base_lib_m:.3g}   ratio {ratio:.2f}x")
    print("\nFRAGMENT OVERLAP (distinct fragments both arms reach, summed over slots)")
    for _, r in d.iterrows():
        tot = r.pol_parts + r.base_parts - r.shared_frags
        print(f"  {r.site}  shared {r.shared_frags:5d} of {tot:6d} distinct "
              f"({100*r.shared_frags/max(tot,1):.1f}%)")


if __name__ == "__main__":
    main()
