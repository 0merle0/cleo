#!/usr/bin/env python3
"""Select the AME pilot target set.

Panels drive the selection (see paper/figure_plan.md):
  - Figure 2 (diversity on working backbones) needs backbones with >=1 passing
    sequence, drawn from the 964.
  - Figure 3 (rescue) needs backbones with 0 passing sequences, drawn from the
    3,136.
  - Panel 3C needs a spread of site difficulty, so sites are chosen across the
    published per-site pass-rate range rather than where we look best.

Clash-failing backbones are excluded from the FAIL class: no_clash is constant
across all 40 sequences of a backbone, so it is a backbone property that
sequence design cannot repair. 299 of the 3,136 zero-pass backbones are
clash-blocked; the true rescue set is 2,837.

Writes a manifest of tar members to extract, plus the selection table.
"""

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "paper/figures/data/rfd2_AME_backbone_index.csv"
OUT_DIR = Path(__file__).resolve().parent


def _spread(pool_size, k):
    """k distinct positions spread evenly across a pool of pool_size.

    Rounding collisions on small pools would otherwise select the same backbone
    more than once, which silently shrinks the pilot set.
    """
    if pool_size < k:
        raise SystemExit(f"pool of {pool_size} too small to pick {k}")
    idx, seen = [], set()
    for i in range(k):
        j = round(i * (pool_size - 1) / (k - 1)) if k > 1 else 0
        while j in seen:  # collision: take the next free slot
            j += 1
        seen.add(j)
        idx.append(j)
    return idx


def pick_sites(d, n_sites, n_each):
    """Sites spanning the difficulty range, restricted to those with enough of
    both classes. Difficulty = fraction of backbones with >=1 passing sequence."""
    # Count each pool from its own predicate. n_pass_bb must NOT be derived as
    # n_bb - n_fail: clash-blocked failures belong to neither pool, so that
    # subtraction silently counts them as passing and overstates the PASS pool.
    s = (d.groupby("benchmark")
           .agg(n_bb=("design_id", "size"),
                n_pass_bb=("rescue_target", lambda x: (~x).sum()),
                n_fail=("true_rescue_target", "sum"))
           .reset_index())
    s["pct_pass_bb"] = s.n_pass_bb / s.n_bb * 100
    ok = s[(s.n_pass_bb >= n_each) & (s.n_fail >= n_each)].sort_values("pct_pass_bb")
    if len(ok) < n_sites:
        raise SystemExit(f"only {len(ok)} sites have >={n_each} of each class")
    # Evenly spaced quantiles across the difficulty range: hard -> easy.
    idx = [round(i * (len(ok) - 1) / (n_sites - 1)) for i in range(n_sites)]
    return ok.iloc[idx].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sites", type=int, default=3)
    ap.add_argument("--n-each", type=int, default=5, help="backbones per class per site")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = pd.read_csv(INDEX)
    sites = pick_sites(d, a.n_sites, a.n_each)

    rows = []
    for _, s in sites.iterrows():
        sub = d[d.benchmark == s.benchmark]
        # PASS: spread across the passing-count range rather than only the best,
        # so Figure 2 is not built on outliers.
        p = sub[~sub.rescue_target].sort_values("n_pass")
        pick_p = p.iloc[_spread(len(p), a.n_each)]
        # FAIL: uniform random among TRUE rescue targets only. Backbones that
        # fail on clash are excluded -- ligand_dist_des_ncac_min is constant
        # across all 40 sequences of a backbone, so a clash failure is a
        # property of the backbone that no sequence can fix.
        pick_f = sub[sub.true_rescue_target].sample(a.n_each, random_state=a.seed)
        for cls, chunk in (("pass", pick_p), ("fail", pick_f)):
            c = chunk.copy()
            c["cls"] = cls
            c["site_pct_pass_bb"] = round(s.pct_pass_bb, 1)
            rows.append(c)

    sel = pd.concat(rows, ignore_index=True)
    sel = sel[["benchmark", "site_pct_pass_bb", "cls", "design_id", "pdb", "n_pass", "n_seq", "clash_ok"]]
    sel.to_csv(OUT_DIR / "pilot_targets.csv", index=False)

    # Manifest for a single streaming pass over the 12.6 GB tarball: both the
    # .pdb and its .trb (the .trb carries the contig mapping we need for
    # fixed_residues).
    members = []
    for p in sel.pdb:
        members += [p, p.replace(".pdb", ".trb")]
    (OUT_DIR / "pilot_members.txt").write_text("\n".join(members) + "\n")

    print(f"sites: {', '.join(sites.benchmark)}")
    print(sites[["benchmark", "n_bb", "n_pass_bb", "pct_pass_bb"]].to_string(index=False))
    print(f"\nselected {len(sel)} backbones ({(sel.cls=='pass').sum()} pass / "
          f"{(sel.cls=='fail').sum()} fail), {len(members)} tar members")
    print(f"-> {OUT_DIR/'pilot_targets.csv'}")
    print(f"-> {OUT_DIR/'pilot_members.txt'}")


if __name__ == "__main__":
    main()
