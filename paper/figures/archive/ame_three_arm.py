#!/usr/bin/env python3
"""The combinability table: sampled vs RL-trained vs trained-for-recombination.

One row per backbone and arm, with the three numbers a combinatorial library is
judged on:

  parents      passing designs the arm produced at its matched fold budget
  retention    % of chimeras built from them that still pass
  surv_frac    % of fragments appearing in >= 1 passing chimera
  U_rare       substitution diversity over surviving fragments, rarefied to a
               matched survivor count so pool size cannot explain a difference

The pair (surv_frac, U_rare) is what separates a genuinely more modular policy
from one that collapsed. Training for recombinability can be satisfied by
emitting near-identical sequences: every chimera then reproduces a parent and
passes, so surv_frac climbs toward 1.0 while U_rare falls. Retention alone
cannot see that; the pair can.

    uv run python paper/figures/ame_three_arm.py
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
AME = CLEO / "experiments" / "ame"
PANEL, FRAGRL = AME / "r1panel", AME / "fragrl"
sys.path.insert(0, str(AME))
from analyze_selection2 import as_matrix  # noqa: E402

ARMS = ("baseline", "policy", "fragrl")


def _frag_table(scored):
    df = pd.read_csv(scored)
    if "parent_fragments" not in df.columns:
        src = Path(str(scored).replace("_scored.csv", ".csv"))
        if not src.exists():
            return None, None
        df = df.merge(pd.read_csv(src)[["name", "parent_fragments"]], on="name", how="left")
    df = df.dropna(subset=["parent_fragments"])
    tbl = defaultdict(lambda: defaultdict(list))
    for _, r in df.iterrows():
        names = str(r.parent_fragments).split("___")
        k, L = len(names), len(r.sequence)
        edges = [round(i * L / k) for i in range(k + 1)]
        for i, nm in enumerate(names):
            tbl[int(nm.split(".")[0])][r.sequence[edges[i]:edges[i + 1]]].append(
                bool(r.rfd2_any_pass))
    return tbl, 100 * df.rfd2_any_pass.mean()


def _subs(seqs, ref):
    if not seqs:
        return 0
    M = as_matrix(list(seqs))
    return len({(j, M[i, j]) for i in range(M.shape[0])
                for j in range(M.shape[1]) if M[i, j] != ref[j]})


def locate(site_or_bb, arm):
    """-> (backbone stem, parents csv, recombinant scored csv) for one arm."""
    if arm == "fragrl":
        bb = site_or_bb
        return bb, (FRAGRL / "eval" / f"{bb}_frag" / f"{bb}_frag_af3scored.csv",
                    FRAGRL / "recomb" / f"{bb}_frag_scored.csv")
    ev = sorted(PANEL.glob(f"eval/run_{site_or_bb}*/*_af3scored.csv"))
    if not ev:
        return None, (None, None)
    bb = ev[0].parent.name[:-3]
    par = ev[0] if arm == "policy" else PANEL / f"{bb}_baseline" / "baseline_scored.csv"
    return bb, (par, PANEL / "recomb" / f"{bb}_{arm}_scored.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    pub = pd.read_csv(CLEO / "paper/figures/data/rfd2_AME_per_site.csv")

    sites = sorted({"_".join(p.name.replace("run_", "").split("_")[:2])
                    for p in PANEL.glob("eval/run_*")})
    rows, surv_store = [], defaultdict(dict)

    for site in sites:
        for arm in ARMS:
            key = site
            if arm == "fragrl":
                bb0, _ = locate(site, "policy")
                if bb0 is None:
                    continue
                key = bb0
            bb, (par_csv, rc_csv) = locate(key, arm)
            if bb is None or rc_csv is None or not Path(rc_csv).exists():
                continue
            n_par = np.nan
            if par_csv and Path(par_csv).exists():
                p = pd.read_csv(par_csv)
                n_par = int(p[p.rfd2_any_pass].sequence.nunique())
            tbl, ret = _frag_table(rc_csv)
            if tbl is None:
                continue
            nf = ns = 0
            for slot, frags in tbl.items():
                nf += len(frags)
                surv = [s for s, fl in frags.items() if any(fl)]
                ns += len(surv)
                surv_store[(site, slot)][arm] = (surv, list(frags))
            dead = pub.loc[pub.benchmark == site, "pct_backbones_0pass"]
            rows.append(dict(site=site, dead=dead.iloc[0] if len(dead) else np.nan,
                             arm=arm, parents=n_par, retention=ret,
                             surv_frac=ns / max(nf, 1)))

    # rarefied U over survivors, matched across whichever arms are present
    ur = defaultdict(float)
    for (site, slot), per_arm in surv_store.items():
        if len(per_arm) < 2:
            continue
        n = min(len(v[0]) for v in per_arm.values())
        if n < 2:
            continue
        for arm, (surv, allf) in per_arm.items():
            ref = np.array([max(set(c), key=list(c).count)
                            for c in as_matrix(allf).T], dtype="S1")
            ur[(site, arm)] += float(np.mean(
                [_subs(list(rng.choice(surv, n, replace=False)), ref)
                 for _ in range(a.draws)]))

    d = pd.DataFrame(rows)
    d["U_rare"] = [ur.get((r.site, r.arm), np.nan) for r in d.itertuples()]
    d = d.sort_values(["dead", "site", "arm"])
    pd.set_option("display.width", 250, "display.float_format", "{:.1f}".format)
    print("COMBINABILITY -- sampled vs RL-trained vs trained-for-recombination\n")
    print(d.to_string(index=False, header=[
        "site", "dead%", "arm", "parents", "retention%", "surv_frac", "U_rare"]))

    for metric in ("retention", "surv_frac", "U_rare"):
        p = d.pivot(index="site", columns="arm", values=metric)
        if "fragrl" not in p.columns:
            continue
        b = p.dropna(subset=["policy", "fragrl"])
        if not len(b):
            continue
        print(f"\n{metric}: fragrl > policy on {(b.fragrl > b.policy).sum()}/{len(b)}"
              f"   mean {b.fragrl.mean():.2f} vs {b.policy.mean():.2f}")
    print("\n(collapse check: surv_frac up while U_rare down means the policy "
          "converged rather than became modular)")


if __name__ == "__main__":
    main()
