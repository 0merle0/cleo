#!/usr/bin/env python3
"""Score individual fragments by the recombinants they appear in.

A library is built from fragments, not from sequences, so the useful question
is which *fragments* are worth synthesising. Every recombinant records the four
fragments it was assembled from, and whether it folded, so each fragment
inherits a pass rate from the chimeras carrying it.

Per fragment:
  n           recombinants containing it
  pass_rate   fraction of those that passed -- how well it plays with others
  n_mut       distinct substitutions it carries vs the slot consensus -- what
              it contributes to library diversity

The two together are the library-design tradeoff in miniature. A fragment that
is highly mutated but drags its chimeras below the cutoff costs more than it
adds; one that passes everywhere but carries no substitutions adds nothing to
explore. What a library wants is fragments high on both, and whether a policy
produces them is a property pass-rate comparisons cannot see.

    uv run python paper/figures/ame_fragment_scores.py [--site M0255]
"""

import argparse
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


def slot_consensus(seqs):
    M = as_matrix(list(seqs))
    return np.array([max(set(c), key=list(c).count) for c in M.T], dtype="S1")


def score_fragments(scored_csv):
    """-> DataFrame, one row per fragment, with pass rate and mutation count."""
    df = pd.read_csv(scored_csv)
    if "parent_fragments" not in df.columns:
        src = Path(str(scored_csv).replace("_scored.csv", ".csv"))
        df = df.merge(pd.read_csv(src)[["name", "parent_fragments"]], on="name", how="left")
    df = df.dropna(subset=["parent_fragments"])

    # fragment name -> [pass flags]
    hits = defaultdict(list)
    # fragment name -> its sequence, recovered from the recombinant it sits in
    seqs = {}
    for _, r in df.iterrows():
        names = str(r.parent_fragments).split("___")
        k = len(names)
        L = len(r.sequence)
        edges = [round(i * L / k) for i in range(k + 1)]
        for i, nm in enumerate(names):
            hits[nm].append(bool(r.rfd2_any_pass))
            seqs[nm] = r.sequence[edges[i]:edges[i + 1]]

    # consensus per slot, so "mutation" means "differs from what this slot
    # usually looks like" rather than from an arbitrary reference
    by_slot = defaultdict(list)
    for nm, s in seqs.items():
        by_slot[nm.split(".")[0]].append(s)
    cons = {slot: slot_consensus(v) for slot, v in by_slot.items() if v}

    rows = []
    for nm, flags in hits.items():
        slot = nm.split(".")[0]
        s = seqs[nm]
        c = cons[slot]
        n_mut = sum(1 for a, b in zip(s.encode(), c) if bytes([a]) != b)
        rows.append(dict(fragment=nm, slot=int(slot), n=len(flags),
                         pass_rate=float(np.mean(flags)), n_mut=n_mut))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=None, help="limit to one site prefix")
    ap.add_argument("--min-n", type=int, default=3,
                    help="fragments seen in fewer recombinants have noisy pass rates")
    a = ap.parse_args()

    out = []
    for f in sorted(PANEL.glob("recomb/*_scored.csv")):
        if "_samples" in f.name:
            continue
        b = f.name.replace("_scored.csv", "")
        arm = "policy" if b.endswith("_policy") else "baseline"
        bb = b[: -len(f"_{arm}")]
        site = "_".join(bb.replace("run_", "").split("_")[:2])
        if a.site and not site.startswith(a.site):
            continue
        d = score_fragments(f)
        d = d[d.n >= a.min_n]
        if not len(d):
            continue
        out.append(dict(site=site, arm=arm, n_frag=len(d),
                        mean_pass=d.pass_rate.mean(), mean_mut=d.n_mut.mean(),
                        # fragments that are both mutated and reliable: the
                        # ones a library actually wants
                        good=int(((d.pass_rate >= 0.8) & (d.n_mut >= d.n_mut.median())).sum()),
                        corr=d.pass_rate.corr(d.n_mut)))
    r = pd.DataFrame(out)
    pd.set_option("display.width", 250, "display.float_format", "{:.3f}".format)
    print("FRAGMENT-LEVEL SCORES  (fragments seen in >= %d recombinants)\n" % a.min_n)
    print(r.sort_values(["site", "arm"]).to_string(index=False, header=[
        "site", "arm", "n_frag", "mean_pass", "mean_mut", "good_frags", "corr(pass,mut)"]))
    if len(r) and r.arm.nunique() == 2:
        p = r.pivot(index="site", columns="arm", values="mean_pass")
        m = r.pivot(index="site", columns="arm", values="mean_mut")
        print("\nmean fragment pass rate   policy vs baseline")
        print(p.to_string())
        print("\nmean substitutions per fragment")
        print(m.to_string())


if __name__ == "__main__":
    main()
