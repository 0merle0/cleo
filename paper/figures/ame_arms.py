#!/usr/bin/env python3
"""The three AME arms, collected the same way, for Figure 1(e) and 1(f).

    sampled    stock ProteinMPNN, T = 0.1 .. 1.0 pooled   r1panel/*_baseline
    plain RL   GRPO on whole sequences, RF3 only          r1rep/*_s2, *_s3
    chimera    GRPO on fragments, oracle pool             fragk/*_fragk4

Every arm is measured on RECOMBINANTS -- fragments cut at k=4, shuffled
across parents, reassembled and folded best-of-5 through AF3. Not on parents.
A policy can look good on parents and still produce a library that falls
apart when its fragments are mixed, which is the failure the whole project is
about.

AF3 is held out of all three training loops, so the number is comparable
across arms that trained differently. That is the only reason these three can
share an axis.

WHY plain RL IS TWO ROWS PER BACKBONE. s2 and s3 are replicates of one config
differing only in `seed`. They are what bounds a lucky draw, and they are the
only replicate evidence in the benchmark -- no pool arm has ever been run
twice. Keep both; collapsing them to a mean throws away the error bar and
leaves three single-seed numbers pretending to be a comparison.

A NOTE ON WHICH CHIMERA ARM. key_results.md section 6 quotes 96.9 / 82.4 /
100.0 for the chimera objective. That is `fragb32`, evaluated as a 256-parent
library (T2). This module uses `fragk4` instead -- a different arm, scored
through the same recomb pipeline as the other two here. 85.9 / 85.9 / 98.4.
The two are not interchangeable and the figure should quote the numbers this
module produces, because they are the ones computed identically to the arms
they are being compared against.

    uv run python paper/figures/ame_arms.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLEO = HERE.parents[1]
AME = CLEO / "experiments" / "ame"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(AME))

BB = {"M0097": "run_M0097_1ctt_cond9_14",
      "M0255": "run_M0255_1mg5_cond17_66",
      "M0315": "run_M0315_1ey3_cond18_25"}

# Arm -> list of (label, path template). A list because plain RL is two seeds.
ARMS = ("sampled", "plain RL", "chimera")

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}


def _files(site, arm):
    """-> [(seed, csv path)] for one arm of one backbone."""
    st = BB[site]
    if arm == "sampled":
        return [(None, AME / "r1panel" / "recomb" / f"{st}_baseline_scored.csv")]
    if arm == "plain RL":
        return [(s, AME / "r1rep" / "recomb" / f"{st}_{s}_scored.csv")
                for s in ("s2", "s3")]
    return [(None, AME / "fragk" / "recomb" / f"{st}_fragk4_scored.csv")]


def load(site, arm):
    """-> [(seed, dataframe)], skipping anything that is not on disk."""
    return [(s, pd.read_csv(f)) for s, f in _files(site, arm) if f.exists()]


def as_matrix(seqs):
    """-> (n, L) array of residue characters. Ragged input is an error."""
    lens = {len(s) for s in seqs}
    if len(lens) != 1:
        raise ValueError(f"ragged sequences: lengths {sorted(lens)}")
    return np.array([list(s) for s in seqs])


def onehot(seqs):
    """-> (n, L*20) float array. Unknown residues become all-zero columns."""
    M = as_matrix(seqs)
    n, L = M.shape
    out = np.zeros((n, L, 20), dtype=np.float32)
    for i in range(n):
        for j in range(L):
            k = AA_IDX.get(M[i, j])
            if k is not None:
                out[i, j, k] = 1.0
    return out.reshape(n, L * 20)


def _distinct_subs(seqs, ref):
    """-> count of distinct (position, residue) substitutions away from ref."""
    M = as_matrix(list(seqs))
    return len({(j, M[i, j]) for i in range(M.shape[0])
                for j in range(M.shape[1]) if M[i, j] != ref[j]})


def collect(draws=200, seed=0):
    """-> one row per (site, arm, seed): pass rate and rarefied diversity.

    DIVERSITY IS RAREFIED, and that is the point of care here. Distinct
    substitutions are counted over PASSING recombinants only -- diversity
    among members that fail is diversity you cannot order -- and an arm that
    passes more often has more survivors to find substitutions in. Counting
    raw would hand the better arm a diversity win for free. Every arm on a
    backbone is therefore subsampled to the same number of survivors before
    counting, so a difference means wider coverage and not a larger sample.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for site in BB:
        loaded = {a: load(site, a) for a in ARMS}
        if any(not v for v in loaded.values()):
            continue

        passing = {(a, s): d[d.rfd2_any_pass].drop_duplicates("sequence")
                   .sequence.tolist()
                   for a, lst in loaded.items() for s, d in lst}
        n = min(len(v) for v in passing.values())

        # ONE reference per backbone -- the column-wise consensus of every
        # arm's survivors pooled. A per-arm reference would measure each arm's
        # spread about its own centre, which is not comparable between arms.
        M = as_matrix([s for v in passing.values() for s in v])
        ref = np.array([max(set(c), key=list(c).count) for c in M.T])

        for a, lst in loaded.items():
            for s, d in lst:
                keep = passing[(a, s)]
                div = float(np.mean([
                    _distinct_subs(rng.choice(keep, n, replace=False), ref)
                    for _ in range(draws)])) if n else float("nan")
                rows.append(dict(site=site, arm=a, seed=s,
                                 n=len(d),
                                 pass_rate=100 * float(d.rfd2_any_pass.mean()),
                                 n_pass=len(keep), n_rare=n, subs=div))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = collect()
    print(d.to_string(index=False))
    print()
    print("pooled over backbones (plain RL pools its two seeds too):")
    print(d.groupby("arm").agg(pass_mean=("pass_rate", "mean"),
                               pass_min=("pass_rate", "min"),
                               pass_max=("pass_rate", "max"),
                               subs_mean=("subs", "mean"))
          .reindex(list(ARMS)).round(1).to_string())
