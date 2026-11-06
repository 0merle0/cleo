"""Shared loaders for the two diversity figures (frontier, novelty).

Both figures answer questions about the *same* two quantities -- how often a
policy's designs pass, and how much of sequence space they cover -- so they
must compute them the same way. The rules that matter:

  * Diversity is measured over **passing, deduplicated** designs only. A policy
    that emits 96 copies of one failing sequence has no diversity to report.

  * The reference used to define a substitution is the per-backbone consensus
    over *every* sequence folded on that backbone, baseline and CLEO pooled.
    Using each arm's own consensus would let an arm look un-mutated simply by
    being internally consistent.

  * Raw U (distinct substitutions) grows with how many designs passed, so it is
    not an independent axis to plot pass rate against. Use U_k, which rarefies
    to a fixed design count. tempsweep T=1.0 is the worked example: U collapses
    to 355 because only 4 designs survived, while those 4 are the most mutually
    distant set in the sweep (mean pairwise Hamming 126, the highest anywhere).
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AME = HERE.parents[1] / "experiments" / "ame"
sys.path.insert(0, str(AME))
from analyze_selection2 import as_matrix, coverage, mean_pairwise  # noqa: E402,F401

BACKBONES = ["run_M0097_1ctt_cond9_14",
             "run_M0904_1qgx_cond39_95",
             "run_M0907_1rbl_cond40_74"]
ARMS = ["random", "logprob_strat", "logprob_band", "logprob_top"]


def _cat(pattern):
    fs = sorted(glob.glob(str(pattern)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def baseline(backbone):
    """LigandMPNN temperature sweep, one frame with a `temperature` column."""
    d = _cat(AME / "tempsweep" / "*_scored.csv")
    return d[d.backbone == backbone].reset_index(drop=True)


def cleo(backbone, arm=None):
    """CLEO evaluation on `backbone`; `arm=None` pools every selection arm.

    Reads backbone19, the sweep that covers all three backbones with the same
    code path. selection2 also holds M0097 but disagrees with it by 35 points
    on logprob_top -- that discrepancy is unresolved, so mixing the two sources
    in one figure would average over a known inconsistency.
    """
    pat = AME / "backbone19" / "eval" / backbone / (f"{arm}_scored.csv" if arm else "*_scored.csv")
    return _cat(pat)


def reference(backbone):
    """Per-position consensus over every folded sequence for this backbone."""
    frames = [f for f in (baseline(backbone), cleo(backbone)) if f is not None]
    M = as_matrix(pd.concat(frames, ignore_index=True).sequence.tolist())
    return np.array([max(set(c), key=list(c).count) for c in M.T], dtype="S1")


def passing(df):
    """Passing, deduplicated designs."""
    return df[df.rfd2_any_pass].drop_duplicates("sequence")


def stats(df, ref, k=15):
    """-> dict(n, n_pass, pass_pct, U, Uk, hamming) for one group of designs."""
    keep = passing(df)
    M = as_matrix(keep.sequence.tolist()) if len(keep) else np.empty((0, len(ref)), "S1")
    U, Uk = coverage(M, ref, k) if len(keep) else (0, 0)
    return dict(n=len(df), n_pass=len(keep), pass_pct=100 * df.rfd2_any_pass.sum() / len(df),
                U=U, Uk=Uk, hamming=mean_pairwise(M) if len(keep) else 0.0)


def subs(df, ref):
    """Set of (position, residue) substitutions vs `ref`, over passing designs."""
    keep = passing(df)
    if not len(keep):
        return set()
    M = as_matrix(keep.sequence.tolist())
    return {(j, M[i, j]) for i in range(M.shape[0])
            for j in range(M.shape[1]) if M[i, j] != ref[j]}
