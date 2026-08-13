#!/usr/bin/env python3
"""E9: which selection rule should decide what we fold?

Every sequence in a finished GRPO run already carries a pass/fail label, so any
selection rule can be scored retrospectively without folding anything. Each rule
picks k sequences from the pool; we report how many of them passed, how many
distinct clusters they form, and how spread out they are.

The bar is *random*, not "more diverse than as-sampled". Diversity is trivial to
win and worthless alone -- a rule that doubles spread and halves passing designs
has made the library worse.

    uv run python paper/figures/ame_selection_bench.py --budgets 100,200,400

Anchor-based rules centre on the consensus of low-temperature LigandMPNN
samples: the solution the base model's likelihood concentrates on. `anchor_band`
selects a distance *shell* around it rather than maximising distance, which is
the specific correction for max-min walking off to unfoldable outliers.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from cleo.design.utils.selection import (  # noqa: E402
    RULES, Distances, consensus, encode,
)

AME = HERE.parents[1] / "experiments" / "ame"


def load_pool(backbone, run_root):
    rows = []
    for m in sorted((run_root / backbone / backbone / "outputs").glob("step_*/metrics.csv")):
        d = pd.read_csv(m)
        d["step"] = int(re.search(r"step_(\d+)", str(m)).group(1))
        rows.append(d)
    if not rows:
        return None
    d = pd.concat(rows, ignore_index=True)
    d["passing"] = d.ame_motif_pass & d.ame_no_clash
    return d.reset_index(drop=True)


def n_clusters(seqs, thresh=0.90):
    if len(seqs) < 2:
        return len(seqs)
    M = encode(seqs)
    L = M.shape[1]
    D = np.array([(M != M[i]).sum(1) for i in range(len(M))], float)
    Z = linkage(squareform((D + D.T) / 2, checks=False), "complete")
    return int(fcluster(Z, t=(1 - thresh) * L, criterion="distance").max())


def mean_pairwise(seqs, cap=300, seed=0):
    M = encode(seqs)
    if len(M) < 2:
        return float("nan")
    r = np.random.default_rng(seed).choice(len(M), min(len(M), cap), replace=False)
    D = np.array([(M[r] != M[i]).sum(1) for i in r], float)
    return D[np.triu_indices(len(r), 1)].mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default=AME / "runs9rmsd")
    ap.add_argument("--backbone", default="run_M0097_1ctt_cond9_14")
    ap.add_argument("--anchor-from", default=AME / "bestofn" / "repro_T0.1_seqs_bo5_scored.csv")
    ap.add_argument("--budgets", default="100,200,400,800")
    ap.add_argument("--reps", type=int, default=3, help="seeds per stochastic rule")
    ap.add_argument("--pool-cap", type=int, default=0,
                    help="subsample the pool for speed (0 = use all)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pool = load_pool(a.backbone, Path(a.run_root))
    if pool is None:
        raise SystemExit(f"no rollouts under {a.run_root}/{a.backbone}")
    if a.pool_cap and len(pool) > a.pool_cap:
        pool = pool.sample(a.pool_cap, random_state=0).reset_index(drop=True)
    seqs = pool.sequence.tolist()
    print(f"pool {len(seqs)}, {pool.passing.sum()} passing "
          f"({100 * pool.passing.mean():.1f}%)")

    anchor_df = pd.read_csv(a.anchor_from)
    key = a.backbone.replace("run_", "")
    if "backbone" in anchor_df:
        anchor_df = anchor_df[anchor_df.backbone.str.contains(key, regex=False)]
    D = Distances(seqs)
    anchor_d = D.to_seq(consensus(anchor_df.sequence))
    print(f"anchor: low-T consensus, pool distance "
          f"min={anchor_d.min():.0f} median={np.median(anchor_d):.0f} max={anchor_d.max():.0f}")

    rows = []
    for k in [int(x) for x in a.budgets.split(",")]:
        # as-sampled: the first k rollouts, i.e. what we actually folded.
        for name in ["as_sampled"] + list(RULES):
            reps = 1 if name in ("as_sampled", "anchor_far") else a.reps
            got = []
            for s in range(reps):
                if name == "as_sampled":
                    idx = np.arange(min(k, len(seqs)))
                else:
                    idx = RULES[name](D, k, seed=s, anchor_d=anchor_d)
                sub = pool.iloc[idx]
                p = sub[sub.passing].sequence.tolist()
                got.append((len(p), n_clusters(p), mean_pairwise(sub.sequence.tolist())))
            g = np.array(got, float)
            rows.append(dict(budget=k, rule=name, passing=g[:, 0].mean(),
                             clusters=g[:, 1].mean(), spread=g[:, 2].mean()))
            print(f"  k={k:5d} {name:15s} passing={g[:, 0].mean():6.1f} "
                  f"clusters={g[:, 1].mean():6.1f} spread={g[:, 2].mean():6.1f}")

    res = pd.DataFrame(rows)
    # The only comparison that matters: passing yield relative to random.
    piv = res.pivot(index="budget", columns="rule", values="passing")
    print("\npassing designs, as a ratio to random selection:")
    print((piv.div(piv["random"], axis=0)).round(2).to_string())
    if a.out:
        res.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
