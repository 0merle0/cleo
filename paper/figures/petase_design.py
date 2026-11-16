#!/usr/bin/env python3
"""The PETase design side: what post-training did, and how library 3 was aimed.

Two things, both read from the runs and libraries the paper actually shipped.

1. TRAINING TRAJECTORIES. `new_petase_seq_dist_w5` and `_w8` are the runs that
   produced the ordered library -- library 2's part names read
   `refw5.step0200.*` and `refw8.*`, so these are not a stand-in for the
   training, they ARE it. Their `*_train_metrics.csv` carries batch
   mean/min/max for every reward term at every step.

   **Step 0 is naive ProteinMPNN.** The policy is initialised from the stock
   decoder, so the step-0 batch is a stock sample scored on exactly the
   reward the run then optimises. That removes the need for a separate
   baseline arm and makes the naive comparison exact rather than matched.

   The band is batch MIN-MAX, not a confidence interval. It is the spread of
   one rollout, and its narrowing is the "reduces variance" claim measured.

2. DISTANCE TO THE TOP-5. Library 3 was designed around the five best
   variants measured in library 2 -- its part names carry `dist0.topk1`
   through `dist0.topk5`, and `distN` tags for parts N mutations away from
   them. So the quantity that says whether the aiming worked is each design's
   distance to that SET, not to G20 alone.

   Minimum Hamming over the five references, not the mean: a design built
   around reference 3 is close to the library's target even if it is far from
   references 1, 2, 4 and 5, and a mean would score it as distant.

WHAT IS NOT HERE, AND WHY. Per-design naive-vs-post-trained distributions --
strip plots of each metric -- are not recoverable. The training log stores
only batch mean/min/max per step, and the folds those numbers came from have
been cleaned: 62 CIFs survive across all 212 steps of w5, and
`momi_final_redesign` has no vanilla arm. Building that panel means
regenerating both arms from the checkpoints, which do survive
(`*_step_0200.pt`, plus the stock decoder as naive): sample ~250 sequences
each, fold best-of-5, score. That is a real job, not a read.

    uv run python paper/figures/petase_design.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RUNS = Path("/home/jgershon/projects/itopt/policy_mpnn")
# The two runs whose checkpoints library 2's parts are named after.
TRAIN_RUNS = {"refw5": "new_petase_seq_dist_w5",
              "refw8": "new_petase_seq_dist_w8"}

# Reward terms, in the order they are worth reading: confidence, then the
# catalytic geometry, then the tether to the reference sequences. Bounds and
# direction are copied from the run config so the panel shows the window the
# reward actually normalised over rather than a window chosen for the plot.
TERMS = [
    ("af3_iptm", "AF3 ipTM", 0.0, 0.8, "max"),
    ("af3_ptm", "AF3 pTM", 0.0, 0.8, "max"),
    ("petase_metrics_hisNE2_serOG", "His NE2 - Ser OG", 2.5, 10.0, "min"),
    ("petase_metrics_hisNE2_esterox", "His NE2 - ester O", 3.5, 10.0, "min"),
    ("petase_metrics_acylox_oxh1bbN", "Oxyanion 1", 2.5, 10.0, "min"),
    ("petase_metrics_acylox_oxh2bbN", "Oxyanion 2", 2.5, 10.0, "min"),
    ("petase_metrics_his_ser_angle", "His-Ser angle", 60, 120, "avg"),
    ("dist_to_ref_seqs_min", "Distance to reference", 0, 100, "min"),
]

LIB = HERE / "data" / "momi_libraries.csv"
LIB2, LIB3 = "first_big_library", "second_big_library"
TOP_K = 5


def load_training(run="refw5"):
    """-> the training metrics for one of the runs that made library 2."""
    stem = TRAIN_RUNS[run]
    f = RUNS / stem / f"{stem}_train_metrics.csv"
    if not f.exists():
        raise FileNotFoundError(f)
    return pd.read_csv(f)


def training_summary(run="refw5", final=20):
    """-> per-term step-0 value against the mean of the last `final` steps.

    The post-trained number is averaged over a window rather than read off
    the last step: one rollout of 16 is a small sample and the last step is
    no more special than the twenty before it.
    """
    d = load_training(run)
    rows = []
    for key, label, lo, hi, mode in TERMS:
        mcol, locol, hicol = (f"{key}_batch_mean", f"{key}_batch_min",
                              f"{key}_batch_max")
        if mcol not in d:
            continue
        naive = float(d[mcol].iloc[0])
        post = float(d[mcol].tail(final).mean())
        # Not every term logs a band. `dist_to_ref_seqs_min` is already a
        # per-batch reduction, so only its mean is written; the term that
        # carries min/max is `dist_to_ref_seqs_avg`. Report NaN rather than
        # substituting a different metric's spread.
        has_band = locol in d.columns and hicol in d.columns
        spread0 = float(d[hicol].iloc[0] - d[locol].iloc[0]) if has_band else np.nan
        spread1 = (float((d[hicol] - d[locol]).tail(final).mean())
                   if has_band else np.nan)
        rows.append(dict(run=run, metric=key, label=label, mode=mode,
                         lower=lo, upper=hi, naive=naive, post=post,
                         spread_naive=spread0, spread_post=spread1))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ top-k

def _top_refs(d, k=TOP_K):
    """-> the k best-measured library-2 sequences, as the aiming target.

    Deduplicated by sequence first. The same variant appears more than once
    across rounds, and a target set with a duplicate in it is a target set of
    k-1 distinct references.
    """
    g = d[(d.library_name == LIB2) & (d.sample_type == "sample")]
    g = g.dropna(subset=["g20_norm_rate", "sequence"])
    g = g.sort_values("g20_norm_rate", ascending=False).drop_duplicates("sequence")
    return g.head(k)


def distance_to_topk(k=TOP_K):
    """-> per-design minimum Hamming distance to the top-k library-2 set.

    Both libraries are on the same 209-aa scaffold with no indels, so Hamming
    is the honest distance and no alignment is needed. Verified rather than
    assumed: a ragged length raises.
    """
    d = pd.read_csv(LIB)
    refs = _top_refs(d, k)
    R = np.array([list(s) for s in refs.sequence])

    out = []
    for lib in (LIB2, LIB3):
        g = d[(d.library_name == lib) & (d.sample_type == "sample")]
        g = g.dropna(subset=["sequence"]).drop_duplicates("sequence")
        lens = {len(s) for s in g.sequence} | {R.shape[1]}
        if len(lens) != 1:
            raise ValueError(f"ragged sequences in {lib}: {sorted(lens)}")
        S = np.array([list(s) for s in g.sequence])
        # (n_designs, k) Hamming, then the minimum over the references.
        dist = (S[:, None, :] != R[None, :, :]).sum(axis=2)
        out.append(pd.DataFrame(dict(library=lib, dmin=dist.min(axis=1),
                                     dmean=dist.mean(axis=1))))
    return pd.concat(out, ignore_index=True), refs


def main():
    for run in TRAIN_RUNS:
        try:
            s = training_summary(run)
        except FileNotFoundError as e:
            print(f"{run}: missing {e}")
            continue
        print(f"=== {run} ({TRAIN_RUNS[run]}) : step 0 -> last 20 steps ===")
        print(s[["label", "mode", "naive", "post", "spread_naive",
                 "spread_post"]]
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print()

    d, refs = distance_to_topk()
    print(f"=== minimum Hamming to the top-{TOP_K} library-2 variants ===")
    print(d.groupby("library").dmin
          .describe()[["count", "min", "25%", "50%", "75%", "max"]]
          .to_string(float_format=lambda v: f"{v:.0f}"))
    print("\nthe top-5 reference variants (library 2, by activity):")
    for i, (_, r) in enumerate(refs.iterrows(), 1):
        print(f"  topk{i}  {r.g20_norm_rate:6.2f} x G20   {r['name'][:72]}")


if __name__ == "__main__":
    main()
