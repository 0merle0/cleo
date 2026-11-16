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

import re

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

CONFIG = HERE.parents[1] / "config" / "design" / "denovo_petase.yaml"


def anchors():
    """-> the reference sequences the PETase run actually anchored to.

    READ FROM THE CONFIG, not reconstructed. `denovo_petase.yaml` carries
    three sequences under the `dist_to_ref_seqs` step, and the reward
    minimises distance to the nearest of them at weight 10 -- the heaviest
    term in the whole aggregation. So these three ARE the design target, and
    anything else is a guess at it.

    This matters because the obvious guess is wrong. All three are exact
    members of library 2, but they are ranks 1, 4 and 21 of 3,371 by
    measured activity (3.35, 2.72 and 2.31 x G20), and the third comes from
    round 3. Taking "the top 3 by activity" instead silently substitutes
    ranks 1-3 and moves the target. The three sit 4-7 mutations apart, so
    they are a tight cluster rather than three separate corners.
    """
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text())
    for st in cfg["reward"]["steps"]:
        if st["name"] == "dist_to_ref_seqs":
            return list(st["cfg"]["ref_seqs"])
    raise KeyError(f"no dist_to_ref_seqs step in {CONFIG}")


def anchor_table():
    """-> where each anchor sits in library 2, by measured activity."""
    import numpy as np
    d = pd.read_csv(LIB)
    g = d[(d.library_name == LIB2) & (d.sample_type == "sample")]
    g = (g.dropna(subset=["g20_norm_rate", "sequence"])
         .sort_values("g20_norm_rate", ascending=False)
         .drop_duplicates("sequence").reset_index(drop=True))
    g["rank"] = np.arange(1, len(g) + 1)
    rows = []
    for i, r in enumerate(anchors(), 1):
        hit = g[g.sequence == r]
        rows.append(dict(anchor=f"ref{i}",
                         rank=int(hit["rank"].iloc[0]) if len(hit) else None,
                         of=len(g),
                         activity=float(hit.g20_norm_rate.iloc[0]) if len(hit) else None,
                         round=int(hit.round_number.iloc[0]) if len(hit) else None))
    return pd.DataFrame(rows)


def distance_to_anchors():
    """-> per-design minimum Hamming distance to the config's anchor set.

    Both libraries are on the same 209-aa scaffold with no indels, so Hamming
    is the honest distance and no alignment is needed. Verified rather than
    assumed: a ragged length raises.
    """
    d = pd.read_csv(LIB)
    refs = anchors()
    R = np.array([list(s) for s in refs])

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

    d, refs = distance_to_anchors()
    print("=== minimum Hamming to the config's three anchor sequences ===")
    print(d.groupby("library").dmin
          .describe()[["count", "min", "25%", "50%", "75%", "max"]]
          .to_string(float_format=lambda v: f"{v:.0f}"))
    print("\nthe anchors, and where they sit in library 2:")
    print(anchor_table().to_string(index=False))


if __name__ == "__main__":
    main()


# The five fragment slots, 1-indexed inclusive residue ranges. Same split as
# fig3_momi uses, so the per-slot panels in Figures 2 and 3 are cut at the
# same places and can be read against each other.
SLOTS = [(1, 40), (45, 84), (89, 126), (131, 163), (168, 209)]


def slot_distance_to_anchors():
    """-> long-form per-slot distance from each design's part to the anchors.

    Per slot rather than per design, because that is the unit the library is
    ordered in: a slot is a well of DNA parts, and "how far did we let this
    slot roam" is a decision that was made five times, once per slot, not
    once for the design.

    MINIMUM over the three anchors is taken PER SLOT, not per design. A
    design can take its slot-1 part from near anchor 1 and its slot-4 part
    from near anchor 3 -- that is what recombination across a tethered set
    means -- so a per-design minimum would misattribute both.

    Residues between the slots (the 4-residue junctions fixed for Golden
    Gate overhangs) are not in any slot and are excluded, which is why the
    slot lengths sum to 193 rather than 209.
    """
    d = pd.read_csv(LIB).dropna(subset=["sequence"])
    R = np.array([list(s) for s in anchors()])

    rows = []
    for lib in (LIB2, LIB3):
        g = d[(d.library_name == lib) & (d.sample_type == "sample")]
        g = g.drop_duplicates("sequence")
        S = np.array([list(s) for s in g.sequence])
        if S.shape[1] != R.shape[1]:
            raise ValueError(f"{lib}: {S.shape[1]} vs anchors {R.shape[1]}")
        for i, (lo, hi) in enumerate(SLOTS, start=1):
            a, b = lo - 1, hi                      # to 0-based, end-exclusive
            dist = (S[:, None, a:b] != R[None, :, a:b]).sum(axis=2)
            rows.append(pd.DataFrame(dict(library=lib, slot=i,
                                          dmin=dist.min(axis=1))))
    return pd.concat(rows, ignore_index=True)


def part_menu(library=None):
    """-> the distinct PARTS each slot offers, by their distN tag.

    This is what library 3's slots were designed to be, read from the part
    names, which carry the distance each part was generated at from the
    corresponding part of the anchor set. The tags are the design record and
    are taken as authoritative here.

    PARTS, not designs. A part appearing in 200 assembled designs is still
    one part in the slot's menu, and the question this answers is what the
    menu looked like -- the same framing clean.pdf uses for its per-fragment
    panel. Counting designs instead would reweight the menu by how often the
    sampler happened to draw each part.

    The one untagged part per slot is `N.g20`, the parent's own fragment. It
    is excluded: it is the thing the distances are measured from the
    anchors' versions of, not a member of the generated menu.

    WHY THE TAGS AND NOT THE SEQUENCES. Recomputing distance from the
    sequences needs the slot boundaries, and two of the five in `SLOTS` are
    wrong. Correlating computed distance against the tag over library 3
    gives r = 0.96 and 0.99 for slots 2 and 3 but 0.09 for slot 1 and 0.41
    for slot 5. Scanning boundaries recovers r = 0.93 at residues 7-40 for
    slot 1 and r = 0.99 at 164-209 for slot 5, against the (1, 40) and
    (168, 209) that `SLOTS` and `fig3_momi` both use. The tags avoid the
    question entirely; the boundary discrepancy is recorded here because
    fig3_momi's per-slot panel still relies on the old numbers.
    """
    library = library or LIB3
    d = pd.read_csv(LIB).dropna(subset=["name"])
    g = d[(d.library_name == library) & (d.sample_type == "sample")]

    seen = {i: {} for i in range(1, 6)}
    for n in g.name.astype(str):
        for i, part in enumerate(re.sub(r":.*$", "", n).split("___"), start=1):
            if i in seen:
                seen[i][part] = True

    rows = []
    for i, parts in seen.items():
        for part in parts:
            m = re.search(r"dist(\d+)", part)
            if m:
                rows.append(dict(slot=i, part=part, dist=int(m.group(1))))
    return pd.DataFrame(rows)
