#!/usr/bin/env python3
"""Re-process a library round from its raw plate reads.

`library1_kinetics.py` does this for library 1 against one folder. This is
the same pipeline generalised over a list of round directories, because
three Figure 3 panels need quantities that jq01's shipped tables do not
carry:

    the filter cascade   their files are `processed_filtered_*`, i.e.
                         already past the gates, so the counts that were
                         DROPPED are not in them
    replicate parity     their files are one row per construct, carrying
                         mean/std/cv but not the individual replicates
    a worked trace       the per-well time course is upstream of everything

All three are recoverable, because `plate_maps/` and `raw_data/` survive for
every round of library 3.

WHAT IS COMPUTED, per well:

    rate      least-squares slope of 4MU_um against real time
    bg        MEDIAN slope of the plate's negative-control wells. Median,
              not mean: on library 1 a single bad control well at 222.7
              against five at ~127 moved the derived G20/momi ratio from
              11.7 to 24.0.
    conc      mscarlett_um, averaged over timepoints
    reported  (rate - bg) / (conc + 1e-4), per replicate, then aggregated

OVERNIGHT READS ARE EXCLUDED. Several rounds carry a `30h_t4` file. That
read is ~10x further out in time than t0-t3 and the reaction has saturated
by then, so including it in the slope pulls every rate toward zero and
flattens exactly the differences the library exists to measure. Rounds are
fitted on their short series only; the filename tag is how they are told
apart.

    uv run python paper/figures/library_kinetics.py
"""

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics

sys.path.append("/projects/ml/itopt")
from neo2_util_updated_251122 import (get_least_squares_fit,  # noqa: E402
                                      get_mscarlett_and_kinetics)

HERE = Path(__file__).resolve().parent
JQ = Path("/home/jq01/projects/jq_p07_active_learning")
STANDARD = "/projects/ml/itopt/neo2_standards/241202_standard_curves_jake_mod.json"

EXPRESSION_MIN = 0.1       # uM, per replicate
MIN_REPLICATES = 2         # at least 2 of 3, the house rule
CV_MAX = 1.0               # sd reaches the mean = hypervariate

# Library 3, every round, as (round, directory holding plate_maps/ + raw_data/).
#
# Library 2's raw reads are NOT here -- its rounds predate this layout and
# only the processed tables survive in jq_e0319 -- so the cascade and the
# parity panel are library-3 (and library-1) quantities. Said plainly rather
# than quietly plotting two libraries where three are claimed.
ROUNDS = {
    0: JQ / "jq_e0322_260115_petase_g20_2nd_big_library_frag_swapping_analysis",
    1: JQ / "jq_e0323_260116_petase_g20_2nd_big_library_rd1_analysis",
    2: JQ / "jq_e0324_260119_petase_g20_2nd_big_library_rd2_analysis" / "1_exp_data_analysis" / "input",
    3: JQ / "jq_e0325_260123_petase_g20_2nd_big_library_rd3_analysis" / "1_exp_data_analysis" / "input",
    4: JQ / "jq_e0327_260128_petase_g20_2nd_big_library_rd4_analysis" / "1_exp_data_analysis" / "input",
}

OUT_DESIGNS = HERE / "data" / "library3_reprocessed_designs.csv"
OUT_REPS = HERE / "data" / "library3_reprocessed_replicates.csv"
OUT_TRACE = HERE / "data" / "library3_reprocessed_traces.csv"


def _int_after(s, key):
    """-> the integer following `key` in `s`, or None."""
    m = re.search(rf"{key}(\d+)", s)
    return int(m.group(1)) if m else None


def _timepoints(folder):
    """-> {timepoint: [raw files]}, overnight reads dropped.

    Keyed off `_t<N>` in the filename. A file whose name also carries an
    hour tag like `30h` is an overnight endpoint and is excluded -- see the
    module docstring.
    """
    out = {}
    for p in sorted(glob.glob(str(folder / "raw_data" / "*.txt"))):
        base = os.path.basename(p)
        m = re.search(r"_t(\d+)", base)
        if not m:
            continue
        if re.search(r"\d+h", base):
            continue
        out.setdefault(int(m.group(1)), []).append(p)
    return dict(sorted(out.items()))


def _map_for(raw, maps):
    """-> the one plate map matching this raw file's round/sample/plate."""
    base = os.path.basename(raw)
    want = {k: _int_after(base, k) for k in ("round", "sample", "plate")}
    hits = []
    for m in maps:
        mb = os.path.basename(m)
        if all(v is None or _int_after(mb, k) == v for k, v in want.items()):
            hits.append(m)
    if len(hits) != 1:
        raise RuntimeError(f"{base}: expected 1 plate map, got {len(hits)}")
    return hits[0]


def load_round(folder):
    """-> (mscarlett, kinetics) for one round, all timepoints."""
    maps = sorted(glob.glob(str(folder / "plate_maps" / "*.csv")))
    if not maps:
        raise RuntimeError(f"no plate maps under {folder}")
    msc, kin = [], []
    for t, raws in _timepoints(folder).items():
        mapping = [_map_for(r, maps) for r in raws]
        m_df, k_df = get_mscarlett_and_kinetics(mapping, raws, STANDARD)
        comp = []
        for _, g in k_df.groupby(["plate_number", "sample_number", "well"]):
            row = g.head(1).copy()
            row["4MU_um_mean"] = g["4MU_um"].mean()
            comp.append(row.drop(columns=["time", "value", "4MU_um"]))
        k_df = pd.concat(comp).reset_index(drop=True)
        k_df["timepoint"] = t
        m_df["timepoint"] = t
        msc.append(m_df)
        kin.append(k_df)
    if not kin:
        raise RuntimeError(f"no usable raw reads under {folder}")
    return (pd.concat(msc, ignore_index=True),
            pd.concat(kin, ignore_index=True))


def fit_wells(msc, kin):
    """-> one row per WELL: raw_rate, r2, concentration. The replicate unit."""
    out = []
    for (pl, sm), g in kin.groupby(["plate_number", "sample_number"]):
        mg = msc[(msc.plate_number == pl) & (msc.sample_number == sm)]
        for well, gw in g.groupby("well"):
            gw = gw.sort_values("timepoint")
            mw = mg[mg.well == well]
            xs = gw["datetime"].values
            ys = gw["4MU_um_mean"].values
            if len(xs) < 2:
                continue
            m, b = get_least_squares_fit(xs, ys)
            row = gw.drop(columns=["datetime", "timepoint", "4MU_um_mean",
                                   "source_file"], errors="ignore").head(1).copy()
            row["raw_rate"] = m
            row["r2_score"] = sklearn.metrics.r2_score(ys, m * xs + b)
            row["mscarlett_um"] = mw["mscarlett_um"].mean()
            out.append(row)
    return pd.concat(out).reset_index(drop=True)


def background(fits):
    """-> {(plate_number, sample_number): median neg-control slope}."""
    bg = {}
    for pl, g in fits.groupby(["plate_number", "sample_number"]):
        neg = g[g.sample_type == "negative_control"]
        good = neg[neg.r2_score > 0.9]
        use = good if len(good) else neg
        bg[pl] = float(use["raw_rate"].median()) if len(use) else np.nan
    return bg


def per_replicate(fits):
    """-> per-well normalised rate. One row per replicate, for the parity plot."""
    bg = background(fits)
    f = fits.copy()
    f["bg_rate_plate"] = [bg.get(k, float("nan"))
                          for k in zip(f.plate_number, f.sample_number)]
    f["norm_rate"] = ((f.raw_rate - f.bg_rate_plate)
                      / (f.mscarlett_um + 1e-4))
    return f


def per_design(reps):
    """-> one row per construct, with every gate recorded separately.

    The gates are kept as their own columns rather than collapsed into one
    pass/fail, because the cascade panel needs the count surviving EACH of
    them and a single boolean cannot be taken apart afterwards.
    """
    out = []
    # (plate, SAMPLE, well). sample_number is load-bearing: a round is run
    # as two plate sets (sample0 and sample1) that reuse plate numbers 1-3,
    # so grouping on plate and well alone silently merges two different
    # constructs that happen to share a well -- which collapsed round 0 from
    # ~1,900 designs to 375.
    for (pl, sm, well), g in reps.groupby(["plate_number", "sample_number",
                                           "source_dna_well"]):
        r = g.head(1).copy()
        conc, norm = g["mscarlett_um"], g["norm_rate"]
        r["n_replicates"] = len(g)
        r["n_expressing"] = int((conc >= EXPRESSION_MIN).sum())
        r["mscarlett_um_mean"] = conc.mean()
        r["bg_sub_conc_norm_rate_mean"] = norm.mean()
        r["bg_sub_conc_norm_rate_std"] = norm.std()
        r["bg_sub_conc_norm_rate_cv"] = norm.std() / (norm.mean() + 1e-4)
        r["r2_score"] = g["r2_score"].mean()
        out.append(r.drop(columns=["well", "raw_rate", "norm_rate",
                                   "mscarlett_um"], errors="ignore"))
    d = pd.concat(out).reset_index(drop=True)
    d["passes_expression"] = d.n_expressing >= MIN_REPLICATES
    d["passes_cv"] = d.bg_sub_conc_norm_rate_cv.abs() <= CV_MAX
    d["passes_filter"] = d.passes_expression & d.passes_cv
    return d


def normalise_to_g20(d):
    """-> `d` with g20_norm_rate, against the plate's positive controls.

    Per plate, as every other library here is normalised. Library 3's
    positive control IS G20, so the reference is the median of the plate's
    positive-control wells rather than a named source well.
    """
    ref = (d[d.sample_type == "positive_control"]
           .groupby(["plate_number", "sample_number"])
           .bg_sub_conc_norm_rate_mean.median())
    d = d.copy()
    d["g20_ref_plate"] = list(
        ref.reindex(list(zip(d.plate_number, d.sample_number))).to_numpy())
    d["g20_norm_rate"] = d.bg_sub_conc_norm_rate_mean / d.g20_ref_plate
    return d


def main():
    designs, reps, traces = [], [], []
    for rnd, folder in ROUNDS.items():
        try:
            msc, kin = load_round(folder)
        except Exception as e:                      # noqa: BLE001
            print(f"  round {rnd}: SKIPPED ({e})")
            continue
        fits = fit_wells(msc, kin)
        r = per_replicate(fits)
        d = normalise_to_g20(per_design(r))
        for frame, bucket in ((r, reps), (d, designs)):
            frame["round_number"] = rnd
            bucket.append(frame)

        # One round's worth of raw time course, for the worked-trace panel.
        k = kin.copy()
        k["round_number"] = rnd
        traces.append(k)
        print(f"  round {rnd}: {len(fits):5d} wells -> {len(d):5d} designs, "
              f"{int(d.passes_filter.sum()):5d} pass")

    if not designs:
        sys.exit("nothing processed")

    D = pd.concat(designs, ignore_index=True)
    R = pd.concat(reps, ignore_index=True)
    T = pd.concat(traces, ignore_index=True)
    OUT_DESIGNS.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT_DESIGNS, index=False)
    R.to_csv(OUT_REPS, index=False)
    T.to_csv(OUT_TRACE, index=False)
    print(f"\nwrote {OUT_DESIGNS.name}  {len(D)} designs")
    print(f"wrote {OUT_REPS.name}  {len(R)} replicate wells")
    print(f"wrote {OUT_TRACE.name}  {len(T)} well-timepoints")

    s = D[D.sample_type == "sample"]
    print("\ncascade, samples only:")
    print(f"  assayed wells     {len(R[R.sample_type == 'sample']):6d}")
    print(f"  constructs        {len(s):6d}")
    print(f"  expressing        {int(s.passes_expression.sum()):6d}")
    print(f"  + CV <= {CV_MAX}       {int(s.passes_filter.sum()):6d}")


if __name__ == "__main__":
    main()
