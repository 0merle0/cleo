#!/usr/bin/env python3
"""Process library 1 the same way libraries 2 and 3 were processed.

Library 1 could not be plotted alongside the other two: it was scored on
TOTAL PRODUCT from the September PET-probe screen, where the reaction had
already saturated, while libraries 2 and 3 are scored on a background-
subtracted, concentration-normalised RATE. Different quantities on one axis.

The December re-test fixes that. It is the same library, the same plate
maps, read on 4MU with t0/t1/t2 files -- the identical structure libraries 2
and 3 use -- and it shows real accumulation against a flat background:

              t0     t1     t2
    G20       38 ->  68 -> 107     (+69, enzymatic)
    A1        33 ->  39 ->  45     (+12, weak)
    P24 neg   30 ->  35 ->  37     (+7,  background)

So this script runs the libraries-2-and-3 pipeline over it, via the shared
`neo2_util_updated_251122` that jq01's and kiera's notebooks both use, and
emits one row per design with `bg_sub_conc_norm_rate_mean` -- the same
column, computed the same way.

    rate      least-squares slope of 4MU_um over t0/t1/t2, per well
    bg        mean negative-control slope, PER PLATE
    conc      mscarlett_um, mean over timepoints
    reported  (rate - bg) / (conc + 1e-4), aggregated over replicate wells

    uv run python paper/figures/library1_kinetics.py
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics

sys.path.append("/projects/ml/itopt")
from neo2_util_updated_251122 import (get_least_squares_fit,  # noqa: E402
                                      get_mscarlett_and_kinetics)

HERE = Path(__file__).resolve().parent
FOLDER = Path("/home/jgershon/projects/itopt/kiera/testing_first_library_on_pet_probe")
STANDARD = "/projects/ml/itopt/neo2_standards/241202_standard_curves_jake_mod.json"
OUT = HERE / "data" / "library1_december_kinetics.csv"

EXPRESSION_MIN = 0.1       # uM, per replicate
MIN_REPLICATES = 2         # at least 2 of 3, the house rule


def next_int(s, key):
    tail = s.split(key)[1]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits)


def load():
    """-> (mscarlett_df, kinetics_df) across all timepoints."""
    msc, kin = [], []
    maps = glob.glob(str(FOLDER / "plate_maps" / "*.csv"))
    i = 0
    while True:
        raws = sorted(glob.glob(str(FOLDER / "raw_data" / f"*_t{i}*.txt")))
        if not raws:
            break
        mapping = []
        for p in raws:
            r, s, pl = (next_int(p, "round"), next_int(p, "sample"),
                        next_int(p, "plate"))
            hit = [m for m in maps if f"round{r}" in m and f"sample{s}" in m
                   and f"plate{pl}" in m]
            if len(hit) != 1:
                sys.exit(f"expected one plate map for {p}, got {hit}")
            mapping.append(hit[0])
        m_df, k_df = get_mscarlett_and_kinetics(mapping, raws, STANDARD)

        # one row per well: mean 4MU over the reads within this timepoint
        comp = []
        for (pl, sm, well), g in k_df.groupby(["plate_number", "sample_number",
                                               "well"]):
            row = g.head(1).copy()
            row["4MU_um_mean"] = g["4MU_um"].mean()
            comp.append(row.drop(columns=["time", "value", "4MU_um"]))
        k_df = pd.concat(comp).reset_index(drop=True)
        k_df["timepoint"] = i
        m_df["timepoint"] = i
        msc.append(m_df)
        kin.append(k_df)
        i += 1
    if not kin:
        sys.exit(f"no raw data under {FOLDER/'raw_data'}")
    return pd.concat(msc, ignore_index=True), pd.concat(kin, ignore_index=True)


def fit(msc, kin):
    """-> one row per well with raw_rate, r2 and mscarlett_um_mean."""
    out = []
    for (pl, sm), g in kin.groupby(["plate_number", "sample_number"]):
        mg = msc[(msc.plate_number == pl) & (msc.sample_number == sm)]
        for well, gw in g.groupby("well"):
            gw = gw.sort_values("timepoint")
            mw = mg[mg.well == well].sort_values("timepoint")
            xs = gw["datetime"].values
            ys = gw["4MU_um_mean"].values
            if len(xs) < 2:
                continue
            m, b = get_least_squares_fit(xs, ys)
            row = gw.drop(columns=["datetime", "timepoint", "4MU_um_mean",
                                   "source_file"]).head(1).copy()
            row["raw_rate"] = m
            row["r2_score"] = sklearn.metrics.r2_score(ys, m * xs + b)
            row["mscarlett_um_mean"] = mw["mscarlett_um"].mean()
            out.append(row)
    return pd.concat(out).reset_index(drop=True)


def aggregate(fits):
    """-> one row per source well, background-subtracted PER PLATE."""
    out = []
    for pl, g in fits.groupby("plate_number"):
        bg = g[(g.sample_type == "negative_control") & (g.r2_score > 0.9)]
        if not len(bg):
            bg = g[g.sample_type == "negative_control"]
        # median, not mean: one bad control well should not set the floor
        bg_rate = bg["raw_rate"].median()
        for well, gw in g.groupby("source_dna_well"):
            r = gw.head(1).copy()
            conc = gw["mscarlett_um_mean"]
            norm = (gw["raw_rate"] - bg_rate) / (conc + 1e-4)
            r["bg_rate_plate"] = bg_rate
            r["n_replicates"] = len(gw)
            r["n_expressing"] = int((conc >= EXPRESSION_MIN).sum())
            r["mscarlett_um_mean"] = conc.mean()
            r["bg_sub_conc_norm_rate_mean"] = norm.mean()
            r["bg_sub_conc_norm_rate_std"] = norm.std()
            r["bg_sub_conc_norm_rate_cv"] = (norm.std()
                                             / (norm.mean() + 1e-4))
            r["r2_score"] = gw["r2_score"].mean()
            out.append(r.drop(columns=["raw_rate", "well"], errors="ignore"))
    return pd.concat(out).reset_index(drop=True)


def main():
    msc, kin = load()
    fits = fit(msc, kin)
    agg = aggregate(fits)

    # Normalise to the G20 WELL, per plate. Library 1 predates G20 being a
    # control, so its positive control is momi WT -- which cannot serve as a
    # denominator, sitting barely above background at ~0.1 uM expression.
    g20 = agg[agg.source_dna_well == "G20"]
    if not len(g20):
        sys.exit("no G20 well found; cannot normalise")
    refs = g20.groupby("plate_number").bg_sub_conc_norm_rate_mean.mean()
    agg["g20_norm_rate"] = (agg.bg_sub_conc_norm_rate_mean
                            / agg.plate_number.map(refs))
    agg["passes_expression"] = agg.n_expressing >= MIN_REPLICATES
    agg["passes_cv"] = agg.bg_sub_conc_norm_rate_cv.abs() <= 1.0
    agg["passes_filter"] = agg.passes_expression & agg.passes_cv

    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(agg)} designs, "
          f"{int(agg.passes_filter.sum())} pass the filter")

    k = agg[agg.passes_filter & (agg.sample_type == "sample")]
    print(f"\nG20 reference per plate: "
          f"{refs.round(6).to_dict()}")
    print(f"background rate per plate: "
          f"{agg.groupby('plate_number').bg_rate_plate.first().round(6).to_dict()}")
    print(f"\nfiltered samples: n={len(k)}  median {k.g20_norm_rate.median():.3f} "
          f"x G20   max {k.g20_norm_rate.max():.3f}")
    print("\ntop 8:")
    print(k.nlargest(8, "g20_norm_rate")[
        ["plate_number", "source_dna_well", "mscarlett_um_mean",
         "bg_sub_conc_norm_rate_mean", "g20_norm_rate",
         "bg_sub_conc_norm_rate_cv"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
