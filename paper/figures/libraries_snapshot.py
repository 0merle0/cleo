#!/usr/bin/env python3
"""Snapshot all THREE PETase libraries into clean, comparable CSVs.

The paper compares three libraries. They live in three different trees, were
processed by two different pipelines, and name their columns differently, so
this puts them on one schema and applies one filter.

    library 1  the ORIGINAL MoMI library -- where G20 was discovered
    library 2  the G20-optimisation campaign   (a.k.a. first_big_library)
    library 3  the experiment-guided library   (a.k.a. second_big_library)

Libraries 2 and 3 are G20's descendants: G20 is their parent and their
positive control, which is why both report `g20_norm_rate` directly.

LIBRARY 1 IS NORMALISED HERE, not at source. It predates G20 being a
control, and G20 sits in it as an ordinary sample (plate 3, well G20). So
`g20_norm_rate` for library 1 is computed by dividing through that well,
putting all three libraries on the same "x G20" axis.

THE EXPRESSION FILTER IS THE POINT. Rate divided by concentration explodes
when concentration is near zero: library 1 unfiltered is topped by
`rate_per_um = 622449` at `expression_um = 0.000`. Requiring >= 0.1 uM
mscarlett -- the project's standard threshold -- removes exactly those, and
is what makes G20 read as the hit it is:

    library 1, Sept, by total product   unfiltered rank 310/372
                                        filtered   rank   1/67

WHICH READ LIBRARY 1 USES, and why RATE is not its metric. The September
PET-probe screen has two timepoints, and `timepoint_2` is used.

  timepoint_1 is CEILING-SATURATED and cannot rank anything: total product
  spans 185-229 across the whole plate (median 212) and the top designs sit
  at the 1e-8 rate floor.

  timepoint_2 discriminates: 115-471, median 154, and G20 is 471 against
  283 for the next design -- a 1.7x gap.

Library 1 is therefore scored on TOTAL PRODUCT, not on rate. G20's fitted
rate is ~0 and its replicate slopes are slightly negative, because the
reaction had already plateaued by the read -- the signature of the most
active design in a saturating assay, not of a dead one. Its replicates are
tight where it matters: 4MU 467/500/446, mscarlett 0.268/0.279/0.288.

SAME PROBE, SAME WAVELENGTHS -- checked in the raw reader files, not
assumed. All three read mScarlet at Ex 536/20 / Em 592/20 and the product
at Ex 310/20 / Em 430/20 on the same Synergy Neo2. So "x G20" is a fair
axis across libraries and the comparison is not cross-assay.

What DOES differ is the gain and the read length on the product channel:

    library 1   gain 80    21 min, 20 reads
    library 2   gain 50     3 min,  3 reads
    library 3   gain 50     3 min,  3 reads

That is why library 1 saturates: 1.6x the gain over 7x the time. Its
reaction had plateaued by the read, so its fitted RATE is ~0 with slightly
negative replicate slopes while its TOTAL PRODUCT is the highest in the
library. Library 1 is therefore ranked on total product; `rate_per_um` is
computed the same way as libraries 2 and 3 for comparability but is not
what library 1 is ranked on, and plotting it without reading this will
mislead.

    uv run python paper/figures/libraries_snapshot.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"

KIERA = Path("/home/jgershon/projects/itopt/kiera")
SHARED = Path("/projects/ml/itopt/petase")

LIB1_SEPT = KIERA / "testing_first_library_on_pet_probe" / "250927"
LIB23 = KIERA / "exp_guided_library" / "260128_all_data_lib1_lib2.csv"

# THE HOUSE FILTER, taken from jq01's processing rather than invented here
# (jq_p07_active_learning/jq_e0321_.../2_process_kinetics.ipynb):
#
#     mscarlett_num_pass >= 2   AND   bg_sub_conc_norm_rate_cv <= cv_threshold
#
# `mscarlett_num_pass` is a COUNT of replicates expressing above 0.1 uM, not
# a boolean -- at least 2 of 3 must clear it. Filtering on the MEAN instead
# is slightly laxer: on library 1 it passes 68 designs against 65, letting
# through wells where one strong replicate carries two weak ones.
EXPRESSION_MIN = 0.1      # uM mscarlett, per replicate
MIN_REPLICATES_PASSING = 2
# CV thresholds are PER LIBRARY, because the quantity each library is ranked
# on differs and so does the CV that qualifies it.
#
#   libraries 2, 3   rate CV <= 1.0, jq01's threshold. A no-op by
#                    construction: they were CV-filtered upstream and their
#                    maxima are 0.281 and 0.444.
#   library 1        TOTAL-PRODUCT CV <= 0.15, chosen from the data here.
#
# Why total product: library 1 is ranked on total product, so that is the
# quantity whose reproducibility matters. Its CV is well behaved -- median
# 0.072, max 0.242 -- unlike the rate CV, which saturation drives to a
# median of 0.956 and would delete G20.
#
# Why 0.15: it removes 7 of 64 designs, and all 7 sit at or below
# background (-0.056 to 0.053 x G20, two of them negative) -- noise-
# dominated wells and nothing else. The top 5 are unchanged at 0.15, 0.20
# and 0.25; at 0.10 the cut reaches 20 of 64 and starts removing real
# designs, dropping B21 and B18 out of the top 5. G20 itself is 0.059,
# comfortably inside. CV correlates only weakly with expression (-0.07) or
# activity (-0.15), so this is not a disguised activity filter.
CV_MAX = {"1_original": 0.15,
          "2_g20_optimisation": 1.0,
          "3_experiment_guided": 1.0}
#
# THE CV GATE CURRENTLY REMOVES NOTHING from any library, and the `cv`
# column is NOT the same quantity in all three:
#
#   libraries 2, 3   bg_sub_conc_norm_rate_cv (the RATE cv), jq01's filter.
#                    Max 0.281 / 0.444 -- they were CV-filtered upstream, so
#                    re-applying it here is a no-op by construction.
#   library 1        mscarlett_cv (the EXPRESSION cv). Max 0.868.
#
# Library 1 cannot use the rate CV: saturation makes it divide by a
# near-zero mean, and a <= 1.0 gate drops 60 of 128 rows including G20. The
# rate CV is still reported as `cv_rate` so nothing is hidden. Treat a CV
# comparison ACROSS libraries as meaningless; within one it is fine.

SCHEMA = ["library", "round_number", "sample_type", "name", "sequence",
          "plate", "well", "expression_um", "n_expressing", "rate", "total_product", "rate_per_um",
          "g20_norm_rate", "cv", "cv_metric", "cv_expression", "cv_rate", "r2",
          "passes_expression", "passes_cv", "passes_filter"]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def library1(timepoint=2):
    """-> the original library, normalised to its own G20 well."""
    f = LIB1_SEPT / f"timepoint_{timepoint}" / "processed_data" / "5_complete_df.csv"
    if not f.exists():
        sys.exit(f"library 1 not found: {f}")
    d = pd.read_csv(f)

    # BACKGROUND SUBTRACT, the same way libraries 2 and 3 do: subtract the
    # mean negative-control value per plate, then divide by concentration.
    #
    # This is not cosmetic here. The negative controls read 128.8 and 158.0
    # total product against a sample median of 154, so the median design in
    # this library is INDISTINGUISHABLE FROM BACKGROUND and the raw numbers
    # are mostly background. G20 is 471 raw, 328 background-subtracted,
    # against 140 for the next design -- subtracting widens the gap from
    # 1.7x to 2.3x.
    # PER PLATE, both the background and the G20 reference. With one plate
    # in this screen the grouping is a no-op, but it is written this way so
    # the guarantee is structural: a plate's gain, read height and lamp all
    # differ slightly, so a background or a reference borrowed from another
    # plate would import that difference as signal.
    if not (d.sample_type == "negative_control").any():
        sys.exit("library 1: no negative controls to background-subtract")

    bg_product = (d[d.sample_type == "negative_control"]
                  .groupby("plate_number")["4MU_um_max_mean"].mean())
    bg_rate = (d[d.sample_type == "negative_control"]
               .groupby("plate_number")["rate_mean"].mean())
    plate = d["plate_number"]
    conc = _num(d["mscarlett_mean"]) + 1e-4

    out = pd.DataFrame({
        "library": "1_original",
        # The original library screen is one round; G20 was picked from it.
        "round_number": 0,
        "sample_type": d["sample_type"],
        "name": d["name"],
        "sequence": d["sequence"],
        "plate": plate,
        "well": d["source_dna_well"],
        "expression_um": _num(d["mscarlett_mean"]),
        # Count of replicates clearing the threshold, as jq01's pipeline does.
        "n_expressing": sum((_num(d[f"mscarlett_um_{i}"]) >= EXPRESSION_MIN).astype(int)
                            for i in (1, 2, 3)),
        "rate": _num(d["rate_mean"]) - plate.map(bg_rate),
        "total_product": _num(d["4MU_um_max_mean"]) - plate.map(bg_product),
        # Same formula as libraries 2 and 3: (rate - bg) / (conc + 1e-4).
        # Reported for comparability; NOT what library 1 is ranked on.
        "rate_per_um": (_num(d["rate_mean"]) - plate.map(bg_rate)) / conc,
        # mscarlett_cv, NOT rate_norm_cv. The rate CV divides by a near-zero
        # mean and explodes: its 75th percentile is exactly 1.732 and G20
        # itself reads 1.732, which would filter out the best design here.
        # Ranked on total product, so qualified on ITS reproducibility.
        "cv": (d[[f"4MU_um_max_{i}" for i in (1, 2, 3)]].std(axis=1, ddof=1)
               / d[[f"4MU_um_max_{i}" for i in (1, 2, 3)]].mean(axis=1)),
        "cv_metric": "total_product_cv",
        "cv_expression": _num(d["mscarlett_cv"]),
        # The rate CV is kept but NOT filtered on. It divides by a near-zero
        # mean here -- median 0.956, 75th percentile exactly 1.732 -- because
        # the reaction had saturated, and a <= 1.0 gate on it removes 60 of
        # 128 rows INCLUDING G20 itself. Carried so the number is visible
        # rather than quietly dropped.
        "cv_rate": _num(d["rate_norm_cv"]),
        "r2": pd.NA,
    })

    # Divide each plate through ITS OWN G20 well.
    refs = (out[out.well == "G20"].groupby("plate")["total_product"].mean())
    missing = set(out.plate.unique()) - set(refs.index)
    if missing:
        sys.exit(f"library 1: plates with no G20 well to normalise against: {missing}")
    out["g20_norm_rate"] = out.total_product / out.plate.map(refs)
    return out


def libraries_23():
    """-> the two follow-up libraries, already G20-normalised at source."""
    if not LIB23.exists():
        sys.exit(f"libraries 2/3 not found: {LIB23}")
    d = pd.read_csv(LIB23)
    name_map = {"first_big_library": "2_g20_optimisation",
                "second_big_library": "3_experiment_guided"}
    unknown = set(d.library_name) - set(name_map)
    if unknown:
        sys.exit(f"unexpected library_name values: {unknown}")

    out = pd.DataFrame({
        "library": d.library_name.map(name_map),
        "round_number": _num(d["round_number"]),
        "sample_type": d["sample_type"],
        "name": d["name"],
        "sequence": d["sequence"],
        "plate": _num(d["plate_number"]),
        "well": d["source_dna_well"],
        "expression_um": _num(d["mscarlett_um_mean"]),
        "n_expressing": _num(d["mscarlett_num_pass"]),
        # This pipeline reports a background-subtracted, concentration-
        # normalised rate; there is no raw-rate or total-product column, so
        # those stay empty rather than being invented from it.
        "rate": pd.NA,
        "total_product": pd.NA,
        "rate_per_um": _num(d["bg_sub_conc_norm_rate_mean"]),
        "g20_norm_rate": _num(d["g20_norm_rate"]),
        "cv": _num(d["bg_sub_conc_norm_rate_cv"]),
        "cv_metric": "bg_sub_conc_norm_rate_cv",
        "cv_expression": pd.NA,
        "cv_rate": _num(d["bg_sub_conc_norm_rate_cv"]),
        "r2": _num(d["r2_score"]),
    })
    return out


def add_filter(d):
    d = d.copy()
    # `n_expressing` DOES NOT MEAN THE SAME THING IN EVERY SOURCE, so it
    # cannot be thresholded uniformly:
    #
    #   library 1   a count this script computes, 0-3
    #   library 2   a BOOLEAN (all rows read 1) -- the kiera pipeline stores
    #               `all(replicates >= 0.1)`, so 1 means "passed", not "one
    #               replicate passed"
    #   library 3   a count, 3 or 6 (jq01's convention)
    #
    # Thresholding at >= 2 across all three silently dropped every row of
    # library 2. Libraries 2 and 3 are in any case already filtered upstream
    # -- their minimum expression is 0.109 and 0.112 uM -- so the rule is
    # applied where it is this script's job to apply it, and the upstream
    # verdict is honoured where it is not.
    counted = d.library.isin(["1_original", "3_experiment_guided"])
    d["passes_expression"] = (
        (d.expression_um >= EXPRESSION_MIN)
        & (~counted | (d.n_expressing >= MIN_REPLICATES_PASSING))
        & (counted | (d.n_expressing >= 1))
    )
    lim = d.library.map(CV_MAX)
    d["passes_cv"] = d.cv.isna() | (d.cv.abs() <= lim)
    d["passes_filter"] = d.passes_expression & d.passes_cv
    return d[SCHEMA]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timepoint", type=int, default=2,
                    help="which library-1 read to use (default 2)")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    parts = [add_filter(library1(a.timepoint)), add_filter(libraries_23())]
    allof = pd.concat(parts, ignore_index=True)

    for lib, g in allof.groupby("library"):
        p = OUT / f"petase_{lib}.csv"
        g.to_csv(p, index=False)
        print(f"wrote {p.name}: {len(g)} rows, "
              f"{int(g.passes_filter.sum())} pass the filter")

    allof.to_csv(OUT / "petase_libraries_all.csv", index=False)
    kept = allof[allof.passes_filter]
    kept.to_csv(OUT / "petase_libraries_filtered.csv", index=False)
    print(f"wrote petase_libraries_all.csv: {len(allof)} rows")
    print(f"wrote petase_libraries_filtered.csv: {len(kept)} rows")

    print("\nper library, filtered:")
    s = (kept[kept.sample_type == "sample"]
         .groupby("library")
         .agg(n=("g20_norm_rate", "size"),
              median_x_g20=("g20_norm_rate", "median"),
              max_x_g20=("g20_norm_rate", "max"),
              pct_above_g20=("g20_norm_rate", lambda v: 100 * (v > 1).mean()))
         .round(2))
    print(s.to_string())


if __name__ == "__main__":
    main()
