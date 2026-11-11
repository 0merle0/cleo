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

!! LIBRARY 3's LATER ROUNDS ARE OVER-NORMALISED IN THE SOURCE FILE USED
HERE, and its headline maximum is inflated by 1.74x. Do not publish the
library 3 top numbers from this snapshot until it is re-sourced.

The kiera combined file (260128_all_data_lib1_lib2.csv) normalises each
plate against that plate's own G20 positive controls. The number of those
controls collapses over rounds -- 5 per plate in round 0, 2 in round 1, and
ONE from round 2 onwards -- so by round 3 each plate's entire scale rests on
a single well. Round 3 plate 1's single control reads 0.000018 against
0.000032 on plate 2, and everything on that plate inflates accordingly.

The tell is that plate medians should agree within a round and do not:

    kiera combined, round 3   plate medians 13.24 / 10.46 / 10.77  (24% spread)
                              max 33.02 x G20 = 429 x momi

    jq01's own round-3 analysis, which normalises plate1+plate3 together and
    plate2 apart:
                              plate medians  8.10 /  8.14 /  8.02  (0.5%)
                              max 18.98 x G20 = 247 x momi

jq01's version is right: agreeing medians are what correct per-plate
normalisation looks like. Their round-4 file likewise tops out at 20.05 x
G20 = 261 x momi. Both land at the ~250 x momi ceiling, against the 429
this snapshot currently reports.

Per-round plate spread, kiera combined: library 2 runs 11-18% across all
rounds; library 3 runs 12%, 11%, 21%, 24% for rounds 0-3, degrading exactly
as the controls thin out.

TO FIX: source libraries 2 and 3 from jq01's per-round analysis outputs
(jq_e0321/e0323/e0324/e0325/e0327) rather than the kiera combined file.

WHERE momi WT IS MEASURED, AND WHAT IT READS. It is NOT a routine control in
libraries 2 and 3: across jq01's 32 plate maps the only controls are
`negative_control` (279 wells) and `1.g20___..._5.g20` (189 wells), and the
momi WT sequence appears in just 4 maps -- the pooled screen and three
`top_design_vs_native` analyses. In library 1 it IS the positive control.

Everywhere it was measured, as a multiple of G20:

    library 1, total product, 21 min     0.042     (G20 ~24x)
    pooled screen, rate, 3 min           0.045     (G20 ~22x)
    e0328 round 5, 37C short            -0.017     below background
    e0328 round 5, 60C short            -0.399     below background
    e0328 overnight, 37C                 0.029     (G20 ~34x)
    e0328 overnight, 60C                 0.147     (G20 ~6.8x)

The two NEGATIVE values are the point: in a short read momi WT is at or
below background, so it cannot anchor anything. It only becomes measurable
when the assay runs long -- library 1's 21 minutes, or an overnight. That is
why G20 and not momi WT is the positive control in every later library, and
why `momi_norm_rate` is a conversion factor rather than a measurement.

The two clean short-assay-independent estimates agree closely -- 0.042 from
library 1's total product and 0.045 from the pooled screen's rate -- which
is the cross-check that lets library 1 share an axis with the others.

IS G20 CONSISTENT ACROSS LIBRARIES? Within a library, yes; between
libraries, no -- which is why every value here is normalised to G20 on its
own plate rather than compared in raw units.

    G20 raw bg-subtracted conc-normalised rate
    library 2   5.357e-05 +/- 5.1e-06   (CV  9%)   plates 5.43 5.26 5.38
    library 3   3.457e-05 +/- 7.2e-06   (CV 21%)   plates 3.43 3.59 3.36
    ratio       0.65x

Plates within a library agree tightly; the two libraries differ by 35%, far
more than that. Raw rates are therefore not comparable across libraries and
the x G20 axis is doing real work.

AND A CROSS-ASSAY CHECK THAT PASSES. momi WT is not in libraries 2 or 3 --
their only positive control is G20 -- but the pooled screen
(jq_e0321_.../260109_processed_filtered_data_round0_pooled_2nd_big_lib.csv)
measures both: momi_wt reads 0.045 x G20, i.e. G20 ~22x momi WT.

Library 1 gives ~24x by a completely different route: total product on a
saturated plate, 327.7 / 13.7. Two assays, two quantities, same answer to
within 10%. That is the strongest evidence that library 1 belongs on the
same axis as the other two despite its saturation.

(jq01's pipeline applies momi_norm_rate = g20_norm_rate * 13. Both routes
here land nearer 22, so treat the 13 as the approximation it is labelled
as.)

THE NEGATIVE CONTROL IS THE RIGHT KIND BUT A THIN ESTIMATE, so library 1's
median is soft and should not be quoted as a precise number.

P24 is vector-only, which is what a background should be. But it is n=2 and
the two readings are 23% apart (128.8 and 158.0 total product), and it sits
ABOVE 19% of the samples. Library 1's median moves with the choice:

    background              median x G20
    neg-ctrl mean (used)        0.032
    neg-ctrl min                0.073
    momi WT                    -0.010
    sample 5th percentile       0.082
    none                        0.327

G20 is unaffected -- it is the normaliser, so it reads 1.00 under every
choice, and its rank is likewise untouched. Statements about the TOP of
library 1 are robust; statements about its median or its shape are not.

AND WHY G20, NOT momi WT, IS THE REFERENCE. Library 1's positive control is
momi WT (1.wt_2.wt_3.wt_4.wt_5.wt), and it reads 143.5 and 170.6 against the
vector control's 128.8 and 158.0 -- barely above background. That is not an
assay failure; momi WT is a weak enzyme and being hard to see above vector
is the premise of the whole campaign.

It does mean momi WT cannot serve as a denominator here: subtracting
background leaves it at 13.7, so anything divided by it inherits that
near-zero's noise. Taken at face value G20 is ~24x momi WT, the same order
as the 13x factor jq01's pipeline applies for `momi_norm_rate`, but on a
denominator that thin it is order-of-magnitude at best -- which is also why
that 13x is a stated approximation rather than a measurement.

G20 is used as the reference instead, in every library, because it is the
one design comfortably above background in all three.

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

That is why library 1 saturates: 1.6x the gain over 7x the time.

`rate_per_um` IS computed for library 1, by exactly the libraries-2-and-3
formula -- (rate - plate background) / (concentration + 1e-4) -- so the
column exists and is built the same way. It carries NO SIGNAL, and that is
a property of the raw data rather than a judgement call. From the reader
files:

  read1 is a kinetic read in which EVERY well declines at the same rate.
  Over 384 wells: median -187.1/min, G20 -187.5/min (the median exactly),
  negative control -220.6/min, sd 27.6. That is photobleaching, uniform
  across the plate. Subtract the control slope and G20 reads +33.1/min
  against a median of +33.5/min -- no separation whatsoever.

  read2 is flat: G20 holds 1716/1717/1678/1695 across four reads while the
  next design sits near 988.

There is no accumulation phase anywhere to fit a rate to, because the
reaction finished before either read. The ~0 and slightly negative rates in
the processed file are the correct answer to the wrong question.

What separates designs here is the LEVEL the signal had already reached --
total product, G20 at 1716 against 988 for the next design. That is what
library 1 is ranked on. `rate_per_um` stays in the schema for symmetry and
tells you nothing; plotting it without reading this will mislead.

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
# Libraries 2 and 3 come from jq01's PER-ROUND analyses, not from the kiera
# combined file. The combined file normalises each plate against that plate's
# own G20 controls, and those thin to ONE well from round 2 on, which
# inflated library 3's round-3 maximum by 1.74x (33.02 against 18.98).
JQ = Path("/home/jq01/projects/jq_p07_active_learning")
LIB2_FILE = (JQ / "jq_e0319_251223_petase_g20_all_rounds_data"
             / "251225_processed_filtered_data_round0_round1_round2_round3_with_val.csv")
LIB3_FILES = {
    0: JQ / "jq_e0322_260115_petase_g20_2nd_big_library_frag_swapping_analysis"
          / "260115_processed_filtered_data_round0.csv",
    1: JQ / "jq_e0323_260116_petase_g20_2nd_big_library_rd1_analysis"
          / "260116_processed_filtered_data_round1.csv",
    2: JQ / "jq_e0324_260119_petase_g20_2nd_big_library_rd2_analysis"
          / "1_exp_data_analysis/output/260119_processed_filtered_data_round2.csv",
    3: JQ / "jq_e0325_260123_petase_g20_2nd_big_library_rd3_analysis"
          / "1_exp_data_analysis/output/260123_processed_filtered_data_round3_all_plates.csv",
    4: JQ / "jq_e0327_260128_petase_g20_2nd_big_library_rd4_analysis"
          / "1_exp_data_analysis/output/260128_processed_filtered_data_round4.csv",
}

# Rounds whose per-plate normalisation fails the plate-median test and are
# renormalised here against the round's POOLED G20 controls.
#
# The test: within one round, plate medians should agree. Where they do not
# and the plates carry only one control each, the single well is setting the
# scale rather than correcting for it.
#
#   library 3 round 2   per-plate spread 21% -> pooled 4%, max 17.97 -> 15.94
#
# Not applied anywhere else, and deliberately not to library 3 round 3: jq01
# already fixed that one by normalising plate1+plate3 together, which gets
# the spread to 2%. Re-pooling it on top would undo their correction and
# push the spread to 39%.
REPOOL = {("3_experiment_guided", 2)}

# G20 -> momi WT conversion for the `momi_norm_rate` column.
#
# momi WT cannot be the MEASURED reference: in a short read it sits at or
# below background, reading negative at 37C and 60C in e0328. So values are
# measured against G20 per plate and converted once here.
#
# 13 is jq01's house constant, hardcoded in all their processing notebooks as
# `g20_norm_rate * 13  # approximate to momi activity`, and is used so these
# numbers match the rest of the project.
#
# NOT SETTLED, and library 1 cannot settle it.
#
# momi WT appears in library 1 only. Searched by sequence: 2 exact matches
# there (its two P23 control rows) and ZERO in libraries 2 and 3, which
# cannot contain it -- they are built from G20 fragments, and G20 is 21
# mutations from momi, so their closest designs sit 9 and 12 mutations away.
#
# The two measurements disagree and library 1 brackets both:
#
#   pooled screen, one plate, r2 0.99994, 0.787 uM, 3/3 reps, CV 0.4%
#       momi = 0.045 x G20  ->  22.2x
#       under 13x that row's own momi_norm_rate reads 0.585, not 1.0
#
#   library 1, its two momi control rows
#       P23:1  0.046 x G20  ->  21.6x
#       P23:2  0.125 x G20  ->   8.0x      2.7x apart
#       mean 11.7x, and the range covers BOTH candidates
#       under 13x their own momi_norm_rate averages 1.12, close to 1.0
#
# So 13 is supported by library 1's controls averaging near 1.0 on their own
# axis, and 22.2 by the pooled screen's much cleaner single measurement.
# Set MOMI_PER_G20 = 22.2 to switch; it is a constant rescale and changes no
# x G20 value, ranking or percentage.
MOMI_PER_G20 = 13.0

EXPRESSION_MIN = 0.1      # uM mscarlett, per replicate
MIN_REPLICATES_PASSING = 2
# CV <= 1.0 EVERYWHERE, which is a principled bound rather than a tuned one:
# CV >= 1 means the standard deviation has reached the mean, so the threshold
# asks "is this population hypervariate?" and not "where does the percentile
# fall?". It is jq01's threshold and it is kept unchanged for all three.
#
# An earlier version of this file set 0.15 for library 1, picked from its own
# percentiles because 1.0 removed nothing there. That was a tuned cutoff
# wearing QC's clothes: it had no meaning outside this dataset and would move
# with the next plate. Removing nothing is the correct outcome when nothing is
# hypervariate.
#
# WHAT THE CV IS COMPUTED ON still differs per library, and that part matters:
# each is qualified on the quantity it is RANKED on.
#
#   libraries 2, 3   rate CV   (bg_sub_conc_norm_rate_cv), max 0.281 / 0.444
#   library 1        TOTAL-PRODUCT CV, max 0.242
#
# Library 1's rate CV would be the wrong quantity twice over: it is not what
# library 1 is ranked on, and saturation drives it to a median of 0.956 with a
# 75th percentile of exactly 1.732 -- hypervariate by this test, but because
# the mean is near zero, not because the measurement is noisy. Its replicates
# are in fact tight: G20 reads 467/500/446. Reported as `cv_rate`, not
# filtered on.
CV_MAX = 1.0

SCHEMA = ["library", "round_number", "sample_type", "name", "sequence",
          "plate", "well", "expression_um", "n_expressing", "rate", "total_product", "rate_per_um",
          "g20_norm_rate", "momi_norm_rate", "cv", "cv_metric", "cv_expression", "cv_rate", "r2",
          "norm_scheme", "n_positive_controls",
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

    # MEDIAN over the individual replicate WELLS, not the mean of row means.
    #
    # Library 1 has 6 negative-control wells and five of them agree closely
    # -- 125.0, 125.6, 126.4, 128.8, 131.9 -- while the sixth reads 222.7.
    # Averaging lifts the background from 127.6 to 143.4, and because momi WT
    # sits only ~30 above background that one well moves the derived
    # G20/momi ratio from 11.7x to 24.0x. The median is unmoved by it.
    _neg = d[d.sample_type == "negative_control"]

    def _bg(cols):
        long = _neg.melt(id_vars="plate_number", value_vars=cols)
        return long.groupby("plate_number")["value"].median()

    bg_product = _bg([f"4MU_um_max_{i}" for i in (1, 2, 3)])
    bg_rate = _bg([f"rate_{i}" for i in (1, 2, 3)])
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
        "norm_scheme": "single_plate_own_g20",
        "n_positive_controls": int((d.sample_type == "positive_control").sum()),
    })

    # Divide each plate through ITS OWN G20 well.
    refs = (out[out.well == "G20"].groupby("plate")["total_product"].mean())
    missing = set(out.plate.unique()) - set(refs.index)
    if missing:
        sys.exit(f"library 1: plates with no G20 well to normalise against: {missing}")
    out["g20_norm_rate"] = out.total_product / out.plate.map(refs)
    return out


def _one_source(path, library):
    """-> normalised rows from one of jq01's processed files."""
    d = pd.read_csv(path)
    out = []
    for rd, grp in d.groupby("round_number"):
        samp = grp[grp.sample_type == "sample"].copy()
        if not len(samp):
            continue
        ctrl = grp[grp.sample_type == "positive_control"]
        n_ctrl = len(ctrl)
        if (library, int(rd)) in REPOOL and n_ctrl:
            # Recover the raw rate (jq01 divided by the plate's own control)
            # and renormalise against the round's pooled controls.
            per_plate = ctrl.groupby("plate_number").bg_sub_conc_norm_rate_mean.mean()
            raw = samp.g20_norm_rate * samp.plate_number.map(per_plate)
            samp["g20_norm_rate"] = raw / ctrl.bg_sub_conc_norm_rate_mean.mean()
            samp["norm_scheme"] = "pooled_round_controls"
        else:
            samp["norm_scheme"] = "per_plate" if n_ctrl else "no_controls_on_round"
        samp["n_positive_controls"] = n_ctrl
        out.append(samp)
    return pd.concat(out, ignore_index=True)


def libraries_23():
    """-> the two follow-up libraries, from jq01's per-round analyses."""
    parts = [_one_source(LIB2_FILE, "2_g20_optimisation")
             .assign(library="2_g20_optimisation")]
    for rd, f in sorted(LIB3_FILES.items()):
        if not f.exists():
            sys.exit(f"library 3 round {rd} missing: {f}")
        parts.append(_one_source(f, "3_experiment_guided")
                     .assign(library="3_experiment_guided"))
    d = pd.concat(parts, ignore_index=True)

    return pd.DataFrame({
        "library": d["library"],
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
        "norm_scheme": d["norm_scheme"],
        "n_positive_controls": d["n_positive_controls"],
    })


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
    d["passes_cv"] = d.cv.isna() | (d.cv.abs() <= CV_MAX)
    d["momi_norm_rate"] = d.g20_norm_rate * MOMI_PER_G20
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
    # `max` is the best design as a multiple of G20 -- but G20 is a SAMPLE in
    # library 1 (well G20) and a POSITIVE CONTROL in libraries 2 and 3. So
    # library 1's max includes G20 itself while the others exclude it, and
    # reading the column across libraries without that in mind compares "the
    # best design, which is G20" against "the best design that is not G20".
    # `max_excl_g20` is the like-for-like number.
    samp = kept[kept.sample_type == "sample"]
    not_g20 = samp[(samp.g20_norm_rate - 1).abs() > 1e-9]
    s = (samp.groupby("library")
         .agg(n=("g20_norm_rate", "size"),
              median_x_momi=("momi_norm_rate", "median"),
              max_x_momi=("momi_norm_rate", "max"),
              pct_above_g20=("g20_norm_rate", lambda v: 100 * (v > 1).mean()))
         .round(2))
    s["max_excl_g20_x_momi"] = (not_g20.groupby("library")
                                .momi_norm_rate.max().round(1))
    print(s.to_string())
    print("\nmax includes G20 for library 1 (it is a sample there) and "
          "excludes it for 2 and 3 (a control); max_excl_g20 is comparable.")


if __name__ == "__main__":
    main()
