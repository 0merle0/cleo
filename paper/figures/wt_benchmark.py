#!/usr/bin/env python3
"""How much faster are our designs than the natural PETases?

The benchmark enzymes -- FastPETase, ThermoPETase, LCC -- and our best
designs were never run in one experiment. They are bridged through G20,
which is a positive control in both:

    benchmarks      jq_e0321 pool screen, measured as x G20
    our designs     libraries 1-3, measured as x G20 on their own plates

so a design's advantage over FastPETase is just the ratio of the two x G20
numbers. That is only as good as the bridge, and the bridge has limits --
see THE CAVEATS at the bottom of the output, which are printed with the
table rather than left in a docstring.

    uv run python paper/figures/wt_benchmark.py
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
POOL = Path("/home/jq01/projects/jq_p07_active_learning"
            "/jq_e0321_260108_petase_g20_2nd_big_library_pool_screen_reading_analysis"
            "/260109_processed_filtered_data_round0_pooled_2nd_big_lib.csv")
SNAP = HERE / "data" / "petase_libraries_filtered.csv"

BENCHMARKS = ["Fast_PETase", "Thermo_PETase", "LCC", "momi_wt"]


def main():
    if not POOL.exists():
        sys.exit(f"pool screen not found: {POOL}")
    if not SNAP.exists():
        sys.exit(f"run libraries_snapshot.py first: {SNAP}")

    pool = pd.read_csv(POOL)
    bench = (pool[pool.name.isin(BENCHMARKS)]
             .set_index("name")[["g20_norm_rate", "mscarlett_um_mean",
                                 "bg_sub_conc_norm_rate_cv"]])
    missing = [b for b in BENCHMARKS if b not in bench.index]
    if missing:
        sys.exit(f"benchmarks missing from the pool screen: {missing}")

    print("BENCHMARKS, from the jq_e0321 pool screen (x G20, one plate)\n")
    b = bench.rename(columns={"g20_norm_rate": "x_G20",
                              "mscarlett_um_mean": "expression_uM",
                              "bg_sub_conc_norm_rate_cv": "cv"})
    print(b.round(3).to_string())

    snap = pd.read_csv(SNAP)
    snap = snap[snap.sample_type == "sample"]
    best = (snap.sort_values("g20_norm_rate", ascending=False)
                .groupby("library").head(1)
                .set_index("library")[["round_number", "well",
                                       "g20_norm_rate", "momi_norm_rate"]])

    print("\n\nBEST DESIGN PER LIBRARY (x G20, its own plates)\n")
    print(best.round(2).to_string())

    print("\n\nHOW MANY TIMES FASTER THAN EACH BENCHMARK\n")
    tab = pd.DataFrame(
        {name: (best.g20_norm_rate / row.x_G20).round(1)
         for name, row in b.iterrows()})
    print(tab.to_string())

    print("""

THE CAVEATS, which matter more than the numbers:

  1. THIS IS A CROSS-EXPERIMENT COMPARISON. The benchmarks and the designs
     were never on one plate. Everything is bridged through G20, so the
     ratios assume G20 means the same thing in both -- and G20's RAW rate
     differs by 35% between libraries 2 and 3 (5.36e-05 against 3.46e-05).
     Normalising to G20 per plate is what makes the comparison possible at
     all, but it is doing real work here.

  2. THE DESIGNS SIT FAR OUTSIDE THE RANGE THE BENCHMARKS WERE MEASURED IN.
     In the pool screen everything lands between 0.045 and 1.92 x G20. Our
     best design is 20.05 x G20 -- roughly 10x beyond the top of that range.
     Nothing here establishes that the assay is linear that far out, and if
     it compresses at the top the advantage is UNDERSTATED; if it saturates
     differently between plates it could be overstated. A direct head-to-head
     on one plate is the only way to settle it.

  3. One plate, one condition, n=1 per benchmark. The pool screen measured
     each benchmark once, with 3 replicate wells.

  4. All of this is the plate-reader proxy assay at ambient temperature, not
     a kinetic characterisation on PET.""")


if __name__ == "__main__":
    main()
