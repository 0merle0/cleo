#!/usr/bin/env python3
"""Snapshot BOTH MoMI PETase libraries, all rounds, into paper/figures/data/.

The source is jq01's combined training set for the second library's round 3,
which is the only table that carries `library_name` and therefore the only
one in which the two libraries can be told apart:

    /home/jq01/projects/jq_p07_active_learning/
      jq_e0325_260123_petase_g20_2nd_big_library_rd3_analysis/
      2_model_training/input/
      260125_1st_big_lib_and_2nd_big_lib_round0_round1_round2_round3_with_val.csv

The copy already in data/momi/ is the FIRST library only (3,424 rows, no
library column), which is why the second library looked missing.

Both libraries are k=5 combinatorial fragment libraries on the same 209-aa
G20 parent, four rounds each, and both tag every design with how it was
selected -- so "how far from parent" and "how it was chosen" are readable off
the name:

    1.refw5.step0200.0318.18bafee5___2.g20___3.g20___4....___5....:bin3
    \------------- one part per slot, g20 = parent -------------/ \tag/

They differ in where their non-parent parts came from: library 1 draws from
`refw5`/`refw8` RL checkpoints, library 2 from `dist3`..`dist9` pools.

    uv run python paper/figures/momi_libraries_snapshot.py
"""

import sys
from pathlib import Path

import pandas as pd

SRC = Path("/home/jq01/projects/jq_p07_active_learning/"
           "jq_e0325_260123_petase_g20_2nd_big_library_rd3_analysis/"
           "2_model_training/input/"
           "260125_1st_big_lib_and_2nd_big_lib_round0_round1_round2_round3"
           "_with_val.csv")
OUT = Path(__file__).resolve().parent / "data" / "momi_libraries.csv"

KEEP = ["library_name", "round_number", "sample_type", "name", "sequence",
        "g20_norm_rate", "momi_norm_rate", "z_score_norm_rate", "val"]


def main():
    if not SRC.exists():
        sys.exit(f"source not readable: {SRC}")
    d = pd.read_csv(SRC)
    missing = [c for c in KEEP if c not in d.columns]
    if missing:
        sys.exit(f"source is missing {missing}; did the schema change?")
    d = d[KEEP].copy()

    # Derived once here rather than re-parsed in every figure: the selection
    # tag after the colon, and how many of the k=5 slots still hold the
    # parent part.
    stem = d.name.astype(str).str.split(":").str[0]
    d["selection"] = (d.name.astype(str).str.extract(r":([A-Za-z0-9]+)$")[0]
                      .fillna("none"))
    d["k"] = stem.str.count("___") + 1
    d["parent_slots"] = stem.apply(
        lambda s: sum(1 for p in s.split("___") if "g20" in p))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(d)} rows")
    print(d.groupby(["library_name", "round_number"]).size()
          .rename("n").to_string())
    s = d[d.sample_type == "sample"]
    print("\nbest g20_norm_rate per library:")
    for lib, sub in s.groupby("library_name"):
        b = sub.loc[sub.g20_norm_rate.idxmax()]
        print(f"  {lib}: {b.g20_norm_rate:.2f}x G20 in round "
              f"{int(b.round_number)} ({b.selection})")


if __name__ == "__main__":
    main()
