#!/usr/bin/env python3
"""Figure 4 --- the experimental platform, and the three libraries.

The story draft's Figure 3. Panel (a) is already built by someone else as
svg/wetlab_pipeline.svg; this lays out the slot it goes in and everything
around it. Panel (e) is new -- the draft labels c, d and f and skips e, and
the worked kinetic trace is the thing missing between "we fit a rate" and
"here are the rates".

    uv run python paper/figures/fig4_platform.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from blockout import slot  # noqa: E402
from figio import save  # noqa: E402
from palette import PAGE_WIDTH_IN  # noqa: E402


def main():
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 7.4))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.62, 1.25, 1.15],
                             left=0.045, right=0.985, top=0.975,
                             bottom=0.03, hspace=0.17)

    top = outer[0].subgridspec(1, 1)
    slot(fig.add_subplot(top[0, 0]), "a",
         "The assembly and assay pipeline",
         "built", "svg/wetlab_pipeline.svg -- splice it into this slot",
         lines=[
             "geneblocks with BsaI adapters -> ECHO acoustic dispense ->",
             "Golden Gate 37 C 1 h -> PCR in the same plate -> PUREfrex,",
             "3 replicates -> add PET-trimer probe -> plate reader",
         ])

    mid = outer[1].subgridspec(1, 3, wspace=0.11)

    slot(fig.add_subplot(mid[0, 0]), "b",
         "The filter cascade",
         "have", "libraries_snapshot.py passes_*",
         lines=[
             "counts surviving each gate:",
             "assayed -> expressed",
             ">= 0.1 uM in every",
             "replicate -> CV < 1.0",
             "state n at each step,",
             "all three libraries",
             "NOTE library 1 keeps",
             "67 of 372 -- the",
             "cascade must show that",
         ])

    slot(fig.add_subplot(mid[0, 1]), "c",
         "Controls agree\nacross plates",
         "have", "snapshot controls -- plot RAW rate",
         lines=[
             "positive-control rate per",
             "plate, per round",
             "within a round the three",
             "plates agree to 1-2%",
             "(20.70 / 20.18 / 20.05)",
             "TRAP: momi_norm_rate is",
             "19.7 BY CONSTRUCTION in",
             "most rounds -- plotting it",
             "shows the definition, not",
             "the reproducibility",
         ])

    slot(fig.add_subplot(mid[0, 2]), "d",
         "Replicate parity",
         "collect", "re-extract from the raw readers",
         lines=[
             "replicate i vs replicate j,",
             "one point per construct",
             "GAP: the snapshot keeps",
             "cv_rate but not the three",
             "values it came from.",
             "Pull them from jq01's",
             "per-round raw readers,",
             "the way library1_kinetics.py",
             "already does",
         ])

    bot = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.11)

    slot(fig.add_subplot(bot[0, 0]), "e",
         "One worked trace",
         "have", "library1_kinetics.py reads these",
         lines=[
             "raw 4MU fluorescence vs time",
             "for one well, with:",
             "  the per-plate background",
             "  (MEDIAN of the no-enzyme",
             "   wells -- one outlier at",
             "   222.7 against five at ~127",
             "   was setting the mean)",
             "  the least-squares fit",
             "  the r^2",
             "this is what 'a rate' means",
             "and nothing else shows it",
         ])

    slot(fig.add_subplot(bot[0, 1]), "f",
         "All three libraries, one axis",
         "have", "petase_libraries_filtered.csv",
         lines=[
             "activity in x momi WT, by round, all three libraries",
             "  1 original            n=67     median  1.15   max  17.1",
             "  2 G20 optimisation    n=3399   median  3.69   max  66.0",
             "  3 experiment-guided   n=3386   median 20.27   max 395.0",
             "G20 does NOT land at one height: 17.11 measured in library 1",
             "against its own momi control, 19.70 derived in libraries 2/3.",
             "That 26% disagreement is the factor's whole uncertainty --",
             "leave it visible rather than normalising it away",
             "(docs/MOMI_NORMALISATION.md)",
         ])

    save(fig, "fig4_platform", dpi=900, tight=False)


if __name__ == "__main__":
    main()
