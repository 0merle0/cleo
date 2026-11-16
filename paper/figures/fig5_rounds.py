#!/usr/bin/env python3
"""Figure 5 --- searching a 10^12 library in four rounds.

The story draft's Figure 4. Every round the draft describes exists in the
snapshot and is identifiable from the design names:

    round 0   single-body      4 of 5 slots are the G20 parent (1665 / 1791)
    round 1   binned combos    tags bin1..bin6, plus top2combos (31 of 32)
    round 2+  model-selected   tags MLPgamma*, topkMLP, top30mu, mostdiverse*

so (a) through (d) are build jobs rather than collection jobs.

    uv run python paper/figures/fig5_rounds.py
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
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 8.4))
    gs = fig.add_gridspec(4, 2, left=0.045, right=0.985, top=0.978,
                          bottom=0.025, wspace=0.11, hspace=0.20)

    slot(fig.add_subplot(gs[0, 0]), "a",
         "Round 1: every part on its own",
         "have", "snapshot round 0 -- 1665 / 1791 designs",
         lines=[
             "each part combined with the four G20 parent parts",
             "activity by slot, one distribution per slot",
             "verified: every round-0 row has exactly one non-parent slot",
         ])

    slot(fig.add_subplot(gs[0, 1]), "b",
         "Single-body effect vs distance",
         "have", "same rows; compute Hamming on the varied part",
         lines=[
             "the draft's claim: parts >4 mutations out still improve",
             "TRAP: do not use the distN tag -- it disagrees with",
             "computed distance on 992 of 1803 parts",
         ])

    slot(fig.add_subplot(gs[1, 0]), "c",
         "Round 2: bin, then draw",
         "have", "snapshot round 1 tags",
         lines=[
             "bin1 .. bin6 by single-body efficacy, ~120-165 each",
             "plus top2combos: 31 of the 2^5 = 32 exhaustive top-2 builds",
             "(the draft writes '25 = 32' -- it means 2^5)",
             "expect and show the drop-off; the last bin is ungated,",
             "i.e. a random draw from the library",
         ])

    slot(fig.add_subplot(gs[1, 1]), "d",
         "Rounds 3-4: the surrogate picks",
         "have", "snapshot rounds 2-4 tags",
         lines=[
             "predicted vs measured, coloured by selection strategy",
             "lib 2: MLPgamma1/5/8, topkMLP, sampledtopkMLP",
             "lib 3: top30mu, top10mu, mostdiverseMuGt3/4",
             "gamma is the diversity weight -- a sweep, so plot it as one",
             "selection is from the top 1%, not top-k: say why (diversity)",
         ])

    slot(fig.add_subplot(gs[2, 0]), "e",
         "What the search cost",
         "derive", "computable; the greedy comparator needs defining",
         lines=[
             "~10^4 assayed variants against a >10^12 library",
             "the draft's '(put real savings here!)'",
             "DECIDE FIRST what greedy means here, or the saving is",
             "a number with no denominator",
         ])

    slot(fig.add_subplot(gs[2, 1]), "f",
         "Against the natural PETases",
         "have", "jq_e0328, 37 C overnight kinetics",
         lines=[
             "one plate: designs, natives AND momi WT together",
             "  best design 150.4    kubu_pm12 56.7",
             "  ThermoPETase 44.5    FAST_PETase 44.3",
             "  LCC 19.8             momi WT 0.383",
             "so, with no cross-experiment factor at all:",
             "  393x over momi WT,  3.4x FastPETase,  7.6x LCC",
             "CAVEATS: overnight 37 C, n=1 per construct, and",
             "the denominator sits near background. This plate's",
             "own G20/momi is 34.0x, not the 19.7x of Fig 4(f)",
             "-- never mix the two",
         ])

    slot(fig.add_subplot(gs[3, 0]), "g",
         "Probe gains do not reach bulk PET",
         "collect", "jq_e0334 nanopet titration -- notebook only",
         lines=[
             "probe activity vs nanoPET / pellet activity, same designs",
             "the draft states this negative result in prose.",
             "A stated negative needs its panel, or cut the sentence.",
             "Also jq_e0331 / jq_e0333 HPLC validation sets",
         ])

    slot(fig.add_subplot(gs[3, 1]), "h",
         "Active variants far from the parent",
         "have", "snapshot -- but check the sign before writing it",
         lines=[
             "activity vs distance from momi, final libraries",
             "WARNING: in the FIRST library activity fell monotonically",
             "with distance (median 218 at 0-9 mutations, 0.2 at 50+).",
             "key_results.md section 7. If libraries 2/3 reverse that,",
             "that reversal is the result -- show both, do not show one",
         ])

    save(fig, "fig5_rounds", dpi=900, tight=False)


if __name__ == "__main__":
    main()
