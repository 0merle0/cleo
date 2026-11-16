#!/usr/bin/env python3
"""Figure 2 --- AME: the method works, and how far it scales.

The second half of what the story draft called Figure 1. Split out because
"the method is this" and "here is the benchmark that says it works" are two
different claims and the draft's references to them were indistinguishable.

All four panels have data; see paper/figure_plan_story.md section 1.

    uv run python paper/figures/fig2_benchmark.py
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
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 5.0))
    gs = fig.add_gridspec(2, 2, left=0.045, right=0.985, top=0.97,
                          bottom=0.035, wspace=0.11, hspace=0.20)

    slot(fig.add_subplot(gs[0, 0]), "a",
         "Recombinants fold, and training for it is why",
         "have", "key_results.md section 6; arm_table.py",
         lines=[
             "recombinant pass rate, AF3 best-of-5, held out of training",
             "sampled  |  plain RL (RF3 only)  |  chimera-trained (pool)",
             "M0097 96.9   M0255 82.4   M0315 100.0",
             "plain-RL seed replicates bound the lucky draw:",
             "the chimera arm clears the band by +25.8 / +52.0 / +34.8 pt",
             "DO NOT plot retention -- unstable denominator, 0.77 to 1.61",
             "on one unchanged config",
         ])

    slot(fig.add_subplot(gs[0, 1]), "b",
         "The library carries more information, not just more members",
         "have", "ame_fragk_diversity.csv; key_results.md section 1",
         lines=[
             "distinct (position, residue) substitutions over members that PASS",
             "rarefied to a matched passing count -- diversity among failures",
             "is diversity you cannot order",
             "sampled 721.6  ->  RL policy 859.6, 6 of 8 backbones",
         ])

    slot(fig.add_subplot(gs[1, 0]), "c",
         "Where in sequence space each library actually sits",
         "derive", "archived ame_pca.py; third arm needs building",
         lines=[
             "one embedding, three arms drawn in it:",
             "  random mutagenesis (scoped)  |  naive ProteinMPNN  |  post-trained",
             "GAP: no random-mutagenesis arm exists at matched size.",
             "jq_e0307_250311_petase_library_random_mutation has AF3 output",
             "for one -- check whether it can serve directly",
         ])

    slot(fig.add_subplot(gs[1, 1]), "d",
         "How fine the library can be cut before it stops folding",
         "have", "ame_fragk.csv -- trained-at-k vs post-hoc cut-at-k",
         lines=[
             "retention vs k, k = 2..64, three backbones, two conditions",
             "k=16 retention 0.910 / 0.871 / 0.984 -- flat, not the edge",
             "cliff at k=36 (0.559 / 0.406 / 0.832) and k=64",
             "draft says 'out to k=16'; the data says flat through 16,",
             "falls by 36 -- claim the measured thing",
             "CAVEAT: M0255 at k=36/64 still climbing at step 200 = lower bound",
         ])

    save(fig, "fig2_benchmark", dpi=900, tight=False)


if __name__ == "__main__":
    main()
