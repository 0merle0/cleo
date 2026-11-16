#!/usr/bin/env python3
"""Figure 3 --- the PETase, and what post-training did to it.

The story draft's Figure 2. Renumbered because the concept/benchmark split
pushes everything down one; see paper/figure_plan_story.md section 0.

    uv run python paper/figures/fig3_petase_design.py
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
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 5.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], left=0.045,
                             right=0.985, top=0.97, bottom=0.035, hspace=0.20)

    top = outer[0].subgridspec(1, 1)
    slot(fig.add_subplot(top[0, 0]), "a",
         "Where momi came from: theozyme to an assayed enzyme",
         "have", "structures/png/momi_fold.png, momi_site.png",
         lines=[
             "PET polymer docked to catalytic residues -> RFD2 backbone ->",
             "ProteinMPNN sequence -> geometric filters -> screen -> momi",
             "the filters, stated in the draft, belong ON this panel:",
             "  Ser OG - His NE2 < 3 A      oxyanion contacts < 3.5 A",
             "  Asp O(-) - His N < 3.5 A    active-site pLDDT > 80",
             "momi is LOW activity -- say so here, it is the premise",
             "of everything downstream",
         ])

    bot = outer[1].subgridspec(1, 3, wspace=0.11)

    slot(fig.add_subplot(bot[0, 0]), "b",
         "Post-training\ntightens the reward",
         "derive", "PETase run metrics; cf. panel_vanilla.py",
         lines=[
             "per-reward distributions,",
             "naive vs post-trained",
             "the claim is VARIANCE",
             "falling, not only the",
             "mean rising -- plot both",
             "or the sentence is",
             "unsupported",
         ])

    slot(fig.add_subplot(bot[0, 1]), "c",
         "Catalytic geometry,\ntwo libraries overlaid",
         "derive", "render_momi.py; needs both arms folded",
         lines=[
             "superposed active-site",
             "atoms, naive cloud vs",
             "optimised cloud",
             "same view, same",
             "superposition target,",
             "or the comparison is",
             "an artefact of framing",
         ])

    slot(fig.add_subplot(bot[0, 2]), "d",
         "Centring the library\non known hits",
         "have", "part names; compute real Hamming",
         lines=[
             "library 3 tethered to the",
             "top-3 measured variants;",
             "library 2 was not",
             "TRAP: the distN tag in a",
             "part name disagrees with",
             "computed distance on",
             "992 of 1803 parts --",
             "compute it, do not read it",
         ])

    save(fig, "fig3_petase_design", dpi=900, tight=False)


if __name__ == "__main__":
    main()
