# Parked figure scripts

Everything here was exploration that the paper no longer builds from. It is
parked rather than deleted: the analyses are real and several answer questions
that are still open. `git log --follow <file>` still works.

Nothing here is imported by a live figure. The live set was computed as the
transitive closure of local imports from the figures the paper actually
builds, so this split is mechanical rather than a judgement call:

    concept_library  fig2_libraries  fig3_momi  figS_posthoc
    momi_fragments   momi_libraries_snapshot   panel_cost
    + what they import: ame_fragk, concept_loop, panel_vanilla,
      figio, palette, schematic

Note `ame_fragk` is LIVE despite the name -- Figure 2 and Figure S1 both
import `collect_posthoc` and `add_diversity` from it.

## Worth knowing before reviving any of these

- **`ame_three_arm.py` contradicts the Figure 2 framing.** It reports fragrl
  beating the policy arm on 0/3 backbones (53.8% vs 76.7%). Figure 2 rests on
  trained-at-k vs post-hoc instead, which sidesteps the disagreement rather
  than resolving it. Anyone leaning on a three-arm claim has to settle this
  first.
- **`ame_frontier.py` and `ame_efficiency.py` do not run.** IndexError and an
  AttributeError on `rfd2_any_pass` respectively -- the columns they expect
  moved. They are parked broken, not parked working.
- **`petase_snapshot.py` is superseded** by `momi_libraries_snapshot.py`,
  which captures BOTH libraries and all rounds. The old one silently lost
  round labels on 519 designs (16%).
- **Everything here was authored 9.6-15.5 in wide**, against a 6.5 in page.
  Their labels land at 3.4-4.7 pt when placed at `\textwidth`. Any revival is
  a re-authoring job at page width, not a drop-in.

## Also here

- `svg/`, `pdf/` -- built output of the parked scripts.
- `png_stale/` -- four preview rasters from an old Inkscape-conversion flow.
  The live PyMOL renders are in `../structures/png/` and are untouched.
- the `*_rl_framing.tex`, `heme_*`, `protease_*`, `*_petase.tex` files are
  prose from earlier framings of the paper.
