# Figure plan — framing the computational process

Written 2026-11-03, after the fragment-count sweep (E18/E18b) finished. The
point of this document is to decide what the computational half of the paper
*shows*, before any more prose is written against figures that may not survive.

## 1. Inventory: what exists today

24 figure scripts, 28 built SVGs. The paper references 16 of them.

**Built and referenced** (16): `panel_c1`–`panel_c10`, `panel_cost_k`,
`panel_cost_size`, `petase_zeroshot`, `ame_klsweep`, `ame_pca`, `ame_pca_drift`,
`ame_distributions`, `ame_convergence`, `ame_bsweep_frontier`,
`ame_bsweep_backbones`.

**Built but referenced nowhere** (10):

| figure | what it shows | verdict |
|---|---|---|
| `ame_fragk` | trained-at-k vs cut-at-k retention, k=2..64 | **promote — headline** |
| `ame_fragk_diversity` | reach / collision / pairwise / substitutions vs k | promote |
| `ame_fragk_curves` | per-step histories, all 18 cells | supplementary |
| `ame_granularity` | post-hoc re-cut of a k=4 policy | fold into `ame_fragk` |
| `ame_frontier` | diversity/fidelity across easy/medium/hard | promote |
| `ame_efficiency` | folds per N passing designs | promote (cost) |
| `ame_novelty` | mutations CLEO reaches that no temperature does | supplementary |
| `ame_transfer_curves` | held-out AF3 vs training level | supplementary |
| `ame_bsweep_thresholds` | batch size vs threshold | cut |
| `ame_bsweep_transfer` | batch size vs transfer | cut |

**Written but never built** (7): `ame_three_arm`, `ame_library_table`,
`ame_fragments`, `ame_surviving_fragments`, `ame_fragment_scores`,
`ame_diversity`, `ame_selection_bench`. Several of these answer questions the
text already asks — `ame_three_arm` in particular is the sampled / RL-trained /
trained-for-recombination contrast, which is the method's own claim.

## 2. The computational spine

The argument the figures have to carry, in order. Each step is a question the
previous step forces.

1. **You can make far more candidates than you can order.** A 200-aa gene is
   \$42 at eBlock pricing; a million unique sequences is \$42M, the same million
   as a k=6 combinatorial library is under \$500. `S = (B/3Lc)^k`.
2. **You cannot rank them either.** Best zero-shot AF3 metric reaches ρ=+0.60 on
   PETase, and the ten strongest predictors invert on a second serine hydrolase
   (ρ-of-ρs = −0.604). So selecting the best million is not an option.
3. **So build combinatorially — but recombinants must fold.** This is the whole
   problem the method exists to solve, and the hinge of the paper.
4. **Train the decoder so its fragments are combinable.** GRPO on the MPNN
   decoder, chimeras built in-loop, score averaged back onto each fragment.
5. **It works, and reaches a corner temperature cannot.** Diversity/fidelity
   frontier, KL anchor with an interior optimum.
6. **Recombinants do fold, and training for it is what makes them fold.**
7. **It scales to 11-residue fragments and then stops.** The k sweep.
8. **What it costs** — in folds, not steps.
9. **Wet lab.** (Deliberately after all of the above.)

Steps 1, 2, 5 and 9 have figures. **Steps 3, 4, 6 and 7 are the gaps**, and 4 is
the worst of them.

## 3. The missing figure: the method schematic

**Nothing in the paper draws the training loop.** There is no panel showing:

```
backbone --> MPNN decoder as policy --> B sequences
                                          |
                                          v
                              split into k equal fragments
                                          |
                                          v
              build_chimeras: rotations=4, (rho,i) -> parent (i + rho*j) mod B
                                          |
                                          v
                        fold best-of-5  (rf3 / boltz / protenix)
                                          |
                                          v
                     score = motif RMSD + clash + confidence
                                          |
                                          v
          average each fragment's score over every chimera containing it
                                          |
                                          v
                        GRPO update, per-fragment advantage
```

Every quantitative claim downstream is unreadable without this, and three
details that referees will ask about only exist in prose or in `CLAUDE.md`:

- the ρ=0 rotation returns the **parent itself** — 32 of every 128 chimeras —
  so the in-loop chimera pass is inflated and is NOT comparable to T2 retention;
- the fold is **best-of-5**, which is the AME benchmark's own unit;
- `FragmentReward.k` and `chimera_metrics_from_df.cfg.k` are **two separate
  fields that must agree**, and disagreement produces a plausible run answering
  the wrong question.

This is the first thing to make, and it is a drawn schematic, not a plot.

## 4. Proposed main figures

Six, with everything else supplementary.

**Fig 1 — The problem.** (a) cost vs library size, `panel_cost_size`;
(b) cost vs k, `panel_cost_k`; (c) ranking failure, `petase_zeroshot`.
*Claim: you can neither afford nor rank your way out; combinatorial is forced.*
Status: all three exist and are already used.

**Fig 2 — The method.** The schematic above, plus one panel of a training curve
showing the chimera pass rising from 0 to ~0.99 (`ame_fragk_curves`, one
backbone, one k).
*Claim: fragment-level credit assignment over in-loop recombinants.*
Status: **schematic does not exist.** Curve panel exists.

**Fig 3 — The policy reaches a corner temperature cannot.** `ame_frontier`
(easy/medium/hard) + `ame_klsweep` (interior optimum) + `ame_pca` (where in
sequence space).
*Claim: this is not just more sampling.*
Status: all built; `ame_frontier` unused, the other two already used.

**Fig 4 — Training for recombination is what makes fragments combinable.**
`ame_three_arm` (sampled / RL / RL-for-recombination) + the post-hoc decline
from `ame_granularity`.
*Claim: the method's central contrast, currently drawn nowhere.*
Status: **`ame_three_arm` has never been built.** Highest-value build.

**Fig 5 — How far recombination scales.** `ame_fragk` (retention trained-at-k vs
cut-at-k, with the gap panel) + `ame_fragk_diversity`.
*Claim: flat through 11-residue fragments, cliff below; the gap over a
k=4-trained policy widens to +32.2 pts; diversity rises as yield falls.*
Status: **built, current, unused.** This is the strongest new result in the
paper and it is not in the paper.

**Fig 6 — What it costs.** `ame_efficiency` (folds per passing design) +
`ame_bsweep_frontier`.
*Claim: the bill is folds, and batch 16 beats batch 32 on folds-to-threshold.*
Status: both built, `ame_efficiency` unused.

## 5. Stale claims to fix before writing against them

- `methods/method.tex`, plan block (ii): *"Only k=4 was ever trained. Everything
  in this paper about granularity re-cuts k=4 designs post hoc."* **False since
  E18.** Six k levels are trained on three backbones.
- `results/fragments.tex`, Block 3: *"Do recombinants fold? (the experiment that
  must be run)"* — it has been run, twice over (T2 recombination, then the full
  k sweep). The heading and its enumerated plan should become results.
- `transfer_gap.md:70` and `experiments_plan.md:825` both still say E18 is
  "staged, not launched."

## 6. Caveats the figures must not hide

- **M0255 at k=36/64 was still climbing at step 200**, so its 40.6% and 10.9%
  are lower bounds. Budget is fixed at 200 steps on purpose: letting it vary
  with k would reintroduce the confound the sweep exists to remove.
- **Every cell is single-seeded**, against a documented 21-point run-to-run
  swing on pass rate. Nothing here resolves an effect smaller than that.
- **In-loop and T2 numbers are not comparable** (parents included vs excluded;
  training oracles vs AF3). Any figure putting them on one axis is wrong.
