# Key results for the white paper

What to show, in order, with the number and where it comes from. Every figure
named here regenerates from `paper/figures/`. Status is one of **solid** (data
in hand, arms matched), **provisional** (direction clear, precision or matching
not yet adequate), **missing** (the panel the story needs and does not have).

Compiled 2026-10-17.

---

## The spine: three claims in sequence

### 1. The recombined library folds, and RL is what makes it fold — **solid**

The headline. Recombine fragments across passing parents and score the
chimeras.

| | library pass % | diversity over passing members |
|---|---:|---:|
| sampled (T=1.0) | 30.5 | 721.7 |
| RL policy | **53.5** | **859.9** |
| backbones won | 8 / 9 | 6 / 8 |

Source: `ame_library_table.py`. Diversity is distinct (position, residue)
substitutions carried by library members that **pass**, rarefied to a matched
count of passing members.

Why those two columns and no third: a low pass rate means mostly dead wells, and
diversity among members that fail is diversity you cannot order. An earlier
per-fragment "survival" column was dropped rather than repaired — see §4.

> The thesis paragraph in `plan_combinatorial.md` still quotes **48% vs 28%**
> from an earlier cut. The current numbers are 53.5 vs 30.5. Update it before
> any text is written from it.

### 2. Fragments from the RL policy are individually better, not just more numerous — **solid**

Pass rate is a property of whole chimeras; this is the per-fragment version, and
it is what rules out "the RL arm just had more parents".

| | per-fragment pass rate | U over survivors (rarefied) |
|---|---:|---:|
| sampled | 0.31 | — |
| RL policy | **0.54** | — |
| backbones won | 8 / 9 | 8 / 8 |

Source: `ame_surviving_fragments.py`. Two supporting facts belong on this panel:

- **Retention holds where the policy loses on parent count.** On M0664 the
  baseline has 2273 passing parents to the policy's 243 — nearly 10× — and its
  chimeras still pass less often.
- **Fragment-space size is 1.00× between arms at matched parent count.** At 45
  residues no two passing designs share a fragment, so library size is set
  entirely by parent count. Retention is the only axis on which the arms differ.

### 3. Granularity: the library scales with k, and the pass rate pays for it — **solid**

| | best k | working constructs at best k |
|---|---:|---:|
| M0097 | 36 | baseline 4.1e50 · policy 1.4e50 |
| M0255 | 36 | baseline 8.6e48 · policy **5.1e49** |
| M0315 | 36 | baseline 2.6e50 · policy **1.4e51** |

Source: `ame_granularity.py`. Best k is 36 (5-residue fragments) for both arms
on all three backbones: chimera pass falls as k rises (M0315 policy 0.86 at k=2
to 0.42 at k=36) but the combinatorial count rises faster. The policy wins the
construct count on 2 of 3 backbones — state it as 2 of 3, not as a sweep.

**This is a post-hoc sweep.** It re-cuts designs that were *trained* at k=4 into
k=2…36. It cannot separate a train/test mismatch from a physical limit on
junction context. E18 trains at each k and is what would.

---

## The methodological result

### 4. The in-loop number overstates, and an oracle pool is what keeps it honest — **provisional**

The reward sits inside a structure predictor, so the training pass rate is not
evidence on its own. Gap = oracle-balanced training level − held-out AF3.

|  | RF3 only | 3-oracle pool |
|---|---:|---:|
| **batch 8** | +0.278 ⚠ warm-started | +0.000 |
| **batch 32** | **+0.058** (running) | **−0.009** |

Source: `ame_transfer_curves.py`; full detail and per-backbone values in
`transfer_gap.md`. Provisional for three reasons, all stated there: M0315 runs
the other way, the batch-32 RF3 arm is at 26–101 of 150 steps, and n = 1 seed
per cell against a documented 21-point run-to-run swing.

The batch-8 RF3 cell is **not a matched baseline** — it warm-starts from a
`fragrl3` checkpoint while every other arm starts fresh. Show it as the arm the
project started from, never as the controlled contrast.

### 5. Two withdrawn results, and why they are worth a paragraph

Both were numbers that looked like controlled comparisons and were not. The
white paper is stronger for saying so once, briefly:

- **Survival fraction.** Counted a fragment as surviving if *any* chimera
  containing it passed. Appearances per fragment are
  (chimeras × slots)/n_fragments, so an arm with fewer parents gives each
  fragment more attempts. The M0078 baseline scored 0.80 survival on a
  per-fragment pass rate of 0.10 — the binomial, not modularity. Replaced by
  per-fragment pass rate (§2); the direction survived, 9/9 became 8/9.
- **The matched-batch oracle claim.** Reported as a one-factor contrast between
  the batch-8 arms. It varies the oracle *and* the initialisation. Caught
  because the RF3 arm reads 0.983 at step 0 on M0315, where the from-scratch
  pool arm reads 0.000–0.047 — an untrained policy does not score 0.983.

---

## Missing: the panel the thesis actually needs

### 6. The current chimera-trained policy is not in the library table — **missing**

`ame_library_table.py`'s third arm, `RL+frag`, is **fragrl2** — an early
fragment-trained generation. It loses to plain RL on library pass (53.6 vs 76.7,
0 of 3 backbones) while carrying more diversity (1311.7 vs 1139.7, 2 of 3).
That is a real result about fragrl2 and it is not the chimera objective the
paper argues for.

The arm the thesis is about — `fragb32`, chimera objective at batch 32, which
completed on all three backbones on 2026-10-17 — has never been evaluated as a
library. **T2, the 256-parent library eval, is what fills this row**, and it is
now unblocked. Until it exists the paper's central claim rests on §1's RL arm,
which was not trained for recombination at all.

---

## Caveats that belong in the text, not in a footnote

- **One seed per cell.** Two unseeded runs of one config gave 68.8% and 89.6%
  control pass — a 21-point swing, larger than most effects here. `seed: 1`
  makes arms comparable and says nothing about sampling error. Replicates, not
  one draw.
- **Oracle scores are not comparable across steps.** One training step is folded
  by one predictor and the three disagree by up to 26 points on identical
  designs. Compare oracle-balanced windows.
- **A quarter of every chimera batch is not a chimera.** `build_chimeras` maps
  (ρ,i) to parent `(i + ρj) mod B`, so ρ=0 returns the unrecombined parent: 32
  of 128 at every k.
- **Only k = 4 was ever trained.** Everything about granularity is post-hoc.
- **Best-of-5 folding is the benchmark's unit.** A single-sample run measures
  something the AME benchmark does not.

---

## Figure inventory

| panel | script | state |
|---|---|---|
| library table | `ame_library_table.py` | current, missing the fragb32 arm |
| surviving fragments | `ame_surviving_fragments.py` | current |
| granularity | `ame_granularity.py` | current |
| transfer curves | `ame_transfer_curves.py` | current; 3 arms, E15 not yet drawn |
| frontier / distributions / PCA / novelty / convergence / efficiency / klsweep | `ame_*.py` | from the benchmark programme; triage before use |
| diversity | `ame_diversity.py` | **broken** — reads `backbone19/eval`, which does not exist |
