# Key results for the white paper

What to show, in order, with the number and where it comes from. Every figure
named here regenerates from `paper/figures/`. Status is one of **solid** (data
in hand, arms matched), **provisional** (direction clear, precision or matching
not yet adequate), **missing** (the panel the story needs and does not have).

Compiled 2026-10-17. This file is the *numbers*; `plan_combinatorial.md` is the
*outline* (R1-R5, the order the paper argues in) and `README.md` maps the rest
of this directory.

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

## The experimental panel

### 7. A recombined PETase library was ordered and assayed — **solid, and narrower than it looks**

`results/experimental.tex`, figure `petase_zeroshot.py`, data snapshot
`figures/data/petase_activity_zeroshot.csv` (3,155 rows; provenance in
`figures/data/README_petase.md`).

Two claims, both measured, and they are the two the paper needs from wet lab:

**The platform is physically real.** Five fragments (42/44/42/37/44 aa) over a
209-aa de novo PETase scaffold; 25–26 variants per slot = **128 protein parts
spanning 1.1x10^7 recombinants**; 1,920 DNA parts ordered as BsaI blocks,
assembled one-pot, one well per recombinant. **3,135 assayed, 67% registering
above background at a median of 29 mutations from the parent.** This is the only
direct experimental test in the paper of the independence assumption
`results/fragments.tex` names as load-bearing.

**Zero-shot prediction is lightly predictive — on this scaffold only.** Best AF3 metric is local and
state-specific — per-atom pLDDT at the catalytic His of the *acyl-enzyme
intermediate*, rho = +0.60 (n = 2,548); active-site shell +0.58. Global pTM/ipTM
reach only +0.32 to +0.36, so **scoring the right state at the right atoms is
worth roughly 2x the global number**. In practice: select the top 5% by
active-site pLDDT and 28% are top-decile actives against a 10% base rate,
**2.8x enrichment** (1.9x if you keep 25%). Supervised held-out (RidgeCV, 200
splits): AF3 0.73, Rosetta 0.75, both 0.80.

**And it does not transfer.** Added 2026-10-18 from
`adaptyv_library/results/REPORT.md`; snapshot in
`figures/data/petase_super_af3_transfer.csv`. The same two-state fold and metric
pipeline were run on a second de novo serine hydrolase (`super`, 160 aa,
Ser128/His95/Asp53, 2,509 purified designs, 2.51 decades). Correlating each
metric's usefulness on one enzyme against the other gives **rho-of-rhos =
−0.604** over 49 shared AF3 concepts, with only **10/49** agreeing in sign
against ~24 expected by chance. **All ten of PETase's strongest predictors
invert**, mean +0.470 → −0.100; `acyl_his_catN_plddt` goes +0.600 → −0.147.
Rosetta fails more mildly — rho-of-rhos −0.036, 26/48 sign agreement, i.e. no
transferable signal rather than an inverted one — but on the weaker shared
subset only (PETase's best Rosetta metrics are Layer-2 interface/ddG terms with
no `super` counterpart).

What *does* replicate is method orthogonality: predicting Rosetta from AF3 gives
median CV R² 0.145 (PETase) / 0.199 (`super`), against 0.556 / 0.527 the other
way. Rosetta carries private information — mostly relax-ensemble **variance**
terms, which a single structure prediction has no analogue for. Same asymmetry,
same magnitude, both enzymes.

**Why this is good news for the paper.** It is the wet-lab counterpart of §4: a
predictor's agreement with the quantity you care about is a property of the
specific system, re-earned per target, never inherited. If zero-shot scores
transferred, the right strategy would be to rank a virtual set and order the
top. Because they do not, selection has to be done by experiment — which is
exactly what a combinatorially compatible library at a 67% register rate is for.
This is now the strongest argument in the paper for the library strategy.

Two caveats carried from the source, both load-bearing: the `acyl_` state is a
**tetrahedral adduct**, not an acyl-enzyme intermediate (AF3 keeps leaving
oxygens), identical in both screens so the comparison is unaffected; and
`super`'s range is **2.51 decades, not 5.70** — 251 of 2,509 designs sit in a
band 2.56 decades below the live population, cause undetermined, and dropping
them flips only 4/65 signs, none in the top 14.

### The three things this panel must not be made to say

1. **Mutation count beats it.** Distance from the parent ranks activity at
   rho = -0.65, better than any AF3 metric. A zero-shot claim has to clear that
   comparison and this one does not. What survives: restricted to the 2,539
   designs >=20 mutations out, active-site pLDDT still gives +0.59, so the AF3
   signal is not just the distance confound re-expressed.
2. **The 20 designs above 100 are a different assay run.** They score 204–261
   against a screen median of 2.2, and `build_unified_and_tasks.py` says plainly
   they are "the same assay, different runs". No shared normalizing control is
   in the record. Every number that would cross that line is not quoted.
3. **There is no wild-type comparator.** The parent was never assayed here, so
   no fold-over-parent figure can be computed. The archived
   `archive/results_petase.tex` plan's **250x** was always a placeholder and
   cannot be filled from this data.

Within the one screen run, searching the library does move activity: median
2.2 -> 12.2 across four design rounds, best recombinant 20.5 -> 47.8. Round 2
has the lowest median and the best single design, which is what a deliberately
broadened library looks like — report it that way rather than smoothing it.

**The uncomfortable one, which belongs in the text:** the most active designs in
the record sit 1–11 mutations from the parent, and activity falls monotonically
with distance (median 218 at 0–9 mutations, 6.9 at 20–29, 0.2 at 50+). On this
scaffold and this assay, recombination at depth bought reachable diversity and a
67% register rate, not a better enzyme than local work. The paper's claim is
about the diversity/fidelity frontier and physical realizability; it should not
be stretched past that.

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
| PETase zero-shot + rounds + distance control | `petase_zeroshot.py` | current; the only wet-lab panel |
| cross-enzyme transfer (PETase vs `super`) | none yet — table is inline in `results/experimental.tex` | data snapshot current; **no figure script** |
