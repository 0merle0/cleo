# Paper plan: RL for combinatorially compatible libraries

Short working plan. Supersedes `outline_computational.md`, whose headline claim
(policy dominates the diversity/pass-rate frontier) did not survive the
13-backbone panel.

---

## Thesis

> Large enzyme libraries are built combinatorially — split designs into
> fragments, synthesise the parts, assemble the product. That only works if the
> recombinants still fold, and nothing in current design practice optimises for
> it. We show that RL fine-tuning produces designs whose fragments recombine:
> 48% of chimeras from RL-designed parents pass, against 28% from sampled
> parents, on 8 of 9 backbones. Training directly on recombination makes it the
> objective rather than a byproduct.

The unit of a combinatorial library is the **fragment**, not the sequence. This
is a paper about making fragments that travel.

---

## Why this is the right frame

Library size is `P^k` from `k·P` synthesised parts — the product-versus-sum
asymmetry is the entire reason to build libraries this way. But `P^k` is a count
of *constructs*, not of *working* constructs. If recombinants do not fold, the
number is fiction.

Nobody measures this. Every de novo enzyme paper orders ~96 designs chosen by
filtering; the combinatorial step, where it appears, is assumed to work. We
measured it, and it is neither free nor fatal: retention ranges from 0.4% to 95%
depending on backbone and on how the parents were produced.

That range is the opening: **how you design the parents determines whether the
library assembles into anything.**

---

## Results

### R1 — Recombination works, and its cost is measurable  **[measured]**

4 equal slots, exact parents excluded, AF3 best-of-5, 256 chimeras per cell,
21 cells across 13 backbones. Retention spans 0.4–95%. On the easier half it is
high enough that the combinatorial multiplier is real: a few hundred parents
yield ~10⁹ constructs of which most fold.

This is the first measurement of fragment retention for de novo enzyme designs
that we are aware of, and it stands alone regardless of what trained it.

### R2 — RL-designed fragments recombine better  ★ headline  **[measured]**

| dead% | site | baseline | policy |
|---|---|---|---|
| 16 | M0664_2dhn | 90.6 | **94.9** |
| 55 | M0097_1ctt | 59.8 | **72.7** |
| 77 | M0255_1mg5 | 25.8 | **79.3** |
| 85 | M0315_1ey3 | 49.6 | **78.1** |
| 86 | M0500_1e3i | 10.9 | **53.5** |
| 95 | M0058_1cju | 12.9 | **44.9** |
| 97 | M0050_1dbt | **11.7** | 5.5 |
| 97 | M0904_1qgx | 12.9 | **47.3** |
| 99 | M0365_1pfk | 0.4 | **5.5** |

**8 of 9, mean paired advantage +23.0 points.** Mean retention 48.1% against
28.5%.

Nothing in the training objective asked for this. The policy was trained on
motif RMSD alone; modularity came out as a property of the designs.

Two things make the result load-bearing rather than incidental:

- It holds **where the policy loses on parent count**. On M0664 the baseline has
  2,273 passing parents to the policy's 243, nearly 10x, and the policy's
  chimeras still pass more often. So this is not the pass-rate story restated.
- Fragment-space *size* at matched parent count is **exactly** 1.00x between
  arms — with 45-residue slots no two passing designs share a fragment, so
  library size is determined entirely by parent count. Retention is the only
  axis on which the arms differ, and it is the one that decides whether the
  library works.

### R3 — Training directly for recombinability  **[running]**

Fragment-level GRPO: sample 8 parents, never fold them, build 32 chimeras by
rotation so every fragment appears in exactly 4, fold those, and give each
fragment an advantage equal to the mean metric over its chimeras. Per-slot
standardisation; `[B, L]` advantage consumed by the existing per-token surrogate
without changing `grpo.py`.

Running on M0255, M0097; M0315 held pending the first step clearing. Comparison
arms are the panel runs — same configuration, sequence-level reward.

Pre-registered: retention rises above the panel arm on all three. The signal is
diluted — simulated against known fragment effects, averaging recovers the
correct pairwise ordering ~80% of the time and ~67% once fragment interactions
are included — so a modest effect is consistent with signal quality rather than
evidence against the idea.

Kill criterion: fragment collapse. The policy can maximise recombinability by
emitting near-identical sequences. Distinct fragments per slot is tracked as a
diagnostic; if it falls below the panel arm the term must be gated on chimeras
differing from parents, not reweighted. Diversity is explicitly **not** added as
a reward — the weight pilot measured that at every weight from 1 to 4 it drove
pass rate to 0/256 with median motif RMSD 5.9–11.8 Å.

### R4 — How finely should a design be cut?  **[partial]**

Library size is the product over slots of *distinct* fragments, not `P^k`.
Fragments collide as they shorten: at 7 residues one M0097 slot holds 21
distinct variants from 178 parents, and `P^k` overstates the library by 10^15 at
k = 36. Collision is the first signature of degradation and it is measurable
without folding anything.

Retention is the second, and it has not appeared yet. M0097 policy runs
81.2 / 72.7 / 77.3 / 73.4% across k = 2, 4, 6, 8 — flat from k = 4 onward, so
22-residue fragments still recombine. The sweep is extended to k = 12, 16, 24,
36 (fragments of 15, 11, 7, 5 residues) to find where it breaks.

The figure is two panels on one x-axis: retention, and expected working library
(`prod |F_i| x retention`). If retention stays flat while the product grows, the
recommendation is "cut as finely as you can synthesise", which is cleaner than
the tradeoff curve originally expected. If it collapses, the crossing point is
the practical answer to how finely to cut — the question anyone building one of
these libraries actually has.

**Needs the fragment-RL checkpoints** to be a three-arm curve rather than two.
That is the version that goes in the paper.

### R5 — What it buys  **[measured]**

At a matched 2,656-fold budget, library size is `P^k` and synthesis is `k·P`,
so a few hundred parents give ~10⁹ constructs from ~10³ parts. Multiplying
through by retention is what makes that a count of *working* constructs, and it
is where the arms separate: on M0255, 182 parents at 79.3% retention beats 90
parents at 25.8% by far more than the parent ratio alone implies.

---

## Protocol (stated once, applies throughout)

**Train under RoseTTAFold3, evaluate under AlphaFold3**, always. The policy never
sees the predictor it is scored by, so no reported number can be a product of
optimising the metric being reported.

**Budget accounting in designs folded**, split into training and inference, and
written per run. Cost is dominated by the trunk, which runs once per sequence;
best-of-5 shares that trunk pass and is close to free, so designs — not
predictions — are the unit.

**Baseline is pooled temperature sampling** at a matched budget (2,656 folds
across T = 0.1–1.0), which reaches more distinct substitutions than any single
temperature and is therefore the strongest version of the comparison.

**Run-to-run σ = 7.75 points** (E21, 10 runs), resolution floor 9.6 points at
n = 5. Stated up front; contrasts are reported against it.

---

## What is cut, and why

- **The diversity/pass-rate frontier claim.** Measured across 13 backbones at
  matched budget, the policy wins both axes on 6/13 and loses on the easy end
  where sampling already works. It does not survive and should not be argued.
- **Rescue.** Clean rescue (baseline 0 → policy > 0) is 1/13, with a
  counterexample beside it (M0078: policy 0, baseline 11). The earlier
  "0/10,000" figures came from T = 1.0 alone; pooled sampling is much stronger.
- **PETase, heme, protease.** Wet lab or out of scope.
- **The selection-rule arc** (E9/E16/E18/E19). Ends unresolved; one paragraph in
  Discussion.

---

## Honest limits

- **One seed per backbone** in the retention panel. With σ = 7.75 on pass rate,
  single draws are not reliable for small contrasts — though a +23-point mean
  advantage on 8/9 is well clear of it.
- **Advantage does not clearly grow with difficulty.** On four backbones it
  looked like it did (+4.3 → +53.5); across nine the correlation is r = 0.23,
  which is weak. Do not claim the trend.
- **Retention falls for both arms as backbones get harder** — from ~90% at 16%
  dead to ~5% at 99%. The policy's advantage persists but the absolute numbers
  on hard backbones are low enough that a library there may not be worth
  assembling.
- **Four slots only.** At 45 residues fragments never collide between arms
  (0.0% overlap), which makes library size pure parent count. Shorter fragments
  would create shared parts and could behave differently; untested.
- **M0050 is a genuine loss** (5.5% vs 11.7%), on 6 policy parents against 18
  baseline. Low-parent cells are where this is weakest.
