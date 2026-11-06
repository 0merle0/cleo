# Arm-by-arm comparisons

What has actually been run, and what it supports. **Read this before claiming
the shuffled pipeline is the better method.**

Every number below is the **recombinant pass rate**: 256 recombinants per
cell, folded best-of-5 through AF3, which is held out of training in all of
these arms. That is the "passing success rate on the final shuffled library",
and it is the number to quote — not retention, which is a ratio with an
unstable denominator (it swings 0.767 → 1.611 across replicates of one
config).

## 1. Pooled oracle, with and without shuffling

One factor. All arms pooled `[rf3, boltz, protenix]`, batch 16, seed 1,
same AF3 evaluator, 256 recombinants per cell.

| | arm | M0097 | M0255 | M0315 | mean | steps |
|---|---|---|---|---|---|---|
| **+ shuffle** | `fragb32` | 96.9 | 82.4 | 100.0 | **93.1** | 300 |
| **+ shuffle** | `fragk4` | 85.9 | 85.9 | 98.4 | **90.1** | 200 |
| **+ shuffle** | `fragmulti` | 65.6 | 28.1 | 70.3 | 54.7 | 300 |
| no shuffle | `multirr` | 66.0 | 32.0 | 65.2 | 54.4 | 150 |
| no shuffle | `multi` | 65.2 | 28.5 | 66.0 | 53.2 | 150 |

**Shuffling raises held-out library pass rate by roughly 36 points**, from
~54% to ~90%, on every backbone.

The no-shuffle arms **replicate tightly** — two independent runs at 53.2 and
54.4 — so the baseline is solid, not a single draw.

### `fragmulti` is the outlier, and it is a flagged run

It sits at 54.7, exactly the no-shuffle level, while the other two shuffled
arms sit near 90. All three of its backbones carry a `.truncated` marker
alongside `.converged`, and it logged 249 of 300 steps.

Treat it as a suspect run rather than evidence that shuffling does nothing —
but do not simply delete it either: if truncation is not the explanation,
then shuffling is unreliable run-to-run and the effect is 2/3 rather than
3/3. That distinction is unresolved and matters.

### Caveats

- **The shuffled arms trained longer** (200–300 steps) than the no-shuffle
  arms (150). Part of the 36 points could be budget, and no no-shuffle arm
  has been run to 300 steps to rule that out.
- `fragk4` uses `N_updates: 8` against everything else's 16, so the two
  high-scoring shuffled arms are not replicates of each other either.
- One seed per cell throughout. The no-shuffle pair is the only genuine
  replicate in the table.

## 2. Why these differ from Figure 2's numbers

They are not in conflict — **the arms are different on both sides**, and
neither difference is about seeds.

| | Figure 2 (`panel_vanilla.py`) | §1 above |
|---|---|---|
| baseline | **vanilla ProteinMPNN**, six temperatures pooled | **plain RL**, pooled oracle |
| | 59.8 / 25.8 / 49.6 | 66.0 / 32.0 / 65.2 |
| trained | `fragk4` | `fragmulti` / `fragb32` |
| | 85.9 / 85.9 / 98.4 | 65.6 / 28.1 / 70.3 and 96.9 / 82.4 / 100.0 |

Figure 2 asks "does post-training beat stock ProteinMPNN", so its baseline
is untrained. §1 asks "does shuffling beat plain RL at matched oracle", so
its baseline is already RL-trained and therefore higher (≈54% against ≈45%
for vanilla). Both are legitimate; they answer different questions, and
quoting one as the other overstates or understates the method.

## 3. The published contrast, and why it is weaker than it reads

`key_results.md` §6 rates "the chimera-trained policy clears the plain-RL seed
band" as **solid**, from `fragb32` against plain-RL replicates:

| backbone | plain RL, RF3, seed 2 | seed 3 | `fragb32` chimera+pool | gap over best seed |
|---|---|---|---|---|
| M0097 | 68.0 | 71.1 | **96.9** | +25.8 |
| M0255 | 29.3 | 30.5 | **82.4** | +52.0 |
| M0315 | 65.2 | 57.0 | **100.0** | +34.8 |

Verified from source; the table is accurate. The remaining weakness is that
**it varies objective *and* oracle together** — chimera+pool against
plain-RL+RF3-only — so on its own it cannot attribute the gain to shuffling.
§1 makes the one-factor version at matched oracle and finds a similar effect
(~36 points), which is what rescues this contrast.

§6 also notes "no pool arm has ever been run twice". That is no longer true:
`fragb32`, `fragk4` and `fragmulti` are all pooled chimera arms, and two of
the three agree near 90% while `fragmulti` sits at 54.7%. The claim survives
2/3, with the third flagged `.truncated` — stronger than §6 could say at the
time, and weaker than a clean three-seed replicate.

## 4. Superseded: the fragrl / fragrl2 three-arm table

Kept because `ame_library_table.py` still points at `fragrl2`, which
`key_results.md` §6b flags as the wrong arm. These are RF3-only, not pooled.

| metric | `fragrl` | `fragrl2` | `policy` (no shuffle) |
|---|---|---|---|
| parents | 102 | 101 | **186** |
| retention % | 53.8 | 53.7 | **76.7** |
| surv_frac | 0.86 | 0.85 | 0.86 |
| U_rare | **1359** | **1344** | 1016 |

Shuffling loses on retention 0/3 and wins on fragment diversity 3/3.
`fragrl2` adds `parent_metric` to `fragrl` and changed essentially nothing
(53.8 → 53.7), so that term is unproven rather than load-bearing.

Reproduce: `uv run python paper/figures/archive/ame_three_arm.py`

## 5. What would settle it

The effect looks real and large (~36 points, every backbone), but three gaps
remain:

1. **Matched training budget.** No no-shuffle arm has run past 150 steps,
   while the shuffled arms ran 200–300. Run `multirr` to 300.
2. **Replicates.** One seed per cell, and the one shuffled arm that was
   effectively repeated came out 36 points low. Three seeds for
   `multirr` and one shuffled config would settle both this and whether
   `fragmulti` is a broken run or a real tail.
3. **Resolve `fragmulti`.** It is marked `.truncated` on all three
   backbones. Either confirm truncation explains it and exclude it, or
   accept shuffling as unreliable run-to-run.

Until then the defensible claim is: *shuffling during training raises
held-out library pass rate from ~54% to ~90% on three backbones, in two of
three runs, at a training budget that was not matched.*
