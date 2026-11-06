# Arm-by-arm comparisons

What has actually been run, and what it shows. Reproduce with:

    uv run python paper/figures/archive/ame_three_arm.py

## Does fragment shuffling beat plain RL fine-tuning?

**On the evidence we have: not on retention, and clearly yes on diversity.**

Four arms, three backbones (the only three with every arm):

| arm | what it is | shuffling | `parent_metric` |
|---|---|---|---|
| `baseline` | vanilla MPNN samples, no fine-tuning | — | — |
| `policy` | GRPO on sequence motif-RMSD | **no** | n/a |
| `fragrl` | GRPO on fragment scores from chimeras | **yes** | **unset** |
| `fragrl2` | same, with the parent-fold term | **yes** | `ame_motif_rmsd` |

| backbone | arm | parents | retention % | surv_frac | U_rare |
|---|---|---|---|---|---|
| M0097 | policy | 178 | **72.7** | 0.8 | 1166 |
| M0097 | fragrl | 115 | 69.9 | 0.9 | **1739** |
| M0097 | fragrl2 | 68 | 66.8 | 0.8 | 1721 |
| M0255 | policy | 182 | **79.3** | 0.9 | 569 |
| M0255 | fragrl | 28 | 30.5 | 0.9 | **812** |
| M0255 | fragrl2 | 20 | 29.7 | 0.9 | 806 |
| M0315 | policy | 199 | **78.1** | 0.9 | 1315 |
| M0315 | fragrl | 163 | 60.9 | 0.7 | **1525** |
| M0315 | fragrl2 | **215** | 64.5 | 0.8 | 1504 |

Summary against the no-shuffling `policy` arm:

| metric | `fragrl` | `fragrl2` | `policy` | wins |
|---|---|---|---|---|
| parents | 102 | 101 | **186** | shuffling 0/3, 1/3 |
| retention % | 53.8 | 53.7 | **76.7** | shuffling **0/3** |
| surv_frac | 0.86 | 0.85 | 0.86 | tie |
| U_rare | **1359** | **1344** | 1016 | shuffling **3/3** |

### What this says

1. **Shuffling loses on retention, 0/3, by 23 points.** That is the metric it
   is designed to improve, so this is the uncomfortable result and should not
   be smoothed over. M0255 is the extreme: 30% against 79%.
2. **Shuffling produces about a third fewer passing parents.** Training the
   fragments does cost the sequences.
3. **Shuffling wins decisively on fragment diversity, 3/3, +33%.** With
   `surv_frac` tied, that is the signature of genuine modularity rather than
   collapse — a policy that satisfied recombinability by emitting near-copies
   would show `surv_frac` climbing while `U_rare` fell, and the opposite
   happens.

So the honest reading is a **trade**, not a win: shuffling buys a more diverse
pool of fragments that survive recombination, and pays for it in parent yield
and in retention. Whether that trade is worth it depends on whether the
library is limited by diversity or by hit rate.

### `parent_metric` did not rescue it

`fragrl2` is `fragrl` plus the parent-fold term, which the protocol recommends
on the grounds that omitting it collapses parent quality. Across these three
backbones it changed **essentially nothing**: retention 53.8 → 53.7, parents
102 → 101, U_rare 1359 → 1344. It helped on M0315 (163 → 215 parents) and hurt
on the other two.

The earlier "28 passing parents against 182" framing compared `fragrl` on
M0255 against `policy`, and reads as though the parent term is the missing
piece. It is not, on this evidence — `fragrl2` scored 20 on that backbone.

### Before leaning on any of this

- **n = 3 backbones, single seed (seed 1).** Unseeded run-to-run variance on
  this benchmark has been measured at 21 points of pass rate, and two of the
  three retention gaps (M0097, 5.9 points) are well inside that.
- **`fragrl2` trained 300 steps against `policy`'s 150**, so it had more
  compute and still lost on retention. Matched on batch (16), lr (1e-4),
  seed (1), algorithm (GRPO) and oracle (RF3 only).
- **These arms are single-oracle (RF3), not pooled.** They predate the pooled
  protocol, so AF3 is genuinely held out for all of them, but none of them is
  the pipeline `fragment_shuffle.yaml` now runs.
- **The `policy` arm computes a mutation-diversity metric** but does not
  reward it — its aggregation is motif RMSD alone.

### The comparison that does not exist yet

**Nothing has been run with the current default**: pooled oracles plus
shuffling. Every shuffling arm here is RF3-only. That is the experiment to run
before the paper claims the pooled+shuffled pipeline is the better method.
