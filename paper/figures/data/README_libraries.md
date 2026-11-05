# The two experimental libraries, and where their rounds live

Both are combinatorial fragment libraries with multiple rounds of
design-build-test, on two different enzymes. Written 2026-11-05 while
scoping Figure 3; nothing here is derived, it is an inventory.

## 0. Snapshot: `momi_libraries.csv`

**Both** PETase libraries, all four rounds each, 6,714 rows. Written by
`momi_libraries_snapshot.py`. This is the file figures should read.

The copy in `data/momi/` is the FIRST library only, 3,424 rows and no
`library_name` column, which is why the second library looked missing. The
only table that distinguishes them is jq01's combined training set for the
second library's round 3, under
`jq_p07_active_learning/jq_e0325_260123_..._rd3_analysis/2_model_training/input/`.

Two columns are derived at snapshot time so no figure re-parses a name:
`selection` (the tag after the colon) and `parent_slots` (how many of the
k=5 slots still hold a `g20` parent part).

### What the two libraries did

| | round 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **first**, n | 1,665 | 708 | 674 | 352 |
| median x G20 | 0.17 | 0.27 | 0.03 | 0.90 |
| max | 2.19 | 2.41 | **3.35** | 2.49 |
| frac > G20 | 3.8% | 9.2% | 12.2% | 41.5% |
| **second**, n | 1,791 | 736 | 367 | 369 |
| median x G20 | 0.83 | 0.36 | 5.54 | **8.06** |
| max | 5.00 | 7.36 | 17.97 | **18.98** |
| frac > G20 | 38.7% | 32.1% | 92.9% | **99.7%** |

The second library is a different campaign, not a continuation: it starts
from a round 0 already ten times better by hit rate (38.7% above G20 against
3.8%) and ends with 368 of 369 designs beating G20 at a median of 8x.

Selection tags differ between them, so they are not comparable arm for arm:

| | round 1 | round 2 | round 3 |
|---|---|---|---|
| first | `bin1`..`bin6`, `top2combos` | `MLPgamma1/5/8`, `af3Only`, `boltzOnly`, `modelhubOnly`, `allOracles` | `sampledtopkMLP`, `topkMLP` |
| second | `bin1`..`bin6`, `top2combos` | `mostdiverseMuGt3`, `top30mu` | `mostdiverseMuGt4`, `top10mu` |

Their parts come from different pools too: library 1 from `refw5`/`refw8` RL
checkpoints, library 2 from `dist3`..`dist9`.

**Unresolved.** jq01's round-3 README says *"Best variant from this round is
30x better than G20"*, but the best `g20_norm_rate` in this table is 18.98.
The 30x is presumably a different measurement -- kinetics rather than the
plate screen. Do not quote either number as the other until that is settled.

## 1. MoMI --- PETase, k = 5 (first library)

209-aa design parent. Names encode the assembly directly:

    1.g20___2.g20___3.refw8.step0150.0042.66b5e39e___4.g20___5.g20:bin3
    \---- one part per slot, `g20` = the parent part ----/  \-tag-/

so the number of non-`g20` slots is how far a design is from the parent, and
the tag after the colon is **how that design was selected**.

In repo: `data/momi/*.csv`, four CUMULATIVE snapshots. The last one,
`251225_..._round0_round1_round2_round3_with_val.csv`, contains everything.

| round | n | selection tags | mean parent slots kept (of 5) |
|---|---|---|---|
| 0 | 1,678 | untagged (baseline) | 4.01 |
| 1 | 714 | `bin1`..`bin6`, `top2combos` | 1.56 |
| 2 | 680 | `MLPgamma1/5/8`, `af3Only`, `boltzOnly`, `modelhubOnly`, `allOracles` | 0.62 |
| 3 | 352 | `sampledtopkMLP`, `topkMLP` | 1.88 |

Activity columns: `momi_norm_rate` (1,561/1,678 non-null in round 0, complete
after), `g20_norm_rate`, `z_score_norm_rate`. `val` marks a held-out 20%.
25 rows are `positive_control`.

Round 2 is an exploration round and the numbers say so: it keeps the fewest
parent slots (0.62), its median activity COLLAPSES to 0.42 against round 0's
2.18, and its maximum is the highest of any round at 47.8. Round 3 pulls
back to 1.88 parent slots and every design beats 1.

## 2. super --- de novo serine hydrolase, k = 4

160 aa, Ser128/His95/Asp53, substrate 4-methylumbelliferyl butyrate. Names
are the same shape with four slots:

    frag1_temp0.2num054:frag2_temp0.2num092:frag3_temp0.5num065:frag4_parent_4

Parts are indexed by the sampling temperature they were drawn at.

**Only round 0 is in this repo** (`data/super/`): the 2,509 purified designs,
plus `241217_super_pool_and_evaluated.csv`, which is the full 324,000-member
combinatorial space with 2,509 marked `has_been_evaluated`.

Rounds 1-3 live outside, in
`/home/jgershon/projects/itopt/super/super_acqf_comparison/`:

| round | arm | n | file |
|---|---|---|---|
| 1 | greedy | 90 | `round_1_data/250617_random_round_1_greedy_data.csv` |
| 1 | gamma15 | 111 | `..._gamma15_data.csv` |
| 1 | gamma30 | 108 | `..._gamma30_data.csv` |
| 2 | greedy | 100 | `round_2_data/250620_random_round_2_greedy_data.csv` |
| 2 | gamma15 | 108 | `..._gamma15_data.csv` |
| 2 | gamma30 | 104 | `..._gamma30_data.csv` |
| 3 | -- | 256 | `round_3_data/processed_data/5_complete_df.csv` |

**The three arms are independent lineages, not three labels on one set.**
Each round-2 file carries a different number of round-1 designs (greedy 39,
gamma15 63, gamma30 85), because each acquisition function ran its own round
1 and then proposed its own round 2 from it. Pooling them would destroy the
comparison.

Round 3's processed file is raw plate data (`rate_norm_mean`, triplicate
wells) and carries no `acqf` column; the arm would have to come from
`round_3_data/mappings/`.

## What this means for a figure

Both libraries have the same shape of story --- a combinatorial library,
several rounds, and an explicit comparison of how the next batch was chosen
(`bin*` / `MLPgamma*` / `topkMLP` on PETase, `greedy` / `gamma15` / `gamma30`
on super). That is the acquisition-function experiment, run twice on two
enzymes.

Before anything is plotted from `super` rounds 1-3 it has to be snapshotted
into this directory, the way `petase_snapshot.py` does. See
`README_petase.md` for why that matters: the committed PETase snapshot went
stale and silently lost 519 round labels.
