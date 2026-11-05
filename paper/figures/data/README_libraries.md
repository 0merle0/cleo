# The two experimental libraries, and where their rounds live

Both are combinatorial fragment libraries with multiple rounds of
design-build-test, on two different enzymes. Written 2026-11-05 while
scoping Figure 3; nothing here is derived, it is an inventory.

## 1. MoMI --- PETase, k = 5

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
