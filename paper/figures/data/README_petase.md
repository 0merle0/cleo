# PETase prediction-vs-activity snapshot

The two files below are snapshots. Their sources live in other projects and are
too large (6.6 MB wide CSV) and too entangled with those projects' paths to read
at figure-build time. Regenerate with `paper/figures/petase_snapshot.py`.

## `petase_activity_zeroshot.csv` -- 3,155 rows, one per assayed design

| column | meaning |
|---|---|
| `seqid` | design id, shared across every source table |
| `activity` | `momi_unified`: per-sequence mean MoMI normalized rate |
| `n_mut_from_parent` | Hamming distance to the 209-aa design parent |
| `momi_round` | design round (0-3), blank for the revalidated block |
| `revalidated` | 1 for the 20 designs re-assayed by Adaptyv after selection |
| `acyl_*`, `substrate_*` | AF3 zero-shot metrics on the acyl-enzyme intermediate and the Michaelis complex |

Sources:

- activity --- `adaptyv_library/rosetta/activity_unified.csv`, built by
  `build_unified_and_tasks.py` from `itopt/kiera/insilico_ssm/260205_all_data_from_jason.csv`
- AF3 metrics --- `itopt/kiera/af3_petase/big_fold/metrics_wide.csv`

**`revalidated` is not a cosmetic flag.** Those 20 designs were measured in a
different run of the same assay, and they are the only designs in the table
above 100. Pooling them with the screen silently compares two runs; every
number in the paper that crosses that line says so.

`n_mut_from_parent` is the distance origin only. The parent sequence itself was
never assayed here, so there is no wild-type activity in this table and no
fold-over-wild-type number can be computed from it.

## `petase_model_cv_scores.csv` -- 600 rows

Held-out scores for RidgeCV over 200 stratified splits, three feature sets
(AF3-only / Rosetta-only / AF3+Rosetta). Copied verbatim from
`adaptyv_library/outputs/rosetta_combined/model_cv_scores.csv`, written by that
project's `rosetta/model_cv.py`.
