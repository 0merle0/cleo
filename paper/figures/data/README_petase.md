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

---

## Cross-enzyme transfer snapshot (added 2026-10-18)

Three files, copied verbatim from
`/home/jgershon/projects/adaptyv_library/results/`, which is where they are
generated (`make_report.py` / `build_results.py` from `RESULTS.json`; the
narrative is `REPORT.md` there). Regenerate at the source, not here.

| file | source name | what |
|---|---|---|
| `petase_super_af3_transfer.csv` | `cross_enzyme_comparison.csv` | 49 AF3 metric concepts, Spearman vs activity on each enzyme |
| `petase_super_rosetta_transfer.csv` | `cross_enzyme_rosetta.csv` | 66 shared Rosetta concepts, same |
| `petase_super_floor_sensitivity.csv` | `super_floor_sensitivity.csv` | every `super` metric with and without the 251-design rate floor |

The second enzyme is `super`, a 160-aa de novo serine hydrolase
(Ser128/His95/Asp53), 2,509 purified designs, substrate
4-methylumbelliferyl butyrate. Same two folded states and the same metric
pipeline as the PETase screen, so the columns are concept-matched rather than
name-matched.

**Why these are in the paper.** `results/experimental.tex` claimed AF3 was
lightly predictive of activity from $\rho = +0.60$ on PETase. That number is
real and it does not transfer: the Spearman-of-Spearmans across the 49 shared
AF3 concepts is **-0.604**, and all ten of PETase's strongest metrics invert on
`super` (mean $+0.470 \rightarrow -0.100$). Only 10 of 49 agree in sign against
~24 expected by chance.

Two caveats carried from the source, both load-bearing:

- The `acyl_` state in both screens is a **tetrahedral adduct**, not an
  acyl-enzyme intermediate -- AF3 strips leaving hydrogens but keeps leaving
  oxygens, so the acid -OH survives and the bonded carbon is 4-coordinate. Both
  enzymes model the same species, so the cross-enzyme comparison is unaffected,
  but the column must not be described as an acyl-enzyme intermediate.
- `super`'s activity range is **2.51 decades, not 5.70**. 251 of 2,509 designs
  (10.0%) sit in a narrow band separated from the live population by 2.56
  decades containing zero designs. Cause undetermined. Dropping all 251 flips
  only 4 of 65 metric signs and none in the top 14, so the transfer verdict
  survives either way (`petase_super_floor_sensitivity.csv`).


### Re-snapshotted 2026-10-18 18:22 after a correction at the source

`cross_enzyme_rosetta.csv` was regenerated and both files were re-copied. The
bug: `concept()` did not bridge PETase's `his_X_dist` names against `super`'s
flip-invariant `his_X_dist_min`, so PETase's single most predictive Rosetta
metric was silently dropped and the shared set was 48 concepts rather than 66.

`super` carries `_min` names because its His95 ring is mislabeled in the input
PDB -- both Ser128-OG and Asp53 contact `NE2` while `ND1` is far from both,
which is impossible for a working triad -- so its readout takes the min over
{ND1, NE2} rather than hardcoding NE2 the way the PETase pipeline does. The
naming that records that fix is what broke the concept match.

What changed: Rosetta $\rho_\rho$ **-0.036 to +0.035**, concepts **48 to 66**,
sign agreement **26/48 to 33/66**. The verdict is unchanged -- still
indistinguishable from zero, still chance-level agreement. The AF3 numbers did
not move at all.

What it retracts: the claim that the Rosetta comparison used only the weaker half
of its range because PETase's best metrics were Layer-2 interface/ddG terms with
no counterpart. The excluded metric was Layer-1 geometry, and the corrected
shared set contains PETase's strongest Rosetta metric at -0.502. Layer-2 is
still absent from `super` (24 PETase metrics, reaching |rho| 0.345) and is still
worth running, but it is not why the comparison looks weak.

Column names also changed: `petase_spearman`/`super_spearman` became
`spearman_petase`/`spearman_super`, and a `state` column was added. Anything
reading these files by column name needs updating.
