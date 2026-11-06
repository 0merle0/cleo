# Running the pooled-oracle reward

    uv run cleo-design-train --config-name pooled_oracle

`config/design/pooled_oracle.yaml` is a complete worked example that runs on
files tracked in this repo. Change the four paths marked `CHANGE ME` to point
at your own target.

## Why pool oracles

Training against one structure predictor buys performance that does not
transfer -- the policy learns that predictor's quirks rather than how to build
the protein. Measured on M0097, 256 designs, best-of-5:

| | RF3 | AF3 | gap |
|---|---|---|---|
| sequences no policy optimised | 36.3% | 40.2% | -3.9 |
| RF3-trained, sequence reward | | | **+18.0** |
| RF3-trained, fragment reward | | | **+36.3** |

The predictors agree within 4 points on designs nobody optimised. On
RF3-trained policies they disagree by 22-40 points. That excess exists only
under the training oracle.

`multi_oracle_from_df` removes the fixed target by drawing ONE predictor per
rollout from a pool. AF3 is kept out of training entirely and used only to
report, so the reported number is a held-out measurement.

## The five things that are easy to get wrong

1. **One template feeds all the oracles.** Boltz, RF3 and Protenix all
   translate the same AF3 JSON. Pass the `.json`, not Boltz's own `.yaml` --
   that way no oracle can fold a different complex than the others. Container
   paths are already defaulted in `oracle.py`; set them only if your site
   differs.

2. **You must reduce best-of-N back to one row.** Every oracle defaults to
   `num_diffusion_samples: 5` with `per_sample_rows: true`, because best-of-5
   is the benchmark's unit. A reward step has to return one row per sampled
   sequence or the reward tensor will not match the batch, so the pipeline
   needs a `best_of_n_from_df` step. `UniversalReward` refuses to build a
   pipeline without it and names the block to add.

3. **Use `schedule: round_robin`, not `random`.** Round robin is
   `oracles[train_step % len(oracles)]` -- exactly uniform by construction.
   The seed-arithmetic `random` draw is not: over a real 115-step run it gave
   43/32/25 against the intended thirds, leaning toward whichever oracle sits
   first, reproducibly across seeds. It is a property of the sequence, not
   noise.

4. **Keep `use_avg_reward: false`.** GRPO's advantage is
   `(r - mean) / (std + 1e-3)` computed *within* the batch, which cancels a
   whole-batch oracle offset. A running-history baseline would compare this
   step's oracle against earlier steps' oracles and inject the ~26-point
   disagreement between predictors straight into the advantage.

5. **Set `early_stopping_plddt_threshold: 0.0` for RF3.** RF3 defaults to 0.5
   and abandons predictions it judges unpromising, writing confidences with no
   structure. Those are exactly the failing designs the reward needs to see;
   left on, they drop out of the batch and bias reward toward what RF3 already
   liked. `rf3_from_df` disables it by default and warns on any row it finds.

## Reading the results

A per-step score is **not comparable across steps** -- one step is one oracle,
and the three disagree by up to 26 points on identical designs. Compare
oracle-balanced windows of `len(oracles) * m` consecutive steps. Each row
records which predictor scored it in `fold_oracle`.

Also set `seed:`. Two unseeded runs of one config gave control pass rates of
68.8% and 89.6% -- a 21-point swing, larger than most effects these runs are
built to detect.

## Known limitation: SMILES ligands

The AF3-JSON path handles ligands as **CCD codes only**; there is no SMILES
support in `oracle.py`. The PETase work uses a PET dimer given as SMILES,
which exists only in the Boltz YAML template, so **the pool cannot currently
run on that target** -- RF3 and Protenix have no way to receive the ligand.
The pool works on ligand-free targets and on any target whose ligands have CCD
codes, which includes the whole AME benchmark. Adding SMILES to the AF3-JSON
translators would lift this.
