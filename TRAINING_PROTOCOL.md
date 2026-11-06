# Training protocol (current)

This is the pipeline as it actually runs today. Earlier configs in `config/`
still work and record what was run at the time, but start here.

**These are the defaults.** `cleo-design-train` with no `--config-name` runs
`fragment_shuffle`: pooled training oracles with AF3 held out, and fragment
shuffling. Older single-oracle configs are an explicit opt-in.

Three worked examples, all runnable on files tracked in this repo:

| config | what it rewards | use it when |
|---|---|---|
| `config/design/pooled_oracle.yaml` | each **sequence** for folding | you want a better single design |
| `config/design/fragment_shuffle.yaml` | each **fragment** for folding in company it has never met | you are building a combinatorial library — **the default** |
| `config/design/denovo_petase.yaml` | PETase catalytic geometry, pooled oracle | the PETase target |

`denovo_petase_single_oracle.yaml` is the pre-pool original. It is kept as a
record of what was run and **no longer runs as-is**: it folds best-of-5 with no
reducing step, which the `UniversalReward` guard now rejects.

```bash
uv run cleo-design-train --config-name fragment_shuffle
```

Change the paths marked `CHANGE ME` to point at your target. See
`config/design/POOLED_ORACLE.md` for the pooled-oracle details.

---

## 1. Pool the training oracles, hold AF3 out

Training against one structure predictor buys performance that does not
transfer: the policy learns that predictor's quirks rather than how to build
the protein. On M0097, 256 designs, best-of-5:

| | gap (train oracle − AF3) |
|---|---|
| sequences no policy optimised | −3.9 |
| RF3-trained, sequence reward | **+18.0** |
| RF3-trained, fragment reward | **+36.3** |

The predictors agree within 4 points on designs nobody optimised, and disagree
by 22–40 points on RF3-trained policies. That excess exists only under the
training oracle.

So: **train on a pool of `[rf3, boltz, protenix]`, report on AF3.** One oracle
is drawn per rollout by `multi_oracle_from_df`. AF3 never enters the objective.

Three things that are easy to get wrong:

- **`schedule: round_robin`, not `random`.** Round robin is
  `oracles[train_step % len(oracles)]`, uniform by construction. The
  seed-arithmetic `random` draw is not: over a real 115-step run it gave
  43/32/25 against the intended thirds, leaning toward whichever oracle sits
  first, reproducibly across seeds.
- **One group is folded by ONE oracle, never a mixture.** GRPO's advantage is
  group-relative, so every design in a group must be scored on the same
  yardstick or part of each advantage becomes a comparison between predictors
  rather than between designs.
- **`use_avg_reward: false`.** The within-batch advantage cancels a whole-batch
  oracle offset. A running-history baseline would compare this step's oracle
  against earlier steps' and inject the ~26-point disagreement straight in.

## 2. Shuffle during training

A combinatorial library does not ask "does this sequence fold?" It asks "does
this fragment still work next to fragments it has never met?" Those are
different questions, and optimising the first does not get you the second.

`chimera_metrics_from_df` is the only step whose folded set is not the set it
was given. It takes the batch's `B` sequences, cuts each into `k` slots, and
rebuilds `rotations × B` **chimeras**: chimera `(ρ, i)` takes slot `j` from
parent `(i + ρ·j) mod B`. Each fragment therefore appears in exactly
`rotations` different backgrounds. A fragment's score is the mean outcome of
the chimeras containing it, and `FragmentReward` hands every position inside
slot `j` that fragment's score — a per-token advantage, not one scalar per
sequence.

Scores are standardised **within slot**, so a slot that is intrinsically harder
does not simply become a lower reward.

**Keep `parent_metric` set.** With chimera quality as the only objective the
policy optimises fragments that travel while the sequences it emits degrade:
28 passing parents against 182 for the arm trained on parent RMSD alone, and
retention fell with them. The fragment term says "your parts travel"; the
parent term says "and you fold". It costs about 25% more trunk passes.

**Cost.** `rotations × batch × samples` folds per step. At 4 × 16 × 5 that is
320 folds/step, so a 200-step run is **64,000 folds**. Budget for it.

## 3. Probe with the held-out oracle during the run

Set `af3_probe_every`. It folds the same chimeras under AF3 on a stride and
writes one column; it touches neither the score nor the advantage.

This exists because the training-oracle number stopped tracking transfer:
M0315 climbed 96.4 → 99.6 on RF3 between generations while its AF3 chimera
pass went 66.8 → 65.6. Without the probe that is invisible until a full
train-then-evaluate cycle finishes. At `every: 25` it costs ~1% of a run.

`af3_probe_cfg` must carry `af3_container`, `af3_script` and `model_dir`.
`oracle_cfg` is the *training* oracle's config and has none of them; when they
were missing the probe raised on every stride and the column stayed NaN — a
broken diagnostic that looks like a quiet one.

## 4. Reduce best-of-N — mandatory, and only implemented for motif metrics

Every oracle defaults to `num_diffusion_samples: 5` with
`per_sample_rows: true`, because best-of-5 is the benchmark's own unit. A
reward step must return one row per sampled sequence or the reward tensor will
not match the batch, so every pipeline needs a `best_of_n_from_df` step.
`UniversalReward` refuses to build one without it and names the block to add.

The reduction is over the benchmark's criterion, not AF3's confidence: min
RMSD over samples, and an OR over the per-sample `pass AND no-clash`
conjunction. Reducing the two columns separately would let a design pass on
the strength of two different predictions.

**`best_of_n_from_df` only knows the rfd2/AME motif columns**
(`{prefix}_motif_rmsd`, `{prefix}_motif_pass_and_no_clash`). A target scored on
anything else — PETase catalytic geometry, for instance — has nothing for it to
reduce, so those configs set `num_diffusion_samples: 1` and run single-sample.
Writing a reducer for other metric families is the obvious next step.

## 5. Ligands

All four oracles translate the **same AF3 JSON template**, so pass the `.json`
and no oracle can fold a different complex than the others.

- **CCD codes where they exist** — `{"ligand": {"id": "B", "ccdCodes": ["ZN"]}}`.
  RF3's docs rank them above SMILES for accuracy.
- **SMILES where they do not** — `{"ligand": {"id": "B", "smiles": "..."}}`.
  Needed for things like the PETase PET dimer, which has no CCD code.
  `templates/oracle/petase_af3_template.json` is a worked example.

One entity carries one or the other, never both. Prefer CCD: Protenix embeds a
SMILES with RDKit and gives up after 90 s on molecules it cannot handle.

## 6. Reading oracle columns under a pool

Every column the drawn oracle produced is renamed to the step's name, so a
reward reads `fold_iptm`, not `boltz_iptm`, and does not know which predictor
ran. `fold_oracle` records the draw.

Only **`path`, `sample`, `iptm`, `ptm`** are common to all three. Boltz alone
has `ligand_iptm`, `protein_iptm`, `complex_*`; RF3 and Protenix alone have
`plddt`, `has_clash`, `ranking_score`, `chain_ptm_*`. A reward term on a field
only one oracle emits would silently vanish for two thirds of a pooled run —
compute it downstream from `{step}_path` instead.

## 7. Seed anything you intend to compare

`seed:` is unset by default, which leaves runs nondeterministic. Two unseeded
runs of one config gave control pass rates of **68.8% and 89.6%** — a 21-point
swing, larger than most effects these experiments are built to detect. A
single-run A/B at n≈96 cannot resolve anything smaller than that, so budget
replicates rather than trusting one draw.

## 8. Reading the output

- **A per-step score is not comparable across steps.** One step is one oracle
  and the three disagree by up to 26 points on identical designs. Compare
  oracle-balanced windows of `len(oracles) × m` consecutive steps. The drawn
  predictor is recorded per row in `fold_oracle`.
- **`frag_chimera_pass` is the training oracle; `frag_chimera_pass_af3` is the
  held-out one.** Report the second. If the two diverge, that is the transfer
  gap, not noise.
- **A resumed run restarts at step 0 into the same `outputs/` tree**, so
  per-step history can be overwritten. Check mtime monotonicity before
  trusting it.
