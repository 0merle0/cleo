# Running things

House preferences for actually executing work here. `TRAINING_PROTOCOL.md` is
the procedure; this is the operational detail around it.

## Environment

`uv` manages dependencies. **Prefix every command with `uv run`** — the bare
interpreter is not the project environment and will fail on imports or, worse,
pick up a different package version.

    uv sync                 # CPU
    uv sync --extra cuda    # GPU

A trap worth knowing: the Debian `pymol` wrapper calls `python3`, which under
`uv run` resolves to the venv, where there is no pymol module. The structure
scripts strip `VIRTUAL_ENV` before shelling out for exactly this reason.

## Training

    uv run cleo-design-train                            # current pipeline
    uv run cleo-design-train --config-name pooled_oracle
    uv run cleo-design-train --config-name denovo_petase

With no `--config-name` this runs `fragment_shuffle`: pooled oracles, AF3 held
out, fragment shuffling. Anything older is an explicit opt-in.

### Always set a seed on anything you will compare

`seed:` is unset by default, which reproduces historical behaviour and leaves
runs nondeterministic. Two unseeded runs of one config moved the **control**
by 21 points. Budget replicates; a single run is not a result.

### Costs, before you queue it

Folds per step are `rotations × batch × samples`. At 4 × 16 × 5 that is **320
folds/step**, so 200 steps is **64,000 folds**. RF3 measured on M0097 at 5
samples: ~9 s per design plus ~100 s model load per call, so batch 32 is
~6.5 min/step and a 150-step run is ~16 GPU-h (AF3: ~11.5 GPU-h for the same
run). Batch in one call — RF3 distributes over GPUs itself.

### SLURM

`--gres=gpu:large:1` works here. `gpu:l40:8` is **not** satisfiable and sbatch
rejects it with "Requested node configuration is not available". Set
`HYDRA_FULL_ERROR=1` in the submit script; Hydra's default traceback hides the
actual exception.

If the client reports `ClusterName needs to be specified`, the node has lost
`/etc/slurm/slurm.conf` — copy it from `digs.ipd` into `~/.slurm` and point
`SLURM_CONF` at it.

## Where output lands

    <output_dir>/<run_name>_config.yaml          the resolved config, as run
    <output_dir>/<run_name>_train_metrics.csv    per-step log
    <output_dir>/<run_name>_last.pt              checkpoint
    <output_dir>/<run_name>_step_NNNN.pt         every checkpoint_every_n_steps
    <output_dir>/<run_name>/outputs/step_NNNN/   per-step metrics.csv + folds

Checkpoint selection defaults to `checkpoint_metric: reward`, mode `max`.
`FragmentReward` overrides `__call__` and never runs the base aggregation, so
a `checkpoint_metric` copied from a standard run dies at the first checkpoint
— the keys it does log are named explicitly in that class.

## Before you trust a number

These have each produced a wrong conclusion here at least once.

- **A resumed run restarts at step 0 into the same `outputs/` tree**, so it
  overwrites its own step history. Check mtime monotonicity before reading
  per-step numbers as a time series.
- **`eval/` holds results; `*/_folds/` does not.** Classify any prune or
  archive candidate by what a tree *contains*, never by its family name —
  `eval/` trees are read by the figure scripts.
- **A per-step score is one oracle's opinion.** Compare oracle-balanced
  windows of `len(oracles) × m` steps; `fold_oracle` records each draw.
- **Check `checkpoint_path` before calling two arms matched.** One arm
  warm-starting from another's checkpoint while the rest start fresh is a
  confound that looks like a result.
- **AF3 must be absent from training** for any number to be held out.
  `multi_oracle_from_df` warns, but a warning in a 10,000-line log is easy to
  miss — grep for it.

## Evaluation

    uv run cleo-design-evaluate --config-name evaluate

Folds with **AF3 only**, best-of-5, reduced to one row per sequence.
Evaluations produced before AF3 became the evaluator folded with Boltz, a
training oracle, and are not comparable with ones produced after.

The AF3 site paths in that config (`af3_container`, `af3_script`,
`model_dir`) are IPD-specific. Change those three lines elsewhere.

## Git

Commit or push only when asked. The working branch here is `paper`; `main` is
the default branch for PRs.
