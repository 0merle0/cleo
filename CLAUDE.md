# CLEO - Combinatorial Libraries for Exploration and Optimization

## What This Is

A framework for protein library design and iterative experimental optimization. It has two main modules:

1. **`cleo.design`** - RL fine-tuning of ProteinMPNN to generate sequences aligned to custom reward functions, then sampling/filtering fragment libraries for experimental testing
2. **`cleo.optimize`** - Training MLP ensemble surrogates on experimental sequence-to-function data and proposing next batches via acquisition function optimization

## Project Structure

```
src/cleo/
  design/
    train_policy.py          # RL training entry point (GRPO or vanillaPG)
    sample_from_policy.py    # Sample sequences from trained checkpoints, split into fragments
    resample_fragments.py    # Recombine fragments into new full-length sequences
    evaluate_sequences.py    # Run sequences through metric pipeline
    dna_utils/               # Reverse translation + Golden Gate adapter design
    utils/
      grpo.py                # GRPO algorithm (PolicyMPNNvGRPO)
      policy.py              # Vanilla REINFORCE (PolicyMPNN)
      reward.py              # UniversalReward - configurable pipeline of metric steps
      oracle.py              # Structure prediction (boltz_from_df, af3_from_df)
      petase.py              # PETase-specific catalytic geometry metrics
      sequence.py            # Hamming distance to reference sequences
      mutation_diversity.py  # Batch mutation diversity reward
      experimental_predictor.py  # Use trained surrogate as a reward step
      geom.py                # Geometry utilities
  optimize/
    train_surrogate.py       # Train ensemble MLP on sequence->activity CSV
    predict_fasta.py         # Score sequences from FASTA with trained model
    batch_optimize.py        # Acquisition function optimization over fragment space
    utils/
      ensemble.py            # MLP ensemble with Gaussian NLL
      fragment.py            # Fragment space utilities
      optimization.py        # BatchUCB + diversity acquisition function
      train_data.py          # Dataset loading (SequenceFunctionDataset, FragmentDataModule)
      pdb_tools.py           # PDB utilities

config/
  design/
    denovo_petase.yaml       # Example: PETase library design with Boltz oracle
    sample.yaml              # Sampling config (checkpoints, fragment_bounds)
    resample_fragments.yaml  # Fragment recombination config
    evaluate.yaml            # Evaluation pipeline config
    dna_fragment_design.yaml # DNA reverse translation config
  optimize/
    base_surrogate.yaml      # Base surrogate training config (inherited by others)
    momi.yaml                # MoMI surrogate config (inherits base_surrogate)
    momi_acqf_opt.yaml       # Acquisition function optimization config
    pred_fasta.yaml          # Prediction config
```

## CLI Commands

All commands use Hydra for configuration. External configs are passed with `--config-path` and `--config-name`.

### Design Pipeline

```bash
# 1. RL training - fine-tune ProteinMPNN with reward functions
cleo-design-train --config-name denovo_petase

# 2. Sample sequences from trained checkpoints, split into fragments
cleo-design-sample --config-name sample

# 3. Recombine fragments into new full-length sequences
cleo-design-resample --config-name resample_fragments

# 4. Evaluate resampled sequences through metric pipeline
cleo-design-evaluate --config-name evaluate

# 5. Reverse translate to DNA with Golden Gate adapters
cleo-design-dna --config-name dna_fragment_design
```

### Optimize Pipeline

```bash
# Train ensemble surrogate on experimental data
cleo-optimize-train --config-name momi data_path=<csv> use_validation=true

# Score sequences from a FASTA
cleo-optimize-predict --config-name pred_fasta

# Acquisition function optimization over fragment space
cleo-optimize-batch --config-name momi_acqf_opt
```

## How to Set Up a New Project

Projects are set up as separate directories that reference (or copy) the cleo codebase. The pattern used in existing projects (momi_final_redesign, glycosidase_design):

### Directory Layout

```
~/projects/<project_name>/
  cleo/                      # Copy of ~/git/cleo (with project-specific utils added)
  configs/
    <project>_design.yaml    # RL training config
    af3_template.json        # AF3 input template for structure prediction
    sample_<project>.yaml    # Sampling config with fragment_bounds
    resample_<project>.yaml  # Resampling config
    evaluate_<project>.yaml  # Evaluation config (same metric steps as training)
  structures/
    <backbone>.pdb           # Input backbone structure
  cleo_runs/                 # Training run outputs (auto-created)
  logs/                      # SLURM logs
  train.submit               # SLURM submission script
```

### What You Need to Provide

1. **Backbone PDB** - the protein structure ProteinMPNN will design on
2. **AF3 template JSON** - AlphaFold3 input template (protein chain + ligand if applicable)
3. **Fixed residues** - catalytic/functional residues that should not be mutated (format: `"A44 A85 A100"`)
4. **Reference sequence(s)** - for computing Hamming distance during training
5. **Fragment bounds** - how to split the protein into fragments for library design (list of `[start, end]` 0-indexed inclusive ranges, typically 4-6 fragments of 20-50 residues)

### Writing a Custom Metrics Module

If your enzyme needs specialized geometric analysis beyond what `petase.py` provides, add a new file to `cleo/design/utils/` in your project's cleo copy. The function signature must be:

```python
def your_metrics_from_df(df_input: pd.DataFrame, cfg: dict, step_name="step") -> pd.DataFrame:
    # cfg gives you access to all fields under the step's `cfg:` in the YAML
    # df_input has columns from prior steps (e.g., af3 prediction paths)
    # Return df_input merged with new metric columns
    # Column names should be prefixed: {step_name}_{metric_name}
```

Then reference it in your config as:
```yaml
- name: my_metrics
  target_fn: cleo.design.utils.my_module.your_metrics_from_df
  cfg:
    ...
```

### Training Config Essentials

Key fields in a design training YAML:

```yaml
run_name: <experiment_name>
output_dir: <path_to_cleo_runs>
pdb: <path_to_backbone.pdb>
checkpoint_path: null           # or path to resume from

algorithm: grpo                 # grpo (recommended) or vanillapg
batch_size: 32                  # sequences sampled per rollout
N_steps: 500                    # RL training steps
lr: 1e-4

# GRPO params
kl_weight: 0.0                 # 0.0 = no KL penalty to reference model
N_updates: 16                  # gradient updates per rollout
update_batch_size: 8            # sub-batch for each update
clip_eps_low: 0.2
clip_eps_high: 0.28

model_type: protein_mpnn        # or ligand_mpnn (if ligand in PDB)
temperature: 1.0                # keep at 1.0 to avoid mode collapse
omit_AA: CX                    # amino acids to never sample
fixed_residues: "A44 A85 ..."   # space-separated chain+resnum

reward:
  _target_: cleo.design.utils.reward.UniversalReward
  steps: [...]                  # ordered list of metric steps
  reward_aggregation: [...]     # weighted normalized sum of metrics
```

### Reward Aggregation

Each metric is normalized to [0,1] using provided bounds, optionally flipped (mode: min), and weighted:

```yaml
reward_aggregation:
  - metric: af3_ptm            # column name from steps output
    lower_bound: 0.0
    upper_bound: 0.8
    weight: 1.0
    mode: max                  # max = higher is better, min = lower is better
```

### SLURM Submission Pattern

```bash
#!/bin/bash
#SBATCH -p gpu-train
#SBATCH --gres=gpu:l40:8
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=64g
#SBATCH -t 2-00:00:00
#SBATCH -o <project>/logs/%x_%j.out

export HYDRA_FULL_ERROR=1
CLEO_DIR=<project>/cleo
PROJECT_ROOT=<project>

cd ${CLEO_DIR}
uv run cleo-design-train \
    --config-path ${PROJECT_ROOT}/configs \
    --config-name <config_name> \
    run_name=<experiment_name> \
    output_dir=${PROJECT_ROOT}/cleo_runs/<experiment_name>
```

## Available Structure Prediction Oracles

- **Boltz** (`cleo.design.utils.oracle.boltz_from_df`) - uses a YAML template, runs CPU or GPU
- **AlphaFold3** (`cleo.design.utils.oracle.af3_from_df`) - uses a JSON template, requires container + model weights

## Available Reward Step Functions

- `cleo.design.utils.oracle.boltz_from_df` / `af3_from_df` - structure prediction
- `cleo.design.utils.petase.add_petase_metrics_to_df` - PETase catalytic triad geometry
- `cleo.design.utils.sequence.compute_dist_to_ref_seqs_from_df` - Hamming distance to references
- `cleo.design.utils.mutation_diversity.mutation_diversity_from_df` - batch mutation diversity
- `cleo.design.utils.experimental_predictor.experimental_predictor_from_df` - trained surrogate as reward

Project-specific steps are added by copying cleo and adding modules to `src/cleo/design/utils/`.

## Dependencies and Environment

- Uses `uv` for dependency management
- `uv sync` for CPU, `uv sync --extra cuda` for GPU
- Key deps: torch, hydra-core, pytorch-lightning, boltz, biopython, prody, biotite, dnachisel
- Python >= 3.10

## Naming Conventions

- Fragment names: `{frag_num}.{unique_id}` (e.g., `1.0003.a7b2c8d1`)
- Resampled sequence names: fragments joined by `___` connector
- Fragment dictionary JSON: keys are string fragment numbers, values are `[[name, seq], ...]`
