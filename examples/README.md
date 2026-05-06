# Example antibody RL fine-tuning configs

Reference configs and SLURM submission scripts for two VHH affinity-maturation
campaigns. Both use GRPO fine-tuning of ProteinMPNN with an AF3 oracle and
include the consensus-novelty + distance-to-reference rewards
(`mutation_diversity_marginal_count_vs_consensus`, `dist_to_ref_seqs_min`).

## Campaigns

- `gdf8_vhh/` — GDF8 VHH (123aa VHH + GDF8 target). Reward set tuned for
  ~5 mutations/seq vs the parent.
- `ltk_vhh/` — LTK VHH (123aa VHH + 297aa LTK target). Reward set tuned for
  ~8 mutations/seq vs the parent.

Each subdirectory contains:
- `*.yaml` — Hydra training config. Defines fixed residues (CDRs designable,
  framework + target chain frozen), reward steps, and aggregation weights.
- `af3_template_*.json` — AlphaFold3 input template for the oracle.
- `train_*.submit` — SLURM submission script (8×L40, 2-day wall, 64G).

## What you need to edit before running

The configs and submit scripts have absolute paths from the original project
(`/home/jgershon/projects/antibody_opt/...`). At minimum, update:

1. **In the `.submit` script**: `CLEO_DIR`, `PROJECT_ROOT`, `EXP_DIR`,
   `--config-path`, output paths.
2. **In the `.yaml` config**: `output_dir`, `pdb`, the `template_path` and
   `af3_container` / `af3_script` / `model_dir` entries under the `af3` reward
   step.
3. **`fixed_residues`**: list every residue you do NOT want to design over
   (everything except your CDRs and any residue on the target chain you want
   to allow mutation on — by default the target chain is fixed).
4. **`ref_seqs`** under both `dist_to_ref_seqs` and `mutation_diversity` steps:
   replace with your parent VHH sequence.

## Running

```bash
sbatch examples/gdf8_vhh/train_gdf8_lep_distw8_consensus.submit
# or
sbatch examples/ltk_vhh/train_ltk_distw8_consensus.submit
```

See top-level `CLAUDE.md` for the broader workflow (sampling, fragment
recombination, evaluation, DNA design).
