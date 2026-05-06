# Example antibody RL fine-tuning configs

Reference configs and SLURM submission scripts for two VHH affinity-maturation
campaigns. Both use GRPO fine-tuning of ProteinMPNN with an AF3 oracle and
include the consensus-novelty + distance-to-reference rewards
(`mutation_diversity_marginal_count_vs_consensus`, `dist_to_ref_seqs_min`).

## Campaigns

- `gdf8_vhh/` — GDF8 VHH (128aa VHH + 143aa GDF8 target). Distance-to-ref
  weight = 2.0; expect a moderate-mutation library.
- `ltk_vhh/` — LTK VHH (123aa VHH + 297aa LTK target). Distance-to-ref
  weight = 2.0.

Each subdirectory contains:
- `*.yaml` — Hydra training config. Defines fixed residues (CDRs designable,
  framework + target chain frozen), reward steps, and aggregation weights.
- `*.pdb` — input backbone (VHH + target chain).
- `af3_template_*.json` — AlphaFold3 input template for the oracle.
- `train_*.submit` — SLURM submission script (8×L40, 2-day wall, 64G).

## What you need to edit before running

`pdb`, `template_path`, and AF3 container/script/model paths are already
correct for the shared cluster — should run as-is. If you adapt these
configs to a new target, update:

1. **In the `.submit` script**: `CLEO_DIR` if your clone lives elsewhere.
2. **In the `.yaml` config**: `pdb` and `template_path` for your target
   structure / AF3 template; `fixed_residues` (everything except your CDRs);
   and `ref_seqs` / `ref_seq` under `dist_to_ref_seqs` and
   `mutation_diversity` (your parent sequence).

## Running

```bash
cd /path/to/cleo
sbatch examples/gdf8_vhh/train_gdf8_lep_distw2_consensus.submit
# or
sbatch examples/ltk_vhh/train_ltk_distw2_consensus.submit
```

See top-level `CLAUDE.md` for the broader workflow (sampling, fragment
recombination, evaluation, DNA design).
