#!/usr/bin/env bash
# Vanilla ProteinMPNN (bundled weights) on the **same reference complexes** as checkpoint_sample_eval:
#   - GDF8: lepionce_run10_packed41_complex.pdb  (VHH + GDF8; CDR masks from gdf8 train yaml)
#   - LTK:  ltk_vhh97_complex_HA.pdb             (VHH + LTK; CDR masks from ltk train yaml)
# Two different PDBs (one per target); each matches the finetuned runs for that campaign.
# After sampling, the Slurm worker also runs evaluate_sequences (same reward steps as training).
# This bash script only samples; for full parity use the bundle + sbatch, or run the worker per task_id.
#
# Run from repo cleo/ (local sampling only, sequential):
#   bash scripts/run_vanilla_mpnn_baselines.sh
#
# Cluster (sample + AF3 / full eval, same as checkpoint_sample_eval):
#   uv run python -m cleo.design.run_vanilla_mpnn_slurm slurm_bundle_dir=slurm_vanilla_mpnn_bundle
#   sbatch slurm_vanilla_mpnn_bundle/submit_array.sh
#
# Layout: cleo_runs/vanilla_mpnn_baseline/<run_name>/{<run_name>.fasta,evaluation.csv,...}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CLEO_ROOT}/.." && pwd)"

NUM_BATCHES="${NUM_BATCHES:-4}"
OUT_ROOT="${CLEO_ROOT}/cleo_runs/vanilla_mpnn_baseline"
mkdir -p "${OUT_ROOT}"

# Must match configs/gdf8_vhh/*.yaml and configs/ltk_vhh/ltk_vhh_design.yaml (and checkpoint_sample_eval).
PDB_GDF8="${REPO_ROOT}/data/gdf8_vhh/lepionce_run10_packed41_complex.pdb"
PDB_LTK="${REPO_ROOT}/data/ltk_vhh/ltk_vhh97_complex_HA.pdb"

run_one() {
  local train_yaml="$1"
  local short="$2" # gdf8 | ltk
  local temp="$3"
  local pdb_path="$4"
  local run_name="${short}_T${temp}_vanilla"
  local run_dir="${OUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}"
  echo "=== ${run_name}  pdb=${pdb_path}  -> ${run_dir}/${run_name}.fasta ==="
  cd "${CLEO_ROOT}"
  uv run python -m cleo.design.sample_from_policy \
    baseline_train_config="${train_yaml}" \
    pdb="${pdb_path}" \
    temperature="${temp}" \
    num_batches="${NUM_BATCHES}" \
    fragment_bounds=null \
    batch_size=null \
    output_dir="${run_dir}" \
    output_name="${run_name}"
}

GDF8_W0="${REPO_ROOT}/configs/gdf8_vhh/gdf8_vhh_lep_run10_distw0.yaml"
GDF8_W1="${REPO_ROOT}/configs/gdf8_vhh/gdf8_vhh_lep_run10_distw1.yaml"
LTK="${REPO_ROOT}/configs/ltk_vhh/ltk_vhh_design.yaml"

for T in 0.1 0.2; do
  run_one "${GDF8_W0}" "gdf8" "${T}" "${PDB_GDF8}"
  run_one "${LTK}" "ltk" "${T}" "${PDB_LTK}"
  # Same GDF8 backbone as w0; only RL reward weights differ in training.
  # run_one "${GDF8_W1}" "gdf8" "${T}" "${PDB_GDF8}"
done

echo "Done. FASTAs under ${OUT_ROOT}/<run_name>/ (eval: use Slurm bundle or run evaluate_sequences on each FASTA)."
