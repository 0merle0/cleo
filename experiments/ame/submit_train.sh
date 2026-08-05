#!/bin/bash
# P0.5 pilot: GRPO fine-tune LigandMPNN on one AME backbone against the
# published motif-RMSD criterion.
#
#   sbatch experiments/ame/submit_train.sh <backbone-stem> [N_steps] [batch_size]
#
# Folds dominate the cost: N_steps x batch_size AF3 calls. Sharded across the
# GPUs allocated here.
#SBATCH -p gpu
#SBATCH --gres=gpu:small:8
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=96g
#SBATCH -t 12:00:00
#SBATCH -J ame_train
#SBATCH -o /home/jgershon/git/cleo/experiments/ame/logs/%x_%j.out

set -euo pipefail

CLEO=/home/jgershon/git/cleo
AME=${CLEO}/experiments/ame
BB=${1:?usage: submit_train.sh <backbone-stem> [N_steps] [batch_size]}
STEPS=${2:-40}
BATCH=${3:-16}

mkdir -p "${AME}/logs"
cd "${CLEO}"

export HYDRA_FULL_ERROR=1

uv run cleo-design-train \
    --config-path "${AME}/configs" \
    --config-name "${BB}" \
    N_steps="${STEPS}" \
    batch_size="${BATCH}"
