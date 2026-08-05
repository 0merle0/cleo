#!/bin/bash
# P0 gate: sample baseline LigandMPNN sequences, fold with AF3, score against the
# RFdiffusion2 benchmark criteria. Proves every reported number computes
# end-to-end before any training compute is committed.
#
#   sbatch experiments/ame/submit_pilot.sh <backbone-stem> [n_seqs]
#
# Defaults to one backbone x 8 sequences -- deliberately small; this is a wiring
# test, not a measurement.
#SBATCH -p gpu
#SBATCH --gres=gpu:small:1
#SBATCH -c 8
#SBATCH --mem=48g
#SBATCH -t 4:00:00
#SBATCH -J ame_pilot
#SBATCH -o /home/jgershon/git/cleo/experiments/ame/logs/%x_%j.out

set -euo pipefail

CLEO=/home/jgershon/git/cleo
AME=${CLEO}/experiments/ame
BB=${1:?usage: submit_pilot.sh <backbone-stem> [n_seqs] [side_chain_context]}
N=${2:-8}
SC=${3:-1}

mkdir -p "${AME}/logs" "${AME}/baseline"
cd "${CLEO}"

SEQS="${AME}/baseline/${BB}_n${N}_sc${SC}.csv"

uv run python "${AME}/sample_baseline.py" \
    --pdb "${AME}/targets/${BB}-atomized-bb-True.pdb" \
    --n "${N}" --temperature 0.1 --side-chain-context "${SC}" --out "${SEQS}"

uv run python "${AME}/fold_and_score.py" --seqs "${SEQS}"
