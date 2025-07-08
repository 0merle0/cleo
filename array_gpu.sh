#!/bin/bash
#SBATCH -p gpu-train
#SBATCH --mem=128g
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH -o logs/ajob_%j_%a.out
#SBATCH -J ajob
#SBATCH -t 2-00:00:00
# get line number ${SLURM_ARRAY_TASK_ID} from tasks file
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" tasks)
# tell bash to run $CMD
echo "${CMD}" | bash

#sbatch -a 1-$(cat tasks|wc -l) array_gpu.sh
