#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=16g
#SBATCH --gres=gpu:a4000:1
#SBATCH -c 1
#SBATCH -o logs/ajob_%j_%a.out
#SBATCH -J ajob
#SBATCH -t 08:00:00
# get line number ${SLURM_ARRAY_TASK_ID} from tasks file
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" tasks)
# tell bash to run $CMD
echo "${CMD}" | bash

#sbatch -a 1-$(cat tasks|wc -l) array_gpu.sh
