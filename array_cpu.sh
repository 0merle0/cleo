#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=8g
#SBATCH -c 1
#SBATCH -o ajob_%j_%a.out
#SBATCH -J ajob
#SBATCH -t 03:00:00
#source activate /software/conda/envs/SE3
# get line number ${SLURM_ARRAY_TASK_ID} from tasks file
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" tasks)
# tell bash to run $CMD
echo "${CMD}" | bash

#sbatch -a 1-$(cat tasks|wc -l) array_cpu.sh
