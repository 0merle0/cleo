#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:a4000:1
#SBATCH -c 1
#SBATCH -o /home/jq01/projects/active_learning/cleo/logs/ajob_%j_%a.out
#SBATCH -J 241213_train_super_esm_vs_esmc
#SBATCH -t 10:00:00

# get line number ${SLURM_ARRAY_TASK_ID} from tasks file
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/jq01/projects/active_learning/cleo/cmds/241213_train_super_esm_vs_esmc_tasks)
# tell bash to run $CMD
echo "${CMD}" | bash
