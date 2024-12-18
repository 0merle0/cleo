#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:a4000:1
#SBATCH -c 1
#SBATCH -o /home/jq01/projects/active_learning/cleo/logs/ajob_%j_%a.out
#SBATCH -J 241209_train_frag_lr1e-4_batchsize_sweep
#SBATCH -t 10:00:00

# get line number ${SLURM_ARRAY_TASK_ID} from tasks file
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/jq01/projects/active_learning/cleo/cmds/241209_batchsize_lr1e-4_frag_tasks)
# tell bash to run $CMD
echo "${CMD}" | bash

#sbatch -a 1-$(cat  /home/jq01/projects/active_learning/cleo/cmds/241209_batchsize_lr1e-4_frag_tasks|wc -l) /home/jq01/projects/active_learning/cleo/submit/241209_train_frag_lr1e-4_batchsize_sweep_array.sh
