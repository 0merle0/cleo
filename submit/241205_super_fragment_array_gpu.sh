#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:a4000:1
#SBATCH -c 1
#SBATCH -o /home/jq01/projects/active_learning/cleo/logs/ajob_%j_%a.out
#SBATCH -J 241205_JQ_super_fragment_array_training
#SBATCH -t 10:00:00

# get line number ${SLURM_ARRAY_TASK_ID} from tasks file
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/jq01/projects/active_learning/cleo/cmds/241205_super_fragment_hidden_tasks)
# tell bash to run $CMD
echo "${CMD}" | bash

#sbatch -a 1-$(cat  /home/jq01/projects/active_learning/cleo/cmds/241205_super_fragment_hidden_tasks|wc -l) /home/jq01/projects/active_learning/cleo/submit/241205_super_fragment_array_gpu.sh
