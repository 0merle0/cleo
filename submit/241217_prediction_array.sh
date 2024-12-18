#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=16g
#SBATCH -c 1
#SBATCH -o /home/jq01/git/cleo_jason_working_branch/logs/ajob_%j_%a.out
#SBATCH -J 241217_prediction
#SBATCH -t 10:00:00

# get line number ${SLURM_ARRAY_TASK_ID} from tasks file
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/jq01/git/cleo_jason_working_branch/cmds/241217_prediction_tasks)
# tell bash to run $CMD
echo "${CMD}" | bash
