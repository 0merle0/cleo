#!/bin/bash

#SBATCH -p gpu-train # cpu / gpu / cpu-bf / gpu-bf
#SBATCH --gres=gpu:l40:2
#SBATCH --mem=128g # cpu memory 
#SBATCH -c 8
#SBATCH -J _0_basecalling_l40_gpu
#SBATCH -t 0-2:00:00 # days-hours:minutes:seconds
#SBATCH -a 1-69 # total tasks/$GROUP_SIZE
#SBATCH -o /home/jq01/wetlab/sequencing/nanopore/240819_isop_lib_subpool3_plate345/240821_sequencing/logs/basecalling_%4a.stdout
#SBATCH -e /home/jq01/wetlab/sequencing/nanopore/240819_isop_lib_subpool3_plate345/240821_sequencing/logs/basecalling_%4a.stderr

GROUP_SIZE=1 # if >1 it will bundles tasks together and run them serially

for I in $(seq 1 $GROUP_SIZE)
do
    J=$(($SLURM_ARRAY_TASK_ID * $GROUP_SIZE + $I - $GROUP_SIZE))
    CMD=$(sed -n "${J}p" /home/jq01/wetlab/sequencing/nanopore/240819_isop_lib_subpool3_plate345/240821_sequencing/cmds/_0_pod2bam_cmds)
    /usr/bin/time -f "Elapse Time: %E, CPU Time,%U, CPU Percentage: %P" bash -c "${CMD}"
    # https://www.cyberciti.biz/faq/unix-linux-time-command-examples-usage-syntax/
    echo "Elapse time format: "hh : mm : seconds.ss", CPU time: "seconds.ss", CPU Percentage: "Need to divide Percentage by 10 to get real pcercentage"" >&2
done
