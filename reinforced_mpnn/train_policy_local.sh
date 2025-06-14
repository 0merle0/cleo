export MASTER_PORT=50067

export PYTHONPATH=$PYTHONPATH:/home/jgershon/git/cleo/reinforced_mpnn/:/home/jgershon/git/cleo/
export HYDRA_FULL_ERROR=1

# apptainer -s run --nv /software/containers/mlfold.sif train_policy.py -cn penicillin_vanilla_pg

source activate newlatent; python train_policy.py -cn heme_vanilla_pg