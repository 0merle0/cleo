import sys, os
import secrets
import json
from cleo import CLEO
import argparse
import torch
import gpytorch
import botorch
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from botorch.acquisition import qUpperConfidenceBound
from model_util import load_gp
from optimization_util import *


def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--fragment_csv',type=str, required=True)
    parser.add_argument('--model_path',type=str, required=True)
    parser.add_argument('--out_path', type=str, default='./')
    parser.add_argument('--acquisition_function', type=str, default='qUCB') 
    parser.add_argument('--optimization_strategy', type=str, default='policy')
    parser.add_argument('--N', type=int, default=24)
    parser.add_argument('--q', type=int, default=8)
    parser.add_argument('--qUCB_beta', type=float, default=0.5)
    parser.add_argument('--num_iter', type=int, default=1500)
    parser.add_argument('--step_size', type=float, default=0.05)
    parser.add_argument('--one_hot', action='store_true', default=False)
    parser.add_argument('--fragment_representation', action='store_true', default=False)
    parser.add_argument('--num_replicates', type=int, default=10)

    return parser.parse_args()

def optimize(optimization_strategy, cleo, acqf, N, q, num_iter, step_size, one_hot, fragment_representation, out_path):
    '''
        optimize acqf using the specified strategy and save the output sequences
    '''

    if optimization_strategy == 'policy':
        # optimizing using policy grad, RL style
        policy = policy_optimize_acquisition_function(acqf, cleo.fragment_dictionary, N=N, q=q, num_iter=num_iter, step_size=step_size, one_hot=one_hot)

        candidate_seqs = get_candidates_from_policy(policy, cleo.fragment_dictionary)

    elif optimization_strategy == 'sequence':
        # optimizing in sequence space directly
        optimized_seqs = optimize_acquisition_function(acqf, cleo.all_seqs, N=N, q=q, num_iter=num_iter, step_size=step_size)

        candidate_seqs, min_distances = round_to_nearest_sequences(optimized_seqs, cleo.all_seqs)
        
    elif optimization_strategy == 'fragment':
        # optimizing in fragment space directly
        optimized_seqs = optimize_acquisition_function(acqf, cleo.all_seqs, N=N, q=q, num_iter=num_iter, step_size=step_size,
                                              fragment_representation=fragment_representation, fragment_dictionary=cleo.fragment_dictionary)

        candidate_seqs = get_candidates_from_fragment_opt(optimized_seqs, cleo.fragment_dictionary)

    else:
        sys.exit(f'optimization strategy is not available: {optimization_strategy}')
    
    datas = {
            'strategy':optimization_strategy,
            'candidate_seqs':candidate_seqs,
            'N':N,
            'q':q,
            'num_iter':num_iter,
            'step_size':step_size,
            'one_hot':one_hot,
            'fragment_representation':fragment_representation,
            }

    with open(out_path, 'w') as f:
        json.dump(datas,f,indent=4)


# add hydra decorator with config
if __name__ == "__main__":
    
    args = get_args()

    cleo = CLEO(fragment_csv=args.fragment_csv)
    
    model = load_gp(args.model_path)
    
    if args.acquisition_function == 'qUCB':
        acqf = qUpperConfidenceBound(model, args.qUCB_beta)
    else:
        sys.exit('acquisition function is not defined')

    
    run_id = secrets.token_hex(4)
    out_path = os.path.join(args.out_path, run_id+'.json')

    print(f'now optimizing {run_id} {args.optimization_strategy}')

    optimize(
            args.optimization_strategy, 
            cleo, 
            acqf, 
            args.N, 
            args.q, 
            args.num_iter, 
            args.step_size, 
            args.one_hot, 
            args.fragment_representation, 
            out_path
            )

    print(f'done {run_id}')
