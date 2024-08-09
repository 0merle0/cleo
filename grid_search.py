import sys, os
import secrets
import json
from cleo import CLEO
import argparse
from model_util import train_gp, evaluate_gp
import torch
import gpytorch
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--fragment_csv', type=str, default=None)
    parser.add_argument('--train_data',type=str, required=True)
    parser.add_argument('--training_iter', type=int, required=True)
    parser.add_argument('--hidden_dimension', type=int, required=True)
    parser.add_argument('--embedding_dimension', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--out_path', type=str, default='./')
    parser.add_argument('--input_column',type=str, default='seq')
    parser.add_argument('--label_column',type=str, default='norm_activity')
    parser.add_argument('--k',type=int, default=10)
    parser.add_argument('--random_split', action='store_true', default=False)
    parser.add_argument('--one_hot', action='store_true', default=False)
    parser.add_argument('--fragment_representation', action='store_true', default=False)
    parser.add_argument('--vocab_size', type=int, default=20)

    return parser.parse_args()


def grid_search_train(train_params):

    model, loss_list = train_gp(train_params=train_params)

    # evaluate train and test data
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if args.one_hot:
            etest = model.posterior(test_set.featurized_seqs.reshape(N_test_seqs,-1).float())
            etrain = model.posterior(train_set.featurized_seqs.reshape(N_train_seqs,-1).float())  
        else:
            etest = model.posterior(test_set.featurized_seqs)
            etrain = model.posterior(train_set.featurized_seqs)

    meta_data = evaluate_gp(model, train_set, test_set=test_set, one_hot=args.one_hot, save_fig=f'{out}.png', show_plot=False)

    data_save = {
        'training_iter':training_iter,
        'one_hot':args.one_hot,
        'fragment_representation':args.fragment_representation,
        'random_split':args.random_split,
        'loss':loss_list[-1],
        'embedding_dim':embedding_dim,
        'hidden_dim':hidden_dim,
        'learning_rate':lr,
        'run_id':run_id,
        'train_spearmanr':meta_data['train_spearmanr'],
        'test_spearmanr':meta_data['test_spearmanr'],
    }

    with open(f'{out}.json','w') as f:
        json.dump(data_save,f)

if __name__ == "__main__":
    
    args = get_args()
    
    # grid_search_setup
    training_iter = args.training_iter
    hidden_dim = args.hidden_dimension
    embedding_dim = args.embedding_dimension
    lr = args.learning_rate
    
    out_folder = os.path.join(args.out_path,'grid_search/')
    os.makedirs(out_folder, exist_ok=True)
    
    cleo = CLEO(args.fragment_csv, data_csvs=args.train_data)

    train_set, test_set = cleo.get_train_and_test_sets(
                                                       one_hot=args.one_hot,
                                                       input_col=args.input_column,
                                                       label_col=args.label_column,
                                                       k=args.k,
                                                       random_split=args.random_split,
                                                       fragment_representation=args.fragment_representation
                                                      )
    
    # get datasets
    if args.one_hot:
        N_test_seqs,aa_L,aa_dim = test_set.featurized_seqs.shape
        N_train_seqs,aa_L,aa_dim = train_set.featurized_seqs.shape
        train_x = train_set.featurized_seqs.reshape(N_train_seqs,-1).float()
        test_x = test_set.featurized_seqs.reshape(N_test_seqs,-1).float()
        
        embedding_dim = None
        hidden_dims = [aa_L*aa_dim,256,hidden_dim]
    
    else:
        N_test_seqs,aa_L = test_set.featurized_seqs.shape
        N_train_seqs,aa_L = train_set.featurized_seqs.shape
        train_x = train_set.featurized_seqs
        test_x = test_set.featurized_seqs
        
        hidden_dims = [aa_L*embedding_dim,hidden_dim]

    train_y = train_set.activities
    test_y = test_set.activities

    # set model_params
    L = train_x.shape[1]
    vocab_size = args.vocab_size # set at 20 for the amino acids ok

    run_id = secrets.token_hex(4).upper()
    out = os.path.join(out_folder, f'model_{run_id}')

    
    # define likelihood for sampling
    train_params = {
        'train_x':train_x, 
        'train_y':train_y, 
        'vocab_size':vocab_size, 
        'embedding_dim':embedding_dim, 
        'hidden_dims':hidden_dims,
        'learning_rate':lr,
        'training_iter':training_iter,
    }


    # run it
    print(f'now training {run_id}')
    grid_search_train(train_params)
    print(f'done {run_id}')
    
