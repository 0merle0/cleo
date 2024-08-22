import glob
import torch
import secrets
import scipy
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import gpytorch
from botorch.acquisition import qUpperConfidenceBound
import pdb_util
from model_util import FragmentDataset, train_gp
from optimization_util import *
from icecream import ic

aa2num = pdb_util.aa12num
num2aa = pdb_util.num2aa1


@hydra.main(version_base=None, config_name="config")
def iterative_opt(cfg: DictConfig) -> None:
    '''
        Big optimization loop with using dataset.
    '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')

    raw_enz_df = pd.read_csv(cfg.dataset_path)
    enz_df = raw_enz_df[raw_enz_df['# Stop'] == 0]

    # get datasets to test
    start_csvs = glob.glob('/home/jgershon/Desktop/BO/arnold_data/*.csv')
    uids = []
    for x in start_csvs:
        uids.append(x.split('/')[-1].split('_pos')[0])
    uids = list(set(uids))

    topk_df = enz_df.sort_values('fitness',ascending=False)[:cfg.top_k_from_dataset]
    topk_dataset = FragmentDataset(topk_df, input_col='AAs', label_col='fitness')


    # make fragment dictionary
    fragment_dictionary = {i+1:[(x,x) for x in num2aa] for i in range(4)}

    # BO settings
    num_rounds = cfg.iterative_opt.num_rounds # 30
    num_restarts = cfg.iterative_opt.num_restarts # 3
    qUCB_beta = cfg.candidate_selection.qUCB_beta # 0.5
    N = cfg.candidate_selection.num_restarts # 16
    q = cfg.candidate_selection.batch_size # 100
    num_iter_acqf = cfg.candidate_selection.num_iter_acqf # 500
    step_size = cfg.candidate_selection.step_size # 0.05

    # surrogate train settings
    training_iter = cfg.surrogate.training_iter # 250
    lr = cfg.surrogate.lr # 0.001
    input_dim = cfg.surrogate.input_dim # 80 # fixed for right now
    hidden_dim = cfg.surrogate.hidden_dim # 32
    
    for restart in range(num_restarts):


        # neet to add logic here for selecting starting dataset

        matches = glob.glob(f'/home/jgershon/Desktop/BO/arnold_data/*{uid}_pos_start*')
        matches.sort()

        start_df = pd.read_csv(matches[0])
        pool_df = pd.read_csv(matches[1])

        train_df = pd.DataFrame({'seq':start_df['AAs'].tolist(),'fitness':start_df['fitness'].tolist()})

        run_meta_data = {
            'round':[],
            'train_spearmanr':[],
            'top1k_spearmanr':[],
            'top_candidate':[],
            'start_dataset':[],
            'test_run_id':[],
        }

        seq_meta_data = {
            'round':[],
            'seq':[],
            'fitness':[],
        }


        # BO ROUND STARTS
        for round_ in range(num_rounds):

            max_fitness = train_df['fitness'].max()
            print(f'round {round_+1} : max seen {max_fitness}')
            run_meta_data['round'].append(round_+1)
            run_meta_data['top_candidate'].append(max_fitness)
            run_meta_data['test_run_id'].append(restart)

            # first train model

            dataset = FragmentDataset(train_df, input_col='seq', label_col='fitness')

            train_x = dataset.featurized_seqs
            train_y = dataset.activities

            input_dim = dataset.featurized_seqs.shape[-1]

            train_params = {
                'train_x':train_x, 
                'train_y':train_y,
                'input_dim':input_dim,
                'hidden_dim':hidden_dim,
                'learning_rate':lr,
                'training_iter':training_iter,
            }

            model, loss_list = train_gp(train_params=train_params,device=device)

            # get train spearmanr
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                etrain = model.posterior(train_x.to(device))
            train_spearmanr = scipy.stats.spearmanr(train_y, etrain.mean.cpu())[0]

            # get top1k spearmanr
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                etopk = model.posterior(topk_dataset.featurized_seqs.to(device))
            top1k_spearmanr = scipy.stats.spearmanr(topk_dataset.activities, etopk.mean.cpu())[0]

            run_meta_data['train_spearmanr'].append(train_spearmanr)
            run_meta_data['top1k_spearmanr'].append(top1k_spearmanr)


            # optimize acqf
            acqf = qUpperConfidenceBound(model, qUCB_beta)

            policy = policy_optimize_acquisition_function(acqf, fragment_dictionary, N=N, q=q, num_iter=num_iter_acqf, step_size=step_size, device=device)

            candidate_seqs = get_candidates_from_policy(policy, fragment_dictionary)

            # add seqs to dataframe
            to_add = {
                'seq':[],
                'fitness':[],
            }
            for c in candidate_seqs:
                if c[1] not in train_df['seq'].tolist():
                    tmp = pool_df[pool_df['AAs']==c[1]]
                    if len(tmp) == 0:
                        fitness = 0
                    else:
                        fitness = tmp['fitness'].tolist()[0]
                    to_add['seq'].append(c[1])
                    to_add['fitness'].append(fitness)
                    seq_meta_data['round'].append(round_+1)
                    seq_meta_data['seq'].append(c[1])
                    seq_meta_data['fitness'].append(fitness)

            to_add_df = pd.DataFrame(to_add)

            train_df = pd.concat([train_df,to_add_df])


        # will need to edit this to change to take in custom path
        out = f'/home/jgershon/Desktop/BO/arnold_data/run_metadata/long30round/dataset_{uid}_test_run_{restart}'

        run_meta_data_df = pd.DataFrame(run_meta_data)
        run_meta_data_df.to_csv(f'{out}_run_meta_data.csv')

        seq_meta_data_df = pd.DataFrame(seq_meta_data)
        seq_meta_data_df.to_csv(f'{out}_seq_meta_data.csv')

if __name__ == "__main__":
    iterative_opt()