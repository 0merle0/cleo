import sys, os
import glob
import random
import subprocess
import torch
import secrets
import scipy
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm
import json

sys.path.append('/home/jgershon/')
import pdb_util

sys.path.append('/home/jgershon/git/cleo')
import fragment_util
from ensemble import Ensemble

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def featurize_sequence(seq):
    """
    Featurize a protein sequence into a one-hot encoded tensor.
    """
    ret = torch.tensor([pdb_util.aa12num[x] for x in seq], dtype=torch.long)
    ret = torch.nn.functional.one_hot(ret, num_classes=20).float()
    ret = ret.unsqueeze(0).reshape(1, -1)
    return ret

def sample_initial_samples(seq_to_score, num_initial_samples, seed):
    """
    Sample initial sequences from the oracle map.
    """
    random.seed(seed)
    initial_sample = random.sample(list(seq_to_score.keys()), num_initial_samples)
    return [(x, seq_to_score[x]) for x in initial_sample]


def data_list_to_data_frame(data_list, opt_round, seq_to_name):
    """
    Convert a list of (sequence, score) tuples into a pandas DataFrame.
    """
    data_to_save = {
        "name": [seq_to_name[d[0]] for d in data_list],
        "sequence": [d[0] for d in data_list],
        "score": [d[1] for d in data_list],
        "opt_round": [opt_round] * len(data_list),
    }
    data_df = pd.DataFrame(data_to_save)

    return data_df

def z_score_normalize_data_df(df):
    """
    Normalize the scores in the DataFrame using z-score normalization.
    """
    df["z_score_norm_rate"] = (df["score"] - df["score"].mean()) / df["score"].std()
    return df

def find_last_checkpoint(train_name, surrogate_ckpt_base_dir):
    """
    Finds the last checkpoint in the specified directory.
    """
    all_paths = glob.glob(os.path.join(surrogate_ckpt_base_dir, train_name + '/*'))
    all_paths.sort()
    return all_paths[-1] # take most recent checkpoint

def greedy_selection(ckpt_path, possible_candidates, num_samples, device):
    """
    Greedy selection of sequences based on model predictions.
    """
    ckpt = torch.load(os.path.join(ckpt_path,"last.ckpt"), map_location=device)
    config = OmegaConf.load(os.path.join(ckpt_path,'config.yaml'))

    # load model
    model = Ensemble(config)
    model.load_state_dict(ckpt['state_dict'])
    model = model.eval()


    pred_data = {
        "sequence": [],
        "pred_mean": [],
        "pred_var": [],
    }

    with torch.no_grad():
        # make greedy predictions for every sequence in the dataset
        for seq in tqdm(possible_candidates, desc="Greedy selection model predictions"):

            input_feat = featurize_sequence(seq)
            out = model(input_feat)
            pred_data["sequence"].append(seq)
            pred_data["pred_mean"].append(out['mu'].item())
            pred_data["pred_var"].append(out['sigma'].item())

    pred_df = pd.DataFrame(pred_data)
    pred_df["UCB"] = pred_df["pred_mean"] + pred_df["pred_var"]

    # sort by UCB and return top num_samples
    pred_df = pred_df.sort_values(by="UCB", ascending=False)
    return pred_df["sequence"].tolist()[:num_samples]


# for sampling from batch acquisition function policy
def sample_unique_sequences_from_policy(dist: torch.distributions.Categorical, max_retries: int = 1000):
    """
    Sample unique sequences from a batch of Categorical distributions.
    Ensures that no two samples in the batch are equal.
    Assumes dist.probs is of shape [B, N, D].
    """
    B, N, D = dist.probs.shape
    samples = dist.sample()  # [B, N]

    for _ in range(max_retries):
        # Convert to tuple rows to compare
        flat = [tuple(row.tolist()) for row in samples]
        _, inverse, counts = torch.unique(torch.tensor(flat), return_inverse=True, return_counts=True, dim=0)
        dup_indices = (counts[inverse] > 1).nonzero(as_tuple=True)[0]

        if len(dup_indices) == 0:
            return samples  # all samples are unique

        # Resample only the duplicates
        resampled = dist.sample()
        samples[dup_indices] = resampled[dup_indices]

    raise RuntimeError(f"Could not sample {B} unique sequences after {max_retries} retries.")

def policy_smaples_to_sequences(samples, fragment_dictionary):
    '''
        take in samples from policy of shape [q, num_frags]
    '''
    q, num_frags = samples.shape
    
    candidate_seqs = []
    for j in range(q):

        name_list = []
        aa_seq_list = []
        for f,frag_id in enumerate(samples[j].tolist()):
            name_list.append(fragment_dictionary[str(f+1)][frag_id][0])
            aa_seq_list.append(fragment_dictionary[str(f+1)][frag_id][1])

        candidate_seqs.append((':'.join(name_list), ''.join(aa_seq_list)))

    return candidate_seqs


def load_and_get_sequences_from_policy(
        run_name, 
        fragment_dictionary, 
        base_path,
    ):
    
    """
    Get sequences from a policy.
    """
    policy_fp = os.path.join(base_path, run_name, "policy.pt")
    if not os.path.exists(policy_fp):
        raise FileNotFoundError(f"Policy file not found: {policy_fp}")

    policy = torch.load(policy_fp, map_location=torch.device('cpu'))

    # Sample unique sequences
    samples = sample_unique_sequences_from_policy(policy)
    
    # Convert samples to sequences
    candidate_seqs = policy_smaples_to_sequences(samples, fragment_dictionary)
    
    return candidate_seqs

def sample_from_batch_acqf_policy(num_to_sample, sequences_seen, run_name, fragment_dictionary, base_path):
    """
    Sample sequences from a batch acquisition function policy.
    """
    proposed_sequences = []
    while len(proposed_sequences) < num_to_sample:
        tmp_seqs = load_and_get_sequences_from_policy(run_name, fragment_dictionary, base_path)
        to_add = [x[1] for x in tmp_seqs if x[1] not in sequences_seen and x[1] not in proposed_sequences]
        proposed_sequences.extend(to_add[:num_to_sample - len(proposed_sequences)])  # ensure we only add up to num_to_sample sequences
        if len(proposed_sequences) >= num_to_sample:
            break
    return proposed_sequences


@hydra.main(version_base=None, config_path="./config")
def run_loop(cfg: DictConfig):
    """
        Big optimization loop with using dataset.
    """
    # setup output directory
    output_folder = os.path.join(cfg.outdir, cfg.run_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dataset_path = os.path.join(output_folder, f"dataset.csv") # path to save dataset

    # load fragment dictionary
    if cfg.fragment_dictionary_path is not None:
        with open(cfg.fragment_dictionary_path, 'r') as f:
            fragment_dictionary = json.load(f)
    else:
        fragment_dictionary = None

    # load oracle data
    oracle_df = pd.read_csv(cfg.oracle_data_path)
    seq_to_score = {k:v for k,v in zip(oracle_df['sequence'].tolist(), oracle_df['score'].tolist())}
    seq_to_name = {k:v for k,v in zip(oracle_df['sequence'].tolist(), oracle_df['name'].tolist())}

    print("Running optimization with the following configuration:")
    print(OmegaConf.to_yaml(cfg))

    # save configuration
    save_cfg_path = os.path.join(output_folder, "config.yaml")
    OmegaConf.save(cfg, save_cfg_path)

    # initialize to 0
    opt_round = 0

    # get initial data set list
    data_list = sample_initial_samples(seq_to_score, cfg.num_samples_per_round, cfg.seed)

    # save initial data list
    sequences_seen = [d[0] for d in data_list]

    # save initial dataset
    data_df = data_list_to_data_frame(data_list, opt_round, seq_to_name)
    data_df = z_score_normalize_data_df(data_df)
    data_df.to_csv(dataset_path, index=False)

    for _ in range(cfg.num_rounds):

        print(f"Now on optimization round {opt_round}...")

        # TRAIN MODEL
        train_name = f"traj_{cfg.run_name}_opt_round_{opt_round:04}"

        cmd = f"python {cfg.train_script_path} -cn {cfg.train_config} "
        cmd += f"run_name={train_name} data.dataset={dataset_path}"

        print(f"Running training with command: {cmd}")
        subprocess.run(cmd, shell=True)

        # find last checkpoint
        ckpt_path = find_last_checkpoint(train_name, cfg.surrogate_ckpt_base_dir)

        # get sequences that have not been seen yet
        possible_candidates = [x for x in seq_to_score.keys() if x not in sequences_seen]

        # SAMPLE NEXT ROUND OF SEQUENCES
        if cfg.candidate_selection == "greedy_batch":
            # predict over all sequences in pool and choose top-k
            proposed_sequences = greedy_selection(ckpt_path, possible_candidates, cfg.num_samples_per_round, DEVICE)

        elif cfg.candidate_selection == "batch_acqf":
            # use batch acquisition function to select next set of sequences to test
            acqf_run_name = f"{cfg.run_name}_acqf_opt_round_{opt_round:04}"

            cmd = f"python {cfg.acqf_opt_script_path} -cn {cfg.acqf_opt_config} "
            cmd += f"run_name={acqf_run_name} outdir={output_folder} "
            cmd += f"surrogate_ckpt={ckpt_path} acqf.gamma={cfg.acqf_gamma} "
            cmd += f"opt_loop.q={cfg.num_samples_per_round} "
            cmd += f"opt_loop.fragment_dictionary={cfg.fragment_dictionary_path}"

            print(f"Running acquisition function optimization with command: {cmd}")
            subprocess.run(cmd, shell=True)

            proposed_sequences = sample_from_batch_acqf_policy(cfg.num_samples_per_round, sequences_seen, acqf_run_name, fragment_dictionary, output_folder)

        elif cfg.candidate_selection == "random":
            random.shuffle(possible_candidates)
            proposed_sequences = random.sample(possible_candidates, cfg.num_samples_per_round)
        
        else:
            raise ValueError(f"Unknown candidate selection method: {cfg.candidate_selection}")

        # get the scores for the proposed sequences
        sequences_seen.extend(proposed_sequences)

        data_list = [
            (seq, seq_to_score[seq]) for seq in proposed_sequences if seq in seq_to_score
        ]            

        # save new data list
        opt_round += 1

        # SAVE NEW DATASET
        data_df = pd.concat(
            [
                data_df,
                data_list_to_data_frame(data_list, opt_round, seq_to_name)
            ], 
            ignore_index=True
        )
        data_df = z_score_normalize_data_df(data_df)
        data_df.to_csv(dataset_path, index=False)

    print(f"Optimization completed. Final dataset saved to {dataset_path}")
    
if __name__ == "__main__":
    run_loop()