import sys, os, copy
import numpy as np
import torch
import pdb as pdb_lib
import pandas as pd
from omegaconf import OmegaConf
import subprocess
from hydra.utils import get_method

from cleo.design.utils.policy import alphabet


class UniversalReward():
    """
        Universal reward function
        will run the steps and reward aggregations
        described in the config file
    """

    def __init__(
            self,  
            output_dir=None, 
            run_name=None,
            steps=None,
            reward_aggregation=None,
        ):

        self.output_dir = output_dir
        self.run_name = run_name
        self.steps = steps
        self.reward_aggregation = reward_aggregation
    
    def get_sequences(self, policy_output, chain_mask=None):
        """
            Return list of sampled sequences
        """
        sampled_sequences = policy_output["S"]

        if chain_mask is not None:
            sampled_sequences = sampled_sequences[:, chain_mask]

        B = sampled_sequences.shape[0]
        
        sequences = [] 
        for i in range(B):
            seq = sampled_sequences[i]
            seq_str = "".join([alphabet[int(s)] for s in seq])
            sequences.append(seq_str)
        return sequences

    def get_input_df(self, sequences):
        """
        Convert list of sequences to DataFrame format required by AF3
        """

        df = pd.DataFrame(
            {
                "sequence": sequences,
                "name": [f"seq_{i:04}" for i in range(len(sequences))],
                "origin.path": [f"seq_{i:04}.path" for i in range(len(sequences))],
            }
        )

        return df
    
    
    @torch.no_grad()
    def __call__(self, step, policy_output, feature_dict, device):

        rundir = os.path.join(
            self.output_dir, 
            self.run_name,
            "outputs", 
            f"step_{step:04}"
        )

        # path exists, delete it
        if os.path.exists(rundir):
            subprocess.run(f'rm -rf {rundir}', shell=True, check=True)  # Clean up outputs to retry
        
        # just take sequences from the first chain
        chain_mask = feature_dict["chain_labels"]==0 # [1, L]
        chain_mask = chain_mask[0] # [L,]

        # Get the sequences from policy output
        sequences = self.get_sequences(policy_output, chain_mask=chain_mask)
        
        # Create a DataFrame for AF3
        df = self.get_input_df(sequences)
        
        # iterate through list of steps here and 
        # run each step of the oracle and filtering pipeline
        print("********* running metric steps *********")
        for _s in self.steps:
            fn = get_method(_s.target_fn)
            _name = _s.name
            print(f"Running step: {_name} using function: {_s.target_fn}")
            _s.cfg.rundir = rundir
            _s.cfg.step = _name
            df = fn(df, _s.cfg, step_name=_name)


        # add code to delete af3 outputs to save space
        # DOING AF3 CLEANUP
        af3_out_dir = os.path.join(rundir, "af3/outputs")
        subprocess.run(f'rm -rf {af3_out_dir}/*', shell=True, check=True)  # Clean up outputs to retry


        # iterate through each metric and normalize
        rewards = []
        weights = []
        print("********* aggregating rewards *********")
        for m in self.reward_aggregation:
            print(f"Processing metric: {m.metric} with mode: {m.mode} and weight: {m.weight}")
            _r = torch.tensor(df[m.metric].tolist())

            _r_clamped = torch.clamp(_r, min=m.lower_bound, max=m.upper_bound)
            _r_norm = (_r_clamped - m.lower_bound) / (m.upper_bound - m.lower_bound + 1e-3)

            if m.mode == "max":
                _reward = _r_norm
            elif m.mode == "min":
                _reward = 1 - _r_norm
            elif m.mode == "avg":
                _reward = 1 - torch.abs(_r_norm - 0.5) * 2  # reward highest at 0.5
            else:
                raise ValueError(f"Unknown mode {m.mode} for metric {m.metric}")

            rewards.append(_reward*m.weight)
            weights.append(m.weight)

        # get final rewards
        reward = torch.stack(rewards, dim=1).sum(dim=1) / sum(weights)

        # make sure reward is properly padded when designing multiple chains
        if len(reward.shape) == 2 and reward.shape[1] != chain_mask.shape[0]:
            padding = torch.zeros(chain_mask.shape[0] - reward.shape[1]).unsqueeze(0).repeat(reward.shape[0], 1)
            reward = torch.cat([reward, padding], dim=1)

        # save df to output dir
        df.to_csv(os.path.join(rundir, f"metrics.csv"), index=False)

        to_log = {}
        for col in df.columns:

            # if the column is numeric, log mean, min, max
            if pd.api.types.is_numeric_dtype(df[col]):
                col_tensor = np.array(df[col].tolist())
                to_log[f"{col}_batch_mean"] = col_tensor.mean().item()
                to_log[f"{col}_batch_min"] = col_tensor.min().item()
                to_log[f"{col}_batch_max"] = col_tensor.max().item()

        
        return reward.to(device), to_log