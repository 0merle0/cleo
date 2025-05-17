from abc import ABC, abstractmethod
import numpy as np
import torch
from policy_utils import alphabet
from af3_inference.af3_inference import main as af3_main
import pdb as pdb_lib
import os 
import pandas as pd

class Reward(ABC):
    """
        Base class for reward functions

        ! should try and ensure rewards are in the range of [0, 1]
    """

    def get_sequences(self, policy_output):
        """
            Return list of sampled sequences
        """
        sampled_sequences = policy_output["S"]
        B = sampled_sequences.shape[0]
        
        sequences = [] 
        for i in range(B):
            seq = sampled_sequences[i]
            seq_str = "".join([alphabet[int(s)] for s in seq])
            sequences.append(seq_str)
        return sequences

    @abstractmethod
    def __call__(self, step, policy_output, feature_dict, device):
        """
            Compute rewards for each sequence in the sampled_sequences
            rewards should be returned as tensor of shape (B,), as well as any metrics you would like to track

            policy_output: Output from the policy network - dict
            feature_dict: Feature dictionary - dict

        """
        pass


class EnrichAminoAcidReward(Reward):
    """
        Simple reward to upweight an amino acid of interest

        Make sure final reward ends up on same device
    """
    def __init__(self, AA_to_enrich=None):
        self.AA_to_enrich = AA_to_enrich
        assert AA_to_enrich in alphabet, f"Amino acid {AA_to_enrich} not in alphabet"
        self.AA_to_enrich_idx = alphabet.index(AA_to_enrich)

    @torch.no_grad()
    def __call__(self, step, policy_output, feature_dict, device):
        sampled_seqs = policy_output["S"]
        num_correct_aas = (sampled_seqs == self.AA_to_enrich_idx).float().sum(dim=-1)
        reward = num_correct_aas / sampled_seqs.shape[1]
        
        metrics = {
            "num_correct_aas": num_correct_aas.detach().mean().cpu()
        }

        return reward.to(device), metrics


class af3_reward(Reward):
    """
        Reward based on AlphaFold3 score
        
        Uses configurable thresholds to map metric values to the [0,1] range.
        For metrics like RMSD where lower is better.
    """
    def __init__(self, output_dir, af3_config, metric_name="rmsd", confidence_normalize_metric=None,lower_threshold=0.25, upper_threshold=3.0, normalization_type="linear"):
        """
        Initialize AF3 reward function
        
        Args:
            af3_config: Configuration for AF3 inference
            metric_name: Column name to use for the reward calculation (default: "rmsd")
            lower_threshold: Values below this get reward 1.0
            upper_threshold: Values above this get reward 0.0
        """
        self.af3_config = af3_config
        self.output_dir = output_dir
        self.metric_name = metric_name
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.normalization_type = normalization_type
        self.confidence_normalize_metric = confidence_normalize_metric 

        if normalization_type == "exponential":
            print(f"Using exponential reward normalization with lower_threshold: {lower_threshold} and upper_threshold: {upper_threshold}")
            # Calculate the steepness to achieve min reward at upper_threshold
            self.exp_steepness = -np.log(lower_threshold)
        
    def sequences_to_dataframe(self, sequences):
        """
        Convert list of sequences to DataFrame format required by AF3
        """

        df = pd.DataFrame({
            'name': [f'seq_{i}' for i in range(len(sequences))],
            'seq': sequences
        })
        return df
        
    def normalize_metric(self, values):
        """
        Normalize metric values to [0,1] range based on thresholds
        For metrics like RMSD where lower is better
        """
        normalized = np.ones_like(values, dtype=np.float32)
        
        # Values above upper_threshold get 0.0
        above_upper = values > self.upper_threshold
        normalized[above_upper] = 0.0
        
        # Values between thresholds get scaled linearly
        between = (values >= self.lower_threshold) & (values <= self.upper_threshold)
        if np.any(between):
            if self.normalization_type == "linear":
                # Linear scaling - same as before
                range_size = self.upper_threshold - self.lower_threshold
                normalized[between] = 1.0 - (values[between] - self.lower_threshold) / range_size
            elif self.normalization_type == "exponential":
                # Exponential scaling
                normalized_values = (values[between] - self.lower_threshold) / (self.upper_threshold - self.lower_threshold)
                # Apply exponential decay: e^(-k*x) - gives 1.0 at lower and 0 at upper threshold
                normalized[between] = (np.exp(-self.exp_steepness * normalized_values) - np.exp(-self.exp_steepness)) / (1 - np.exp(-self.exp_steepness))
            
        return normalized
        
    def __call__(self, step, policy_output, feature_dict, device):
        # Get the sequences from policy output
        sequences = self.get_sequences(policy_output)
        
        # Create a DataFrame for AF3
        input_df = self.sequences_to_dataframe(sequences)
        # need to update the datadir based on the current iteration
        iter = policy_output["iter"] if "iter" in policy_output else 0
        self.af3_config.datadir = os.path.join(self.output_dir, f"af3_outputs/iter_{step}")
        # Create a temporary directory for AF3 outputs if needed
        os.makedirs(self.af3_config.datadir, exist_ok=True)
        
        # Run AF3 inference
        result_df = af3_main(self.af3_config, input_df=input_df)
        # Look for the exact metric name in result columns
        if self.metric_name in result_df.columns:
            metric_values = result_df[self.metric_name].values
        else:
            # If the exact name is not found, look for it with state prefixes
            state_columns = [col for col in result_df.columns if col.endswith(f"_{self.metric_name}")]
            if state_columns:
                metric_values = result_df[state_columns[0]].values
            else:
                # If not found at all, return zeros
                print(f"Warning: Metric '{self.metric_name}' not found in AF3 results")
                print(f"Available columns: {result_df.columns.tolist()}")
                metric_values = np.zeros(len(sequences))
        
        # Normalize to [0,1] range
        normalized_scores = self.normalize_metric(metric_values)

        if self.confidence_normalize_metric:
            conf_values = result_df[self.confidence_normalize_metric].values
            # pdb_lib.set_trace()
            normalized_scores = normalized_scores * conf_values

        input_df['reward'] = normalized_scores
        # save a csv to self.af3_config.datadir
        input_df.to_csv(os.path.join(self.af3_config.datadir, f"af3_metrics.csv"), index=False)
        # Convert to tensor and return
        reward = torch.tensor(normalized_scores, dtype=torch.float32)
        # convert raw rmsds to a dictionary
        metrics = {
            f"{self.metric_name}_mean": result_df[self.metric_name].values.mean(),
            f"{self.metric_name}_std": result_df[self.metric_name].values.std(),
            f"{self.metric_name}_min": result_df[self.metric_name].values.min(),
            f"{self.metric_name}_max": result_df[self.metric_name].values.max(),
            f"{self.confidence_normalize_metric}_mean": conf_values.mean() if self.confidence_normalize_metric else None,
            f"{self.confidence_normalize_metric}_std": conf_values.std() if self.confidence_normalize_metric else None,
            f"{self.confidence_normalize_metric}_min": conf_values.min() if self.confidence_normalize_metric else None,
            f"{self.confidence_normalize_metric}_max": conf_values.max() if self.confidence_normalize_metric else None
        }

        return reward.to(device), metrics