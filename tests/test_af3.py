#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
import numpy as np
import os
from policy_utils import PolicyMPNN, alphabet, restype_STRtoINT
from reward_utils import af3_reward
from typing import List, Dict, Any
import pdb as pdb_lib

def sequence_to_indices(sequence: str) -> List[int]:
    """Convert an amino acid sequence to a list of indices."""
    return [restype_STRtoINT.get(aa, 20) for aa in sequence]  # Default to 'X' (20) for unknown AAs

def create_mock_policy_output(sequences: List[str]) -> Dict[str, torch.Tensor]:
    """
    Create a mock policy output dictionary containing sequences converted to indices.
    
    Args:
        sequences: List of amino acid sequences
        
    Returns:
        Dictionary with "S" key containing tensor of sequence indices
    """
    # Convert sequences to tensors of indices
    max_length = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    
    # Initialize tensor with all 'X' (index 20)
    S = torch.full((batch_size, max_length), 20, dtype=torch.long)
    
    # Fill with actual sequence indices
    for i, seq in enumerate(sequences):
        indices = sequence_to_indices(seq)
        S[i, :len(indices)] = torch.tensor(indices, dtype=torch.long)
    
    return {"S": S}

@hydra.main(version_base=None, config_path="../config", config_name="policy_mpnn")
def test_af3_reward(cfg: DictConfig) -> None:
    """
    Test the AF3 reward function using sequences from a list file.
    
    Args:
        cfg: Configuration object
    """
    # Ensure cfg has test_seq_list
    if not hasattr(cfg, "test_seq_list"):
        raise ValueError("test_seq_list not found in config. Please add path to a .list file with protein sequences.")
    
    # Read sequences from list file
    sequences = []
    with open(cfg.test_seq_list, 'r') as f:
        for line in f:
            # Remove whitespace and skip empty lines
            seq = line.strip()
            if seq:
                sequences.append(seq)
    
    print(f"Loaded {len(sequences)} sequences from {cfg.test_seq_list}")
    
    # Create dataframe
    seq_df = pd.DataFrame({'protein_sequence': sequences})
    
    # Create mock policy output
    policy_output = create_mock_policy_output(sequences)
    
    # Test directly with reward class
    from hydra.utils import instantiate
    reward_fn = instantiate(cfg.reward)
    # pdb_lib.set_trace()
    # We pass None for feature_dict since the reward function doesn't use it
    rewards = reward_fn(policy_output, None)
    
    # Return statistics
    print(f"\nReward Statistics:")
    print(f"Min: {rewards.min().item():.4f}")
    print(f"Max: {rewards.max().item():.4f}")
    print(f"Mean: {rewards.mean().item():.4f}")
    print(f"Std: {rewards.std().item():.4f}")
    
if __name__ == "__main__":
    test_af3_reward()