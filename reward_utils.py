from abc import ABC, abstractmethod
import numpy as np
import torch
from policy_utils import alphabet

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
    def __call__(self, policy_output, feature_dict):
        """
            Compute rewards for each sequence in the sampled_sequences
            rewards should be returned as tensor of shape (B,)

            policy_output: Output from the policy network - dict
            feature_dict: Feature dictionary - dict
        """
        pass


class EnrichAminoAcidReward(Reward):
    """
        Simple reward to upweight an amino acid of interest
    """
    def __init__(self, AA_to_enrich):
        self.AA_to_enrich = AA_to_enrich
        assert AA_to_enrich in alphabet, f"Amino acid {AA_to_enrich} not in alphabet"
        self.AA_to_enrich_idx = alphabet.index(AA_to_enrich)

    def __call__(self, policy_output, feature_dict):
        sampled_seqs = policy_output["S"]
        batched_reward = (sampled_seqs == self.AA_to_enrich_idx).float().sum(dim=-1)
        batched_reward = batched_reward / sampled_seqs.shape[1]
        return batched_reward


