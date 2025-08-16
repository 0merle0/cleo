
import sys, os
import secrets
import random
import numpy as np
import torch

# 0 indexed inclusive
def make_fragment_dict(seq_list, fragment_bounds):
    """
    Create a dictionary of fragments from a list of sequences.
    """
    fragment_dict = {}
    for i in range(len(fragment_bounds)):
        start, end = fragment_bounds[i]
        fragment_dict[f"fragment_{i+1}"] = [(f"{n:04}."+secrets.token_hex(4), seq[start:end+1]) for n, seq in enumerate(seq_list)]
    return fragment_dict

# function to sample sequences
def sample_sequences(fragment_dict, num_samples, min_sample, max_iter=10000):
    """
    Sample sequences from the fragments in the dictionary.
    """
    
    all_frags = []
    for k in fragment_dict.keys():
        all_frags.extend([x[0] for x in fragment_dict[k]])

    # continue sampling until all fragments are sampled at least min_sample times
    valid_sample = False
    iter_count = 0
    while not valid_sample:
        samples = []
        sampled_counts = [0] * len(all_frags)

        # get weights for each fragment to make sampling more uniform
        frag_weights = {f"fragment_{i+1}": [0]*len(fragment_dict[f"fragment_{i+1}"]) for i in range(len(fragment_dict))}

        for i in range(num_samples):
            _name = []
            _seq = []
            for i in range(len(fragment_dict)):
                weights = frag_weights[f"fragment_{i+1}"]
                # normalize weights
                weights = (1 - np.array(weights)/(max(weights)+1e-3))
                weights = weights / np.sum(weights)
                frag_name, frag_seq = random.choices(
                    fragment_dict[f"fragment_{i+1}"],
                    weights=weights,
                    k=1,
                )[0]
                sampled_counts[all_frags.index(frag_name)] += 1
                w_index = fragment_dict[f"fragment_{i+1}"].index((frag_name, frag_seq))
                frag_weights[f"fragment_{i+1}"][w_index] += 1
                _name.append(frag_name)
                _seq.append(frag_seq)
            
            samples.append(("_".join(_name), "".join(_seq)))

        # check if all fragments are sampled
        valid_num = sum([1 for x in sampled_counts if x >= min_sample])
        if valid_num == len(all_frags):
            valid_sample = True

        iter_count += 1

        if iter_count > max_iter:
            print(f"Warning: max iterations reached ({max_iter}) without sampling all fragments.")
            break

    return samples


# make new dataframe from merged fragments
def get_fragment_rewards(seq_list, rewards, fragment_dict, fragment_bounds):
    """
    Aggregate reward data for each fragment.

        assume input is list of sequences and rewards for each sequence

    Should return (B,L) tensor with the reward for each fragment
    """

    # should be (B,L)
    fragmented_rewards = torch.zeros(len(fragment_dict[f'fragment_1']), len(seq_list[0]))

    for i in range(len(fragment_dict)):
        start, end = fragment_bounds[i]

        for frag_name, frag_seq in fragment_dict[f'fragment_{i+1}']:
            
            batch_idx = int(frag_name.split(".")[0])
            frag_reward = np.mean([r for r,s in zip(rewards, seq_list) if frag_seq in s])

            fragmented_rewards[batch_idx, start:end+1] = torch.tensor(frag_reward)

    return fragmented_rewards
