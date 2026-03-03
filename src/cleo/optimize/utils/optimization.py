"""Acquisition functions and REINFORCE-based optimization over fragment space.

Provides BatchUCBwithEntropy (upper confidence bound with batch diversity) and
opt_loop (REINFORCE optimization of a categorical policy over a combinatorial
fragment dictionary). Also includes legacy helper functions for direct sequence
optimization and nearest-sequence rounding that may be used from notebooks.
"""
import torch
import copy
from tqdm import tqdm
from cleo.optimize.utils import pdb_tools as pdb_util
from cleo.optimize.utils import fragment as fragment_util
import numpy as np
import random
import pandas as pd
import os


def get_candidates_from_policy(policy, fragment_dictionary, connector='_'):
    """Sample a batch of candidate sequences from a categorical policy."""
    action = policy.sample()

    q, num_frags = action.shape

    candidate_seqs = []
    for j in range(q):
        name_list = []
        aa_seq_list = []
        for f, frag_id in enumerate(action[j].tolist()):
            name_list.append(fragment_dictionary[f+1][frag_id][0])
            aa_seq_list.append(fragment_dictionary[f+1][frag_id][1])

        candidate_seqs.append((connector.join(name_list), ''.join(aa_seq_list)))

    return candidate_seqs


def get_candidates_from_fragment_opt(fragment_space, fragment_dictionary):
    """Extract candidate sequences from optimized fragment-space tensor."""
    N, q, num_frags, d = fragment_space.shape

    num_total_frags = [len(fragment_dictionary[x]) for x in fragment_dictionary]

    candidate_seqs = []
    for j in range(q):
        name_list = []
        aa_seq_list = []
        for f in range(num_frags):
            frag_id = torch.argmax(fragment_space[0, j, f][:num_total_frags[f]])
            name_list.append(fragment_dictionary[f+1][frag_id][0])
            aa_seq_list.append(fragment_dictionary[f+1][frag_id][1])
        candidate_seqs.append(('_'.join(name_list), ''.join(aa_seq_list)))

    return candidate_seqs


def get_seqs_from_action(action, fragment_dictionary):
    """Map discrete fragment actions to integer-encoded sequences."""
    N, q, _ = action.shape
    L = len(''.join([fragment_dictionary[x][0][1] for x in fragment_dictionary]))
    batched_seqs = torch.zeros(N, q, L)
    batched_names = []

    for i in range(N):
        batched_names.append([])
        for j in range(q):
            name_list = []
            aa_seq_list = []
            for f, frag_id in enumerate(action[i, j].tolist()):
                name_list.append(fragment_dictionary[f+1][frag_id][0])
                aa_seq_list.append(fragment_dictionary[f+1][frag_id][1])

            batched_names[i].append('_'.join(name_list))
            num_seq = [pdb_util.aa12num[x] for x in ''.join(aa_seq_list)]
            batched_seqs[i, j] = torch.tensor(num_seq)

    return batched_seqs, batched_names


def get_feasible_mask(fragment_dictionary):
    """Build a boolean mask over the policy dimensions to zero out infeasible fragment indices."""
    max_fragments = 0
    for i in range(len(fragment_dictionary)):
        if len(fragment_dictionary[i+1]) > max_fragments:
            max_fragments = len(fragment_dictionary[i+1])

    feasible_mask = torch.zeros(len(fragment_dictionary), max_fragments)
    for i in range(len(fragment_dictionary)):
        feasible_mask[i, :len(fragment_dictionary[i+1])] = 1

    return feasible_mask.bool()


def get_one_hot(num_start_seqs, num_classes=20):
    """Convert integer-encoded sequences to one-hot tensors."""
    if not isinstance(num_start_seqs, torch.Tensor):
        tmp = torch.tensor(num_start_seqs)
    else:
        tmp = num_start_seqs
    return torch.nn.functional.one_hot(tmp, num_classes=num_classes)


def policy_optimize_acquisition_function(acqf,
                                        fragment_dictionary,
                                        N=24,
                                        q=8,
                                        num_iter=1000,
                                        step_size=0.05,
                                        print_metrics=False,
                                        device='cpu'):
    """Optimize acquisition function via REINFORCE over a categorical fragment policy.

    Based on the policy optimization strategy from https://openreview.net/pdf?id=WV1ZXTH0OIn
    """
    feasible_mask = get_feasible_mask(fragment_dictionary)

    policy = torch.randn(q, feasible_mask.shape[0], feasible_mask.shape[1]) * 0.1
    feasible_mask = (feasible_mask[None]).repeat(q, 1, 1)

    policy[~feasible_mask] = -torch.inf
    policy = policy.to(device)
    policy = torch.nan_to_num(policy)
    policy = policy.requires_grad_(True)

    optimizer = torch.optim.Adam([policy], lr=step_size)
    collected_rewards = []

    print('policy optimization over fragment space')

    for i in tqdm(range(num_iter)):
        optimizer.zero_grad()

        soft_policy = torch.softmax(policy, dim=-1)
        m = torch.distributions.Categorical(soft_policy)

        sampled_actions = []
        sampled_log_probs = []
        for j in range(N):
            action = m.sample()
            sampled_log_probs.append(m.log_prob(action)[None])
            sampled_actions.append(action[None])

        actions = torch.cat(sampled_actions, dim=0)
        log_probs = torch.cat(sampled_log_probs, dim=0)

        sampled_seqs = get_seqs_from_action(actions, fragment_dictionary)[0].long()
        sampled_seqs = torch.nn.functional.one_hot(sampled_seqs, num_classes=20)
        sampled_seqs = sampled_seqs.reshape(sampled_seqs.shape[0], sampled_seqs.shape[1], -1)

        reward = acqf(sampled_seqs.to(device))

        beta_term = 0
        if i > 0:
            beta_term = np.mean(collected_rewards)

        collected_rewards.append(float(reward.mean().detach()))

        beta_subtract_reward = reward - beta_term

        loss = (-log_probs * beta_subtract_reward[..., None, None].repeat(1, q, actions.shape[-1])).sum(dim=(1, 2)).mean()

        loss.backward()
        optimizer.step()

        if i % 50 == 0 and print_metrics:
            print(f'step: {i}/{num_iter}\t loss:{float(loss):.3f}\t reward: {float(reward.mean()):.3f}')

    print(f'done, max acquisition value: {float(reward.mean()):.3f}')

    return torch.distributions.Categorical(soft_policy)


def optimize_acquisition_function(acqf, pool, N=1, q=8, print_metrics=False, num_iter=1000, step_size=0.05, fragment_representation=False, fragment_dictionary=None):
    """Optimize the acquisition function directly in continuous sequence space."""

    start_seqs = [x for x in random.choices(pool, k=q)]
    if fragment_representation:
        names = [x[0] for x in start_seqs]
        num_start_seqs = fragment_util.featurize_fragments(names, fragment_dictionary)
        num_classes = max([len(fragment_dictionary[x]) for x in fragment_dictionary])
    else:
        num_classes = 20
        num_start_seqs = []
        for seq in start_seqs:
            num_start_seqs.append([pdb_util.aa12num[x] for x in seq[1]])

    one_hot_start_seqs = get_one_hot(num_start_seqs, num_classes=num_classes).float()
    _, L, aa_dim = one_hot_start_seqs.shape

    opt_seqs = one_hot_start_seqs.reshape(q, -1)[None]
    opt_seqs = opt_seqs.requires_grad_(True)

    optimizer = torch.optim.Adam([opt_seqs], lr=step_size)

    print(f'optimizing in sequence space')
    for i in tqdm(range(num_iter)):
        optimizer.zero_grad()

        acq_values = acqf(opt_seqs)
        loss = -acq_values.sum()

        loss.backward()
        optimizer.step()

        if i % 50 == 0 and print_metrics:
            print(f'step: {i}/{num_iter}\t acq values: {float(acq_values.mean()):.3f}')

    print(f'done, max acquisition value: {float(acq_values.mean()):.3f}')

    return opt_seqs.reshape(N, q, L, aa_dim)


def round_to_nearest_sequences(output_seqs, seq_pool):
    """Round continuous sequence representations to the nearest sequence in a pool."""
    print(f'rounding to nearest sequences')
    q = output_seqs.shape[1]
    output_aa_seqs = []
    for i in range(q):
        num_seq = torch.argmax(output_seqs[0, i], dim=-1).tolist()
        aa_seq = ''.join([pdb_util.num2aa1[x] for x in num_seq])
        output_aa_seqs.append(aa_seq)

    nearest_seqs = []
    min_distances = []
    for qseq in tqdm(output_aa_seqs):
        min_dist = len(output_aa_seqs[0]) + 1
        seq_match = ''
        for pseq in seq_pool:
            dist = sum([x != y for x, y in zip(qseq, pseq[1])])
            if dist < min_dist:
                seq_match = copy.deepcopy(pseq)
                min_dist = dist
        nearest_seqs.append(seq_match)
        min_distances.append(min_dist)

    return nearest_seqs, min_distances


def get_candidate_acquisition_values(candidate_seqs,
                                    acqf,
                                    one_hot=False,
                                    fragment_representation=False,
                                    fragment_dictionary=None):
    """Evaluate the acquisition function on a list of candidate sequences."""
    if fragment_representation:
        names = [x[0] for x in candidate_seqs]
        candidate_num_seqs = fragment_util.featurize_fragments(names, fragment_dictionary)
        num_classes = max([len(fragment_dictionary[x]) for x in fragment_dictionary])
    else:
        num_classes = 20
        candidate_num_seqs = []
        for seq in candidate_seqs:
            candidate_num_seqs.append([pdb_util.aa12num[x] for x in seq[1]])

    if one_hot:
        q = len(candidate_seqs)
        candidate_input = get_one_hot(candidate_num_seqs, num_classes=num_classes).reshape(q, -1)[None]
    else:
        if not isinstance(candidate_num_seqs, torch.Tensor):
            candidate_input = torch.tensor(candidate_num_seqs)[None]
        else:
            candidate_input = candidate_num_seqs[None]

    acquisition_values = acqf(candidate_input)

    return float(acquisition_values)


class BatchUCBwithEntropy:
    """Batch acquisition function combining UCB with diversity (entropy or pairwise similarity).

    In sequence_wise mode, rewards individual sequences based on UCB + (1 - similarity).
    In batch mode, rewards the batch mean UCB + per-position entropy.
    """
    def __init__(self, model, model_batch_size=64, gamma=0.1, eps=1e-8, sequence_wise=True):
        self.model = model
        self.model_batch_size = model_batch_size
        self.gamma = gamma
        self.eps = eps
        self.sequence_wise = sequence_wise

    @torch.no_grad()
    def __call__(self, X):

        N, q, d = X.shape
        X_r = X.reshape(N*q, d)
        r = X_r.shape[0]

        num_batches = (r + self.model_batch_size - 1) // self.model_batch_size

        ucb_list = []
        for i in range(num_batches):
            start = i * self.model_batch_size
            end = min((i + 1) * self.model_batch_size, r)
            batch_X = X_r[start:end]

            out = self.model(batch_X)

            ucb = out['mu'] + out['sigma']
            ucb_list.append(ucb)

        if self.sequence_wise:
            ucb_per_seq = torch.cat(ucb_list, dim=0).reshape(N, q)

            X_seq_view = X.view(N, q, -1, 20)
            L = X_seq_view.shape[2]
            seq_flat = X_seq_view.view(N, q, L*20).float()
            seq_similarities = torch.matmul(seq_flat, seq_flat.transpose(-1, -2)) / L
            seq_similarities = seq_similarities.mean(dim=-1)

            reward = ucb_per_seq + self.gamma * (1 - seq_similarities)

            metrics = {
                'ucb': ucb_per_seq.mean().item(),
                'seq_similarity': seq_similarities.mean().item()
            }

        else:
            batched_ucb = torch.cat(ucb_list, dim=0).reshape(N, q).mean(dim=1)

            X_seq_view = X.view(N, q, -1, 20)
            X_freqs = X_seq_view.sum(dim=1) / q
            entropy = (-torch.sum(X_freqs * torch.log(X_freqs + self.eps), dim=-1)).mean(dim=-1)

            reward = batched_ucb + self.gamma * entropy

            metrics = {
                'ucb': batched_ucb.mean().item(),
                'entropy': entropy.mean().item(),
            }

        return reward, metrics


def opt_loop(acqf, fragment_dictionary, N, q, num_iter, lr, out_path, connector, device):
    """REINFORCE optimization loop over a categorical fragment policy.

    Optimizes a policy that samples q candidate sequences per iteration,
    evaluates them with the acquisition function, and updates via policy gradient.
    Supports both batch-level (REINFORCE with baseline) and sequence-level (GRPO)
    advantage estimation depending on the acquisition function's output shape.
    """
    feasible_mask = get_feasible_mask(fragment_dictionary)

    policy = torch.randn(q, feasible_mask.shape[0], feasible_mask.shape[1]) * 0.001
    feasible_mask = (feasible_mask[None]).repeat(q, 1, 1)

    policy[~feasible_mask] = -torch.inf
    policy = policy.to(device)
    policy = torch.nan_to_num(policy)
    policy = policy.requires_grad_(True)

    optimizer = torch.optim.Adam([policy], lr=lr)
    collected_rewards = []

    metric_logs = {"step": [], "reward": []}

    with tqdm(total=num_iter, desc='Reward: ') as pbar:
        for i in range(num_iter):
            optimizer.zero_grad()

            soft_policy = torch.softmax(policy, dim=-1)
            m = torch.distributions.Categorical(soft_policy)

            sampled_actions = []
            sampled_log_probs = []
            for j in range(N):
                action = m.sample()
                sampled_log_probs.append(m.log_prob(action)[None])
                sampled_actions.append(action[None])

            actions = torch.cat(sampled_actions, dim=0)
            log_probs = torch.cat(sampled_log_probs, dim=0)

            sampled_seqs = get_seqs_from_action(actions, fragment_dictionary)[0].long()
            sampled_seqs = torch.nn.functional.one_hot(sampled_seqs, num_classes=20)
            sampled_seqs = sampled_seqs.reshape(sampled_seqs.shape[0], sampled_seqs.shape[1], -1)

            reward, metrics = acqf(sampled_seqs.to(device))

            if i % 100 == 0:
                for k, v in metrics.items():
                    if k not in metric_logs:
                        metric_logs[k] = []
                    metric_logs[k].append(v)
                metric_logs["step"].append(i)
                metric_logs["reward"].append(float(reward.mean().detach()))

                metrics_df = pd.DataFrame(metric_logs)
                metrics_path = os.path.join(out_path, "metrics.csv")
                metrics_df.to_csv(metrics_path, index=False)

            if reward.shape[0] == N and reward.shape[1] == q:
                adv = (reward - reward.mean(dim=1).unsqueeze(-1)) / (reward.std(dim=1).unsqueeze(-1))
                loss = (-log_probs.sum(dim=2) * adv).sum(dim=1).mean()
            else:
                beta_term = 0
                if i > 0:
                    beta_term = np.mean(collected_rewards)

                collected_rewards.append(float(reward.mean().detach()))

                beta_subtract_reward = reward - beta_term

                loss = (-log_probs * beta_subtract_reward[..., None, None].repeat(1, q, actions.shape[-1])).sum(dim=(1, 2)).mean()

            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Reward': f'{float(reward.mean()):.3f}'})
            pbar.update(1)

    candidates = get_candidates_from_policy(m, fragment_dictionary, connector=connector)

    return candidates, m
