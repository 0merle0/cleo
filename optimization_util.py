import torch
import copy
from tqdm import tqdm
import pdb_util
import numpy as np
import random
import fragment_util

def get_candidates_from_policy(policy, fragment_dictionary):
    '''
        take in policy (torch.distributions.Categorical) and sample batch
    '''
    action = policy.sample()

    q, num_frags = action.shape
    
    candidate_seqs = []
    for j in range(q):

        name_list = []
        aa_seq_list = []
        for f,frag_id in enumerate(action[j].tolist()):
            name_list.append(fragment_dictionary[f+1][frag_id][0])
            aa_seq_list.append(fragment_dictionary[f+1][frag_id][1])

        candidate_seqs.append(('_'.join(name_list), ''.join(aa_seq_list)))

            
    return candidate_seqs

def get_candidates_from_fragment_opt(fragment_space, fragment_dictionary):
    '''
        take output of optimizing in framgent space and return candidate list of seqs
    '''
    N, q, num_frags, d = fragment_space.shape

    num_total_frags = [len(fragment_dictionary[x]) for x in fragment_dictionary]

    candidate_seqs = []
    for j in range(q):

        name_list = []
        aa_seq_list = []
        for f in range(num_frags):
            frag_id = torch.argmax(fragment_space[0,j,f][:num_total_frags[f]])
            name_list.append(fragment_dictionary[f+1][frag_id][0])
            aa_seq_list.append(fragment_dictionary[f+1][frag_id][1])
        candidate_seqs.append(('_'.join(name_list), ''.join(aa_seq_list)))

    return candidate_seqs

def get_seqs_from_action(action, fragment_dictionary):
    '''
        map from action to actual sequences
    '''
    N, q, _ = action.shape
    L = len(''.join([fragment_dictionary[x][0][1] for x in fragment_dictionary]))
    batched_seqs = torch.zeros(N,q,L)
    batched_names = []

    for i in range(N):
        batched_names.append([])
        for j in range(q):

            name_list = []
            aa_seq_list = []
            for f,frag_id in enumerate(action[i,j].tolist()):
                name_list.append(fragment_dictionary[f+1][frag_id][0])
                aa_seq_list.append(fragment_dictionary[f+1][frag_id][1])

            batched_names[i].append('_'.join(name_list))
            num_seq = [pdb_util.aa12num[x] for x in ''.join(aa_seq_list)]
            batched_seqs[i,j] = torch.tensor(num_seq)
            
    return batched_seqs, batched_names

def get_feasible_mask(fragment_dictionary):
    '''
        from the fragment dictionary get a mask for the feasible regions to sample from
        if there are different numbers of fragments, policy will be [num_fragments, max(sequences per fragment)]
        mask will be used to set the probability of sampling bad fragments to 0    
    '''
    max_fragments = 0
    for i in range(len(fragment_dictionary)):
        if len(fragment_dictionary[i+1]) > max_fragments:
            max_fragments = len(fragment_dictionary[i+1])

    feasible_mask = torch.zeros(len(fragment_dictionary), max_fragments)
    for i in range(len(fragment_dictionary)):
        feasible_mask[i,:len(fragment_dictionary[i+1])] = 1

    return feasible_mask.bool()

def get_one_hot(num_start_seqs, num_classes=20):
    '''
        from list of num_seqs to 
    '''
    if type(num_start_seqs) != torch.tensor:
        tmp = torch.tensor(num_start_seqs)
    
    return torch.nn.functional.one_hot(tmp, num_classes=num_classes)


def policy_optimize_acquisition_function(acqf, 
                                        fragment_dictionary, 
                                        N=24, 
                                        q=8, 
                                        num_iter=1000, 
                                        step_size=0.05, 
                                        print_metrics=False, 
                                        device='cpu'):
    '''
        using policy optimization strategy from: https://openreview.net/pdf?id=WV1ZXTH0OIn
        optmize acquisition funciton by defining policy over fragment space
    '''
    feasible_mask = get_feasible_mask(fragment_dictionary)

    # initialize policy
    policy = torch.randn(q,feasible_mask.shape[0],feasible_mask.shape[1])*0.1 #scaling factor
    feasible_mask = (feasible_mask[None]).repeat(q,1,1)

    # set non fragments to << 0 so they do not get sampled
    policy[~feasible_mask] = -torch.inf
    policy = policy.to(device)
    policy = torch.nan_to_num(policy)
    policy = policy.requires_grad_(True)



    optimizer = torch.optim.Adam([policy], lr=step_size)
    collected_rewards = []

    print('policy optimization over fragment space')

    # REINFORCE LOOP
    for i in tqdm(range(num_iter)):
        
        optimizer.zero_grad()

        # softmax policy to get probabilities

        soft_policy = torch.softmax(policy,dim=-1)

        # make categorical distribution
        m = torch.distributions.Categorical(soft_policy)

        sampled_actions = []
        sampled_log_probs = []
        for j in range(N):
            # sample action
            action = m.sample()
            sampled_log_probs.append(m.log_prob(action)[None])
            sampled_actions.append(action[None])

        actions = torch.cat(sampled_actions,dim=0)
        log_probs = torch.cat(sampled_log_probs,dim=0)

        # convert from fragments to num seq
        sampled_seqs = get_seqs_from_action(actions, fragment_dictionary)[0].long()
        sampled_seqs = torch.nn.functional.one_hot(sampled_seqs, num_classes = 20)
        sampled_seqs = sampled_seqs.reshape(sampled_seqs.shape[0],sampled_seqs.shape[1],-1)

        # get reward
        reward = acqf(sampled_seqs.to(device))

        # substract beta term "moving average"
        beta_term = 0
        if i > 0:
            beta_term = np.mean(collected_rewards)

        collected_rewards.append(float(reward.mean().detach()))

        beta_subtract_reward = reward - beta_term

        # calculate policy loss
        loss = (-log_probs * beta_subtract_reward[...,None,None].repeat(1,q,actions.shape[-1])).sum(dim=(1,2)).mean()

        # opt
        loss.backward()
        optimizer.step()

        if i%50==0 and print_metrics:
            print(f'step: {i}/{num_iter}\t loss:{float(loss):.3f}\t reward: {float(reward.mean()):.3f}')

    print(f'done, max acquisition value: {float(reward.mean()):.3f}')
    
    # return policy as a distribution obj
    return torch.distributions.Categorical(soft_policy)

def optimize_acquisition_function(acqf, pool, N=1, q=8, print_metrics=False, num_iter=1000, step_size=0.05, fragment_representation=False, fragment_dictionary=None):
    '''
        opitmize the acquisition function directly
        take grads from acq value to 
    '''

    start_seqs = [x for x in random.choices(pool,k=q)]
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

    opt_seqs = one_hot_start_seqs.reshape(q,-1)[None] # add in "restart" dim
    opt_seqs = opt_seqs.requires_grad_(True)

    # opt loop
    optimizer = torch.optim.Adam([opt_seqs], lr=step_size)

    print(f'optimizing in sequence space')
    for i in tqdm(range(num_iter)):
        optimizer.zero_grad()

        # this performs batch evaluation, so this is an N-dim tensor
        acq_values = acqf(opt_seqs)  # torch.optim minimizes
        loss = -acq_values.sum()

        loss.backward()  # perform backward pass
        optimizer.step()  # take a step


        if i%50==0 and print_metrics:
            print(f'step: {i}/{num_iter}\t acq values: {float(acq_values.mean()):.3f}')

    print(f'done, max acquisition value: {float(acq_values.mean()):.3f}')
    
    return opt_seqs.reshape(N,q,L,aa_dim)

def round_to_nearest_sequences(output_seqs, seq_pool):
    '''
        after optimizing in sequence space, clamp to nearest available sequence
    '''
    
    print(f'rounding to nearest sequences')
    q = output_seqs.shape[1]
    output_aa_seqs = []
    for i in range(q):
        num_seq = torch.argmax(output_seqs[0,i],dim=-1).tolist()
        aa_seq = ''.join([pdb_util.num2aa1[x] for x in num_seq])
        output_aa_seqs.append(aa_seq)
    
    nearest_seqs = []
    min_distances = []
    for qseq in tqdm(output_aa_seqs):
        min_dist = len(output_aa_seqs[0])+1
        seq_match = ''
        for pseq in seq_pool:
            dist = sum([x!=y for x,y in zip(qseq,pseq[1])])
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
    '''
        given seq list [(name,aa_seq),] return the acquisition function value for the batch
    '''
    
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
        candidate_input = get_one_hot(candidate_num_seqs, num_classes=num_classes).reshape(q,-1)[None]
    else:
        if type(candidate_num_seqs) != torch.tensor:
            candidate_input = torch.tensor(candidate_num_seqs)[None]
        else:
            candidate_input = candidate_num_seqs[None]

    acquisition_values = acqf(candidate_input)

    return float(acquisition_values)
