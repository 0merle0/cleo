import sys, os
import glob
import json
import numpy as np
import torch
import pandas as pd

def get_fragment_dictionary(fragments_csv):
    '''
        parse fragment order csv
        csv must have columns: 'name','fragment','seq'
    '''
    # read in csv
    library_df = pd.read_csv(fragments_csv)
    
    # get number of fragments
    num_fragments = len(library_df['fragment'].unique())
    
    # build fragment dictionary
    fragment_dictionary = {}
    for i in range(num_fragments):
        tmp = library_df[library_df['fragment']==i+1]
        fragment_dictionary[i+1] = [(f'frag{i+1}_{name}',frag) for name,frag in zip(tmp['name'].tolist(),tmp['seq'].tolist())]
    
    # alphabetically sort all fragments
    for x in fragment_dictionary:
        fragment_dictionary[x].sort()

    return fragment_dictionary


def make_all_sequences(fragment_dictionary, seq_list=[('','')], frag_num=1):
    '''
        recursive func to make all possible sequences in list
    '''
    new_seq_list = []
    for name,seq in seq_list:
        for name_to_add,frag_to_add in fragment_dictionary[frag_num]:
            if frag_num == 1:
                catname = name_to_add
            else:
                catname = name+'_'+name_to_add
            new_seq_list.append((catname,seq+frag_to_add))
    
    frag_num += 1
    if frag_num-1 < len(fragment_dictionary):
        return make_all_sequences(fragment_dictionary, seq_list=new_seq_list, frag_num=frag_num)
    else:
        print(f'Generated {len(new_seq_list)} possible sequences.')
        return new_seq_list


def get_all_sequences(fragment_dictionary):
    '''
        wrapper func to make all seqs in the library
    '''
    all_seqs = make_all_sequences(fragment_dictionary)
    all_seqs.sort()
    return all_seqs

def featurize_fragments(names, fragment_dictionary):
    '''
        split sequences by fragment and return features for training over fragment space
    '''

    frag_nums = []
    for n in names:
        split = n.split('frag')[1:]

        # check to make sure split is correct
        assert len(split) == len(fragment_dictionary)
        
        frag_list = []
        for s in split:
            tmp = 'frag'+s
            if '_' == tmp[-1]:
                tmp = tmp[:-1]
            frag_list.append(tmp)
        
        tmp_nums = []
        for i,f in enumerate(frag_list):
            frag_idx = [x[0] for x in fragment_dictionary[i+1]].index(f)
            tmp_nums.append(frag_idx)
            
        frag_nums.append(tmp_nums)

    return torch.tensor(frag_nums)
