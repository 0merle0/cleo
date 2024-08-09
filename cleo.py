import sys
import os
import copy
import glob
import json
import shutil
import torch
import random
import botorch
import scipy
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import gpytorch
from sklearn.model_selection import train_test_split
from botorch.acquisition import qUpperConfidenceBound
import fragment_util
import pdb_util
import model_util

class CLEO():
    
    def __init__(self, fragment_csv=None, data_csvs=None):
        '''
            fragment_csv - path to csv file of fragments
            data_csvs - list of paths to csv files with collected data
        '''

        self.data_df = None
        self.observed = None
        self.pool = None
        self.fragment_dictionary = None
        self.all_seqs = None
        self.L = None
        
        # make fragment dictionary and list of all seqs
        if fragment_csv != None:
            self.fragment_dictionary = fragment_util.get_fragment_dictionary(fragment_csv)
            self.all_seqs = fragment_util.get_all_sequences(self.fragment_dictionary)
            self.L = max([len(x[1]) for x in self.all_seqs])

        
        if data_csvs != None:
            # these will get initialized when load_data is called
            self.data_df = self.load_data(data_csvs)
            
        # if provided both fragments and data
        if data_csvs != None and fragment_csv != None:
            self.observed, self.pool = self.get_candidate_pool(self.data_df) 
    
    def build_dataset():
        '''
            make dataset with available data
        '''
        pass

    def select_acquisition_function():
        '''
            choose which acquisition function to use
        '''
        pass

    def initialize_surrogate():
        '''
            initialize and return the surrogate model to fit
        '''
        pass

    def train_surrogate():
        '''
            train the surrogate function and save state dict of trained function
        '''
        pass

    def get_candidates():
        '''
            use botorch packages to return set of candidate points to test next (usually will be done in batch mode)
        '''
        pass
    
    def load_data(self, csv_list):
        '''
            take csv list and load into merged dataframe, make sure columns match
        '''
        if type(csv_list) == str:
            csv_list = [csv_list]

        df_list = []
        for csv in csv_list:
            tmp = pd.read_csv(csv)
            df_list.append(tmp)
        data_df = pd.concat(df_list)
        return data_df
        
        
    def get_candidate_pool(self, data_df):
        '''
            figure out which sequences have already been tested and which have not yet
        '''
        assert 'name' in data_df.columns, 'please make sure that there is a "name" column in your data frame'
        
        observed_names = list(set(data_df['name'].tolist()))
        
        observed = []
        pool = []
        
        for n,s in self.all_seqs:
            if n in observed_names:
                observed.append((n,s))
            else:
                pool.append((n,s))
        return observed, pool
        
        
    def get_train_and_test_sets(self, one_hot=False, input_col='seq', label_col='norm_activity', k=10, random_split=False, fragment_representation=False):
        '''
            split train and test by top-k holdout for hyperparameter tuning
            
            will return Dataset objects made from "seq_activity_dataset" class
        '''
        if random_split:
            # random split from sklearn
            train_df, test_df = train_test_split(self.data_df, test_size=0.1)
            
        else:
            # sort df to take top-k
            df = self.data_df.sort_values(label_col, ascending=False)

            test_df = df[:k]
            train_df = df[k:]

            
        # make datasets
        train_set = model_util.seq_activity_dataset(train_df,
                                         one_hot=one_hot,
                                         input_col=input_col,
                                         label_col=label_col,
                                         fragment_dictionary=self.fragment_dictionary,
                                         fragment_representation=fragment_representation)

        test_set = model_util.seq_activity_dataset(test_df,
                                        one_hot=one_hot,
                                        input_col=input_col,
                                        label_col=label_col,
                                        fragment_dictionary=self.fragment_dictionary,
                                        fragment_representation=fragment_representation)
        
        return train_set, test_set

