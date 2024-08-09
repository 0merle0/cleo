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
from icecream import ic

class seq_activity_dataset(Dataset):
    
    def __init__(self, df, input_col='seq', label_col='norm_activity', name_col='name', fragment_dictionary=None, fragment_representation=False):
        super(seq_activity_dataset, self).__init__()

        '''
            initialize and build training set from csvs
        '''
        self.seqs = df[input_col].tolist()
        self.activities = torch.tensor(df[label_col].tolist())
        
        if fragment_representation:
            # if training over fragment space
            self.featurized_seqs = fragment_util.featurize_fragments(df[name_col].tolist(), fragment_dictionary)
            num_classes = max([len(fragment_dictionary[x]) for x in fragment_dictionary])
        else:
            # get num seqs
            self.featurized_seqs = []
            for seq in self.seqs:
                self.featurized_seqs.append([pdb_util.aa12num[x] for x in seq])
            num_classes = 20

        # if one hot then featurize the sequences to be one hot
        self.featurized_seqs = get_one_hot(self.featurized_seqs, num_classes=num_classes)
        self.featurized_seqs = self.featurized_seqs.reshape(self.featurized_seqs.shape[0],-1)
        
    def __len__(self):
        return len(self.featurized_seqs)
    
    def __getitem__(self, index):
        return self.featurized_seqs[index], self.activities[index]
    

def get_one_hot(seqs,  num_classes=20):
    '''
        go from num seq to one hot seq
    '''
    one_hot_seqs = []
    for x in seqs:
        one_hot_seqs.append(torch.nn.functional.one_hot(torch.tensor(x),num_classes=num_classes)[None])
    return torch.cat(one_hot_seqs,dim=0)

class feature_extractor(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super(feature_extractor, self).__init__()
        
        # make intermediate layers
        self.hidden_layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
    def forward(self, x):
        return self.hidden_layers(x.float())
        
    
class GP(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    # must inherit botorch model to work
    def __init__(
        self,
        train_x=None, 
        train_y=None, 
        likelihood=None, 
        input_dim=None,
        hidden_dim=None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.embedding = feature_extractor(input_dim, hidden_dim)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        self._num_outputs = 1 # added according to: github.com/pytorch/botorch/issues/354
        
    def forward(self, x):
        
        x = self.embedding(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def save_gp(model, train_params, save_path):
    '''
        save gp model
    '''
    to_save = {
            'state_dict':model.state_dict(),
            'train_params':train_params,
            }

    torch.save(to_save, save_path)

def load_gp(model_path):
    '''
        path to model saved with train_params and state_dict
    '''
    tmp = torch.load(model_path)
    train_params = tmp['train_params']
    state_dict = tmp['state_dict']

    gp_keys = ['train_x','train_y','input_dim','hidden_dim']
    gp_params = {k:train_params[k] for k in gp_keys}

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_params['likelihood'] = likelihood

    model = GP(**gp_params)

    model.load_state_dict(state_dict)

    return model


# SETUP TRAINING
def train_gp(train_params=None, device='cpu'):
    '''
        train gp according to params and return trained model, likelihood, and losses
    '''
    
    training_iter = train_params['training_iter']
    train_x = train_params['train_x'].to(device)
    train_y = train_params['train_y'].to(device)

    gp_keys = ['train_x','train_y','input_dim','hidden_dim']
    gp_params = {k:train_params[k] for k in gp_keys}
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_params['likelihood'] = likelihood

    model = GP(**gp_params).to(device)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['learning_rate'])  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    model.train()
    likelihood.train()

    loss_list = []
    print(f'now training model : {device}')
    # TRAINING
    for i in tqdm(range(training_iter)):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)

        # Calc loss and backprop gradients
        loss = -mll(output, train_y)

        loss.backward()

        optimizer.step()
        
        loss_list.append(loss.item())


    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    
    return model, loss_list


def evaluate_gp(model, train_set, test_set=None, one_hot=True, show_plot=True, save_fig=None):
    '''
        evaluate the gp and get plot of pred vs true
    '''
    train_x = train_set.featurized_seqs.to('cpu')
    if test_set != None:
        test_x = test_set.featurized_seqs.to('cpu')

    # evaluate train and test data
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        etrain = model.posterior(train_x)
        if test_set != None:
            etest = model.posterior(test_x)

    # get spearman
    spearmanr = scipy.stats.spearmanr(train_set.activities, etrain.mean)[0]
    if test_set != None:
        spearmanr_test = scipy.stats.spearmanr(test_set.activities, etest.mean)[0]

    eval_data = {
        'true_activities':[],
        'predicted_activities':[],
        'label':[]
    }
    
    if test_set != None:
        all_activities = torch.cat([train_set.activities, test_set.activities],dim=0)
    else:
        all_activities = train_set.activities

    eval_data['true_activities'] += train_set.activities.tolist()
    eval_data['predicted_activities'] += etrain.mean.squeeze(-1).tolist()
    eval_data['label'] += ['train']*len(etrain.mean.tolist())


    if test_set != None:
        eval_data['true_activities'] += test_set.activities.tolist()
        eval_data['predicted_activities'] += etest.mean.squeeze(-1).tolist()
        eval_data['label'] += ['test']*len(etest.mean.tolist())

    eval_df = pd.DataFrame(eval_data)
    
    # get difference between means
    plt.figure(dpi=150)
    plt.plot([int(all_activities.min()-1),int(all_activities.max()+1)],[int(all_activities.min()-1),int(all_activities.max()+1)],color='k',linestyle='--',alpha=0.3)
    if test_set != None:
        plt.title(f'spearmanr train: {spearmanr:.3f}, spearmanr test: {spearmanr_test:.3f}')
    else:
        plt.title(f'spearmanr train: {spearmanr:.3f}')
    sns.scatterplot(data=eval_df, y='predicted_activities', x='true_activities', hue='label')
    plt.xlabel('true')
    plt.ylabel('predicted')
    plt.xlim([int(all_activities.min()-1),int(all_activities.max()+1)])
    plt.ylim([int(all_activities.min()-1),int(all_activities.max()+1)])
    
    if save_fig != None:
        plt.savefig(save_fig)
    if show_plot:
        plt.show()
        
    
    meta_data = {
            'train_spearmanr':spearmanr,
            }
    
    if test_set != None:
        meta_data['test_spearmanr'] = spearmanr_test

    return meta_data
