import sys, os
import pandas as pd
import numpy as np

import torch
from omegaconf import OmegaConf
import hydra

from ensemble import Ensemble
from data_util import FragmentDataset
from tqdm import tqdm

@torch.no_grad()
@hydra.main(version_base=None, config_path="./config", config_name="eval_config")
def prediction(cfg):
    OmegaConf.set_struct(cfg, False) # allow for dynamic attribute assignment to be modified dynamically at runtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # check if gpu is available
    print(f"Using device: {device}")
    
    # load checkpoint
    ckpt = torch.load(os.path.join(cfg.base_path, cfg.ckpt_name), map_location=torch.device('cpu'))
    config = OmegaConf.load(os.path.join(cfg.base_path,'config.yaml'))
    
    # load model
    model = Ensemble(config)
    model.load_state_dict(ckpt['state_dict'])   # load model state _dict from checkpoint
    model = model.to(device)   # move model to gpu if available

    # load full super dataset (324k samples)
    data = pd.read_csv(cfg.dataset_path)    # full dataset
    #data = data.sample(500)  # sample smaller dataset
    dataset = FragmentDataset(config.data.dataset_cfg, data)

    # run through the dataset
    mu = [] # mean of the ensemble models
    sigma = [] # standard deviation of the ensemble models
    ucb = []  # upper bound confidence bound of the ensemble models
    lcb = []  # lower bound confidence bound of the ensemble models
    # pred_mean = []   # mean of the individual models in the ensemble 
    # pred_var = []    # standard deviation of the individual models in the ensemble 

    for i in tqdm(range(len(dataset))):
        x, y = dataset[i] #  x has the input features, y is the output label  
        x = x.to(device)    # move x to the gpu. y is not moved to the gpu because it is not used in the forward pass.
        output = model(x[None])
        output = {k: v.cpu() for k, v in output.items()}   # move  back to cpu
        mu.append(output["mu"].item())
        sigma.append(output["sigma"].item())
        ucb.append(output["mu"].item()+output["sigma"].item())
        lcb.append(output["mu"].item()-output["sigma"].item())
        # pred_mean.append(torch.flatten(output["mean"]).tolist())
        # pred_var.append(torch.flatten(output["var"]).tolist())

    data['mu_pred'] = mu     # predicted mean from the model
    data['sigma_pred'] = sigma   # predicted standard deviation from the model
    data['ucb_pred'] = ucb    # predicted upper bound confidence from the model
    data['lcb_pred'] = lcb    # predicted lower bound confidence from the model
    data['model_base_path'] = cfg.base_path    # base_path from the model

    data.to_csv(os.path.join(cfg.base_path, f'{cfg.run_prefix}_predictions.csv'), index=False)

if __name__ == "__main__":
    prediction()