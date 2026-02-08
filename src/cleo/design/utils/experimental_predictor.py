import sys, os
import torch
import pandas as pd
import glob
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np


local_user_path = "/home/jgershon/git/cleo"
sys.path.append(local_user_path) 
from cleo.optimization.utils.ensemble import Ensemble
from cleo.optimization.utils.train_data import SequenceFunctionDataset



@torch.no_grad()
def experimental_predictor_from_df(df_input, cfg, step_name="experimental_predictor"):


    # load checkpoint and config
    ckpt = torch.load(os.path.join(cfg.ckpt_path, cfg.ckpt_name), map_location=torch.device('cpu'))
    model_config = OmegaConf.load(os.path.join(cfg.ckpt_path,'config.yaml'))

    # load model
    model = Ensemble(model_config)
    model.load_state_dict(ckpt['state_dict'])


    # get dataset
    label_col = model_config.data.dataset_cfg.label_col
    # make spoof label col if not present
    if label_col not in df_input.columns:
        df_input[label_col] = 0.0

     # make dataset
    dataset = FragmentDataset(model_config.data.dataset_cfg, df_input)
    
    # run through the dataset
    mu = []
    sigma = []
    for x, y in tqdm(dataset):
        output = model(x[None])
        mu.append(output["mu"].item())
        sigma.append(output["sigma"].item())


    df_output = pd.DataFrame({
        "name": df_input["name"].tolist(),
        f"{step_name}_mu": mu,
        f"{step_name}_sigma": sigma,
    })

    df_merged = pd.merge(df_input, df_output, on="name", how="inner")

    return df_merged

    