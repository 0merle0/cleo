"""
Reward step that uses a pre-trained experimental activity predictor
(ensemble MLP) to score sampled sequences. Typically used in later
rounds of optimisation once experimental data is available for training
a sequence-to-function model.
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

from cleo.optimize.utils.ensemble import Ensemble
from cleo.optimize.utils.train_data import SequenceFunctionDataset


@torch.no_grad()
def experimental_predictor_from_df(df_input, cfg, step_name="experimental_predictor"):
    """Run an ensemble predictor on sequences and return mean/sigma columns."""

    ckpt = torch.load(os.path.join(cfg.ckpt_path, cfg.ckpt_name), map_location=torch.device('cpu'))
    model_config = OmegaConf.load(os.path.join(cfg.ckpt_path, 'config.yaml'))

    model = Ensemble(model_config)
    model.load_state_dict(ckpt['state_dict'])

    label_col = model_config.data.dataset_cfg.label_col
    if label_col not in df_input.columns:
        df_input[label_col] = 0.0

    dataset = SequenceFunctionDataset(model_config.data.dataset_cfg, df_input)

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
