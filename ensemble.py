import torch
import botorch
import scipy
from tqdm import tqdm
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

class MLP(nn.Module):
    """Simple MLP model with ReLU activation and dropout."""
    
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                out_dim,
                p_drop=0.2,
                ):
        super(MLP, self).__init__()
        
        self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden_dim, out_dim),
            )
    def forward(self, x):
        return self.layers(x.float())


class Ensemble(pl.LightningModule):
    def __init__(self, cfg):

        super(Ensemble, self).__init__()
        self.num_models = cfg.num_models
        self.mlp_cfg = cfg.mlp_cfg
        self.lr = cfg.lr
        self.mse_loss = nn.MSELoss()

        self.automatic_optimization = False # we will handle optimization manually

        # create models
        self.models = [MLP(**self.mlp_cfg) for _ in range(self.num_models)]

        self.save_hyperparameters()

    def configure_optimizers(self):
        """Create optimizers for each model."""
        optimizers = [torch.optim.Adam(model.parameters(), lr=self.lr) for model in self.models]
        return optimizers
    
    def training_step(self, batch, batch_idx):
        """Training step for each model."""
        x, y = batch
        optimizers = self.optimizers()

        to_log = {}
        for n, (model, opt) in enumerate(zip(self.models, optimizers)):
            opt.zero_grad()
            y_hat = model(x)
            loss = self.mse_loss(y_hat, y)
            self.manual_backward(loss)
            opt.step()
            to_log[f"train/model_{n+1}_mse"] = loss.item()

        self.log_dict(to_log)

    def forward(self, x):
        """Forward pass for each model."""
        return torch.stack([model(x) for model in self.models])
    