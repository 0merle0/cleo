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
from scipy import stats

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
        self.models = nn.ModuleList([MLP(**self.mlp_cfg) for _ in range(self.num_models)])

        # keep track of validation predictions
        self.val_y = []
        self.val_y_hat = []

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
            loss = self.mse_loss(y_hat, y.unsqueeze(-1))
            self.manual_backward(loss)
            opt.step()
            to_log[f"train/model_{n+1}_mse"] = loss.item()

        self.log_dict(to_log)


    def validation_step(self, batch, batch_idx):
        """Validation step for ensemble."""
        x, y = batch

        to_log = {}
        y_hat_list = []
        for n, model in enumerate(self.models):
            y_hat = model(x)
            loss = self.mse_loss(y_hat, y.unsqueeze(-1))
            to_log[f"val/model_{n+1}_mse"] = loss.item()
            y_hat_list.append(y_hat)

        y_hat_stack = torch.stack(y_hat_list)
        self.val_y.append(y)
        self.val_y_hat.append(y_hat_stack.permute(1,0,2)) # permute to (batch, model, out_dim)
        self.log_dict(to_log)

    def on_validation_epoch_end(self):
        y_true = torch.cat(self.val_y, dim=0)
        y_pred = torch.cat(self.val_y_hat, dim=0)
        
        to_log = {}

        pred_mean = y_pred.mean(dim=1).squeeze()
        to_log['val/mse'] = self.mse_loss(pred_mean,y_true).item()
        
        # compute pearsonr correlation
        corr, _ = stats.pearsonr(pred_mean,y_true)
        to_log['val/pearsonr'] = corr

        pred_std = y_pred.std(dim=1)
        to_log['val/std'] = pred_std.mean().item()
        
        self.log_dict(to_log)
        
        # clear lists for next epoch
        self.val_y_hat.clear()
        self.val_y.clear()

    def forward(self, x):
        """Forward pass for each model."""
        return torch.stack([model(x) for model in self.models]).permute(1,0,2) # permute to (batch, model, out_dim)
    