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
        self.cfg = cfg
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
    
    def split_batch(self, x):
        """Split batch into equal parts for each model."""
        batch_size = x.size(0)
        assert batch_size % self.num_models == 0, "Batch size must be divisible by number of models."
        split_batch_size = batch_size // self.num_models
        x_list = [x[i*split_batch_size:(i+1)*split_batch_size] for i in range(self.num_models)]
        return x_list
    
    def training_step(self, batch, batch_idx):
        """Training step for each model."""
        x, y = batch
        optimizers = self.optimizers()
        
        # Split batch amongst models
        if self.cfg.split_batch_mode:
            x_list = self.split_batch(x)
            y_list = self.split_batch(y)
        else:
            x_list = [x]*self.num_models
            y_list = [y]*self.num_models

        to_log = {}
        for n in range(self.num_models):
            model = self.models[n]
            _x = x_list[n]
            _y = y_list[n]
            opt = optimizers[n]

            opt.zero_grad()
            y_hat = model(_x)
            loss = self.mse_loss(y_hat, _y.unsqueeze(-1))
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
        """Compute ensemble metrics at end of validation epoch."""
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
    