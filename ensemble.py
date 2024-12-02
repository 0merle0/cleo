import sys, os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
from scipy import stats
from torch.distributions.normal import Normal
import wandb

class BaseModel(nn.Module):
    """
    Simple MLP model with ReLU activation and dropout.
    """
    
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        
        self.trunk = nn.Sequential(
                nn.Linear(cfg.input_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.p_drop),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.p_drop),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
        
        self.mean_head = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        if cfg.predict_variance:
            self.var_head = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def forward(self, x):
        out = {}
        x = self.trunk(x)
        out["mean"] = self.mean_head(x)
        if self.cfg.predict_variance:
            # transform output with softplus to ensure positive variance
            # taken from https://arxiv.org/pdf/1612.01474
            var = self.var_head(x)
            if self.cfg.variance_transform == "softplus":
                var = torch.log(1 + torch.exp(var))
                var = torch.clamp(var, min=1e-6)
            elif self.cfg.variance_transform == "sigmoid":
                var = torch.sigmoid(var)
                var = torch.clamp(var, min=1e-6)
            elif self.cfg.variance_transform == "clamp":
                var = torch.clamp(var, min=1e-6)
            else:
                raise ValueError(f"Variance transform {self.cfg.variance_transform} not supported.")

            out["var"] = var*self.cfg.variance_scale
        return out


class Ensemble(pl.LightningModule):
    def __init__(self, cfg):

        super(Ensemble, self).__init__()
        self.cfg = cfg

        # This is a native pytorch-lightning variable
        #   setting to false will enable manual optimization.
        self.automatic_optimization = False 

        # create models
        model_list = [BaseModel(cfg.base_model) for _ in range(cfg.num_models)]
        self.models = nn.ModuleList(model_list)

        # keep track of validation predictions
        self.val_y = []
        self.val_mean = []
        self.val_var = []

    def configure_optimizers(self):
        """
        Create optimizers for each model.
        """
        optimizers = [torch.optim.Adam(model.parameters(), lr=self.cfg.lr) for model in self.models]
        return optimizers
    
    def split_batch(self, x):
        """
        Split batch into equal parts for each model.
        """
        batch_size = x.size(0)
        assert batch_size % self.cfg.num_models == 0, "Batch size must be divisible by number of models."
        split_batch_size = batch_size // self.cfg.num_models
        x_list = [x[i*split_batch_size:(i+1)*split_batch_size] for i in range(self.cfg.num_models)]
        return x_list
    
    def gaussNLL(self, mean, var, gt, reduction='mean', var_clip=1e-6):
        """
        Gaussian negative log likelihood loss.
        mean: torch.Tensor, shape: (batch, out_dim)
        var: torch.Tensor, shape: (batch, out_dim)
        y: torch.Tensor, shape: (batch, out_dim)
        reduction: str, 'mean' or 'sum'
        var_clip: float, minimum value for variance to avoid instability
        """
        # Ensure sigma is positive to avoid instability (e.g., use a clamp)
        var = torch.clamp(var, min=var_clip)

        # NLL loss from: https://arxiv.org/pdf/1612.01474
        #   modified to remove constant terms
        loss = torch.log(var) + ((gt - mean) ** 2) / var
        
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss
    
    def calc_loss(self, gt, output):
        """
        Calculate loss.
        """
        if self.cfg.loss.loss_fn == "mse":
            mean = output["mean"]
            loss = F.mse_loss(mean, gt, reduction='mean')

        elif self.cfg.loss.loss_fn == 'nll':
            assert self.cfg.base_model.predict_variance, "Variance must be predicted for NLL loss."
            mean = output["mean"]
            var = output["var"]
            loss = self.gaussNLL(mean, var, gt, reduction='mean')
        
        else:
            raise ValueError(f"Loss type {self.cfg.loss_type} not supported.")

        return loss
    
    def mix_gaussians(self, mean, var):
        """
        Mix ensemble predictions and return mu (mean) and sigma (std)
        mixture method from https://arxiv.org/pdf/1612.01474

        input: (means, vars) | torch.Tensor, shape: (batch, model, out_dim)

        returns: (mu, sigma) | torch.Tensor, shape: (batch, out_dim)
        """
        mixed_mean = mean.mean(dim=1).squeeze()
        mixed_var = (var + mean**2).mean(dim=1).squeeze() - mixed_mean**2
        return mixed_mean, torch.sqrt(mixed_var)
    
    def training_step(self, batch, batch_idx):
        """
        Training step for each model.
        """
        x, y = batch
        optimizers = self.optimizers()
        
        # Split batch amongst models
        if self.cfg.split_batch_mode:
            x_list = self.split_batch(x)
            y_list = self.split_batch(y)
        else:
            x_list = [x]*self.cfg.num_models
            y_list = [y]*self.cfg.num_models

        to_log = {}
        mean_loss = 0
        for n in range(self.cfg.num_models):
            model = self.models[n]
            _x = x_list[n]
            _y = y_list[n]
            opt = optimizers[n]

            opt.zero_grad()
            output = model(_x.float())
            loss = self.calc_loss(_y, output)
            self.manual_backward(loss)
            opt.step()
            key = f"train/model_{n+1}_{self.cfg.loss.loss_fn}"
            to_log[key] = loss.item()
            mean_loss += loss.item()/len(self.models)

        to_log[f"train/mean_{self.cfg.loss.loss_fn}"] = mean_loss
        self.log_dict(to_log)


    def validation_step(self, batch, batch_idx):
        """
        Validation step for ensemble.
        """
        x, y = batch

        to_log = {}
        output_list = []
        mean_loss = 0
        for n, model in enumerate(self.models):
            output = model(x.float())
            loss = self.calc_loss(y, output)
            key = f"val/model_{n+1}_{self.cfg.loss.loss_fn}"
            to_log[key] = loss.item()
            output_list.append(output)
            mean_loss += loss.item()/len(self.models)
        
        to_log[f"val/mean_{self.cfg.loss.loss_fn}"] = mean_loss

        self.val_y.append(y.cpu())

        mean_stack = torch.stack([o["mean"].cpu() for o in output_list])
        self.val_mean.append(mean_stack.permute(1,0,2)) # permute to (batch, model, out_dim)
        
        if "var" in output:
            var_stack = torch.stack([o["var"].cpu() for o in output_list])
            self.val_var.append(var_stack.permute(1,0,2))

        self.log_dict(to_log)

    def on_validation_epoch_end(self):
        """
        Compute ensemble metrics at end of validation epoch.
        """
        gt = torch.cat(self.val_y, dim=0)
        mean_stack = torch.cat(self.val_mean, dim=0)
        var_stack = None
        if self.cfg.base_model.predict_variance:
            var_stack = torch.cat(self.val_var, dim=0)
            mean_pred, std_pred = self.mix_gaussians(mean_stack, var_stack)
        else:
            mean_pred = mean_stack.mean(dim=1).squeeze()
            std_pred = mean_stack.std(dim=1).squeeze()
            
        to_log = {}

        output = {"mean": mean_stack, "var": var_stack}
        to_log[f"val/{self.cfg.loss.loss_fn}"] = self.calc_loss(gt, output).item()
        
        # compute pearsonr correlation
        corr, _ = stats.pearsonr(gt, mean_pred)
        to_log["val/pearsonr"] = corr

        to_log["val/std"] = std_pred.mean().item()
        
        # compute correlation between variance and squared error
        se = (gt - mean_pred)**2
        se_var_corr, _ = stats.pearsonr(se, std_pred**2)
        to_log["val/se_var_corr"] = se_var_corr

        to_log["val/mse"] = se.mean().item()

        self.log_dict(to_log)
        
        # clear lists for next epoch
        self.val_mean.clear()
        self.val_var.clear()
        self.val_y.clear()

    def posterior(self, x):
        """
        Compute the posterior.
        """
        out = self.forward(x)
        return Normal(out["mu"], out["sigma"])

    def forward(self, x):
        """
        Inference method.
        """
        output_list = []
        for model in self.models:
            output = model(x.float())
            output_list.append(output)

        mean_stack = torch.stack([o["mean"] for o in output_list]).permute(1,0,2)
        var_stack = None
        if self.cfg.base_model.predict_variance:
            var_stack = torch.stack([o["var"] for o in output_list]).permute(1,0,2)
            mu, sigma = self.mix_gaussians(mean_stack, var_stack)
        else:
            mu = mean_stack.mean(dim=1).squeeze()
            sigma = mean_stack.std(dim=1).squeeze()
        
        to_return = {
            "mu": mu, # shape: (batch, out_dim)
            "sigma": sigma, # shape: (batch, out_dim)
            "mean": mean_stack, # shape: (batch, model, out_dim)
            "var": var_stack, # shape: (batch, model, out_dim)
        }
        return to_return

    def test_step(self, batch, batch_idx):
        """
        Test step for ensemble. Collects predictions for final evaluation.
        """
        x, y = batch

        # Initialize storage lists on first batch
        if batch_idx == 0:
            self.test_y = []
            self.test_mean = []
            self.test_var = []
            self.test_min_dists = []
            self.test_mean_dists = []

        output_list = []
        for model in self.models:
            output = model(x.float())
            output_list.append(output)

        self.test_y.append(y.cpu())
        mean_stack = torch.stack([o["mean"].cpu() for o in output_list])
        self.test_mean.append(mean_stack.permute(1,0,2))

        self.test_min_dists.append(self.trainer.datamodule.test_dataset.min_dists[batch_idx*len(y):(batch_idx+1)*len(y)])
        self.test_mean_dists.append(self.trainer.datamodule.test_dataset.mean_dists[batch_idx*len(y):(batch_idx+1)*len(y)])
        
        if self.cfg.base_model.predict_variance:
            var_stack = torch.stack([o["var"].cpu() for o in output_list])
            self.test_var.append(var_stack.permute(1,0,2))

    def on_test_end(self):
        """
        Compute ensemble metrics on full test dataset and create scatter plot.
        """
        gt = torch.cat(self.test_y, dim=0)
        mean_stack = torch.cat(self.test_mean, dim=0)
        var_stack = None
        if self.cfg.base_model.predict_variance:
            var_stack = torch.cat(self.test_var, dim=0)
            mean_pred, std_pred = self.mix_gaussians(mean_stack, var_stack)
        else:
            mean_pred = mean_stack.mean(dim=1).squeeze()
            std_pred = mean_stack.std(dim=1).squeeze()
            
        to_log = {}

        # Compute metrics on full test set
        output = {"mean": mean_stack, "var": var_stack}
        # to_log[f"test/{self.cfg.loss.loss_fn}"] = self.calc_loss(gt, output).item()
        
        corr, _ = stats.pearsonr(gt.numpy().flatten(), mean_pred.numpy().flatten())
        to_log["test/pearsonr"] = corr
        to_log["test/std"] = std_pred.mean().item()
        
        se = (gt - mean_pred)**2
        se_var_corr, _ = stats.pearsonr(se.numpy().flatten(), std_pred.numpy().flatten()**2)
        to_log["test/se_var_corr"] = se_var_corr
        to_log["test/mse"] = se.mean().item()

        self.logger.experiment.log(to_log)

        # Create scatter plot
        plt.figure(figsize=(10, 10))
        plt.scatter(gt.numpy(), mean_pred.numpy(), alpha=0.5)
        plt.plot([gt.min(), gt.max()], [gt.min(), gt.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Test Predictions (Pearson r = {corr:.3f})')
        
        if self.cfg.base_model.predict_variance:
            idx = torch.randperm(len(gt))[:100]
            plt.errorbar(gt[idx].numpy(), mean_pred[idx].numpy(), 
                        yerr=2*std_pred[idx].numpy(),
                        fmt='none', alpha=0.2, color='gray')
        
        # Log everything to wandb
        self.logger.experiment.log({
            "test/predictions_scatter": wandb.Image(plt)
        })
        
        plt.close()

        if self.cfg.base_model.predict_variance:
            # 1. True fitness vs min distance plot
            min_dists = torch.cat(self.test_min_dists, dim=0).numpy()
            std_pred_numpy = std_pred.numpy()

            plt.figure(figsize=(10, 10), dpi=100)
            sns.stripplot(x=min_dists, y=std_pred_numpy, alpha=0.1)
            plt.xlabel('Minimum Distance from Training Set')
            plt.ylabel('Standard Deviation of Predictions')
            dist_std_corr, _ = stats.pearsonr(min_dists, std_pred_numpy)
            plt.title(f'Distance vs. Uncertainty (Pearson r = {dist_std_corr:.2f})')
            self.logger.experiment.log({
                "test/distance_vs_uncertainty": wandb.Image(plt)
            })
            plt.close()

            # 2. True fitness vs sigma plot
            plt.figure(figsize=(10, 10), dpi=100)
            sigma_true_corr, _ = stats.pearsonr(gt.numpy().flatten(), std_pred.numpy().flatten())
            plt.scatter(gt.numpy(), std_pred.numpy(), alpha=0.1)
            plt.xlabel('True Fitness')
            plt.ylabel('Sigma (std)')
            plt.title(f'True Fitness vs. Uncertainty (Pearson r = {sigma_true_corr:.2f})')
            self.logger.experiment.log({
                "test/sigma_vs_true": wandb.Image(plt)
            })
            plt.close()

            # 3. Squared error vs sigma plot
            plt.figure(figsize=(10, 10), dpi=100)
            plt.scatter(se.numpy(), std_pred.numpy(), alpha=0.1)
            plt.xlabel('Squared Error')
            plt.ylabel('Standard Deviation of Predictions')
            plt.title(f'Error vs. Uncertainty (Pearson r = {se_var_corr:.2f})')
            self.logger.experiment.log({
                "test/sigma_vs_error": wandb.Image(plt)
            })
            plt.close()
        
        # Clean up
        del self.test_y
        del self.test_mean
        del self.test_var
        del self.test_min_dists
        del self.test_mean_dists
    