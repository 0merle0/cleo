import sys, os, copy
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
import logging
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import parameters_to_vector


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(AttentionPooling, self).__init__()
        self.query_vector = nn.Parameter(torch.randn(1, embed_dim))
        self.key_proj = nn.Linear(input_dim, embed_dim)
        self.value_proj = nn.Linear(input_dim, embed_dim)
        self.scale = embed_dim**0.5

    def forward(self, x, mask=None):
        """Tensor of shape (B, L, D) where B is batch size, L is sequence length,
        and D is feature dimension output shape (B, D).
        """
        K = self.key_proj(x)
        if mask != None:
            K * mask.unsqueeze(-1)
        V = self.value_proj(x)
        Q_cls = self.query_vector.expand(x.size(0), -1, -1)

        # Compute attention scores
        attention_scores = torch.bmm(Q_cls, K.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        context_vector = torch.bmm(attention_weights, V)
        # Residual Connection
        context_vector += Q_cls

        return context_vector.squeeze(1)


class BaseModel(nn.Module):
    """
    Simple MLP model with ReLU activation and dropout.
    """

    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg

        activation = nn.ReLU()
        dropout = nn.Dropout(self.cfg.p_drop)

        def build_output_head(hidden_dim, output_dim, head_type):
            if head_type == "mlp":
                out_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    activation,
                    nn.Linear(hidden_dim, hidden_dim),
                    activation,
                    nn.Linear(hidden_dim, output_dim),
                )
            elif head_type == "linear":
                out_head = nn.Linear(hidden_dim, output_dim)
            else:
                raise ValueError(f"Output head type {head_type} not supported.")
            return out_head


        if self.cfg.model_type == "linear":
            self.cfg.hidden_dim = self.cfg.input_dim

        elif self.cfg.model_type == "mlp":
            self.trunk = nn.Sequential(
                nn.Linear(self.cfg.input_dim, self.cfg.hidden_dim),
                activation,
                dropout,
                nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
                activation,
                dropout,
                nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            )

        elif self.cfg.model_type == "conv1d":
            self.trunk = nn.Sequential(
                nn.Conv1d(
                    self.cfg.input_dim, self.cfg.hidden_dim, self.cfg.kernel_size
                ),
                dropout,
                activation,
                nn.Conv1d(
                    self.cfg.hidden_dim, self.cfg.hidden_dim, self.cfg.kernel_size
                ),
                activation,
                dropout,
            )
            self.attn_pool = AttentionPooling(self.cfg.hidden_dim, self.cfg.hidden_dim)
        else:
            raise ValueError(f"Model type {self.cfg.model_type} not supported.")

        self.mean_head = build_output_head(self.cfg.hidden_dim, self.cfg.output_dim, self.cfg.output_head_type)
        if self.cfg.predict_variance:
            if self.cfg.concat_mean_pred:
                self.var_head = build_output_head(self.cfg.hidden_dim+1, self.cfg.output_dim, self.cfg.output_head_type)
            else:
                self.var_head = build_output_head(self.cfg.hidden_dim, self.cfg.output_dim, self.cfg.output_head_type)

    def forward(self, x):
        if self.cfg.model_type == "conv1d":
            x = x.permute(0, 2, 1)
            x = self.trunk(x)
            x = x.permute(0, 2, 1)
            x = self.attn_pool(x)
        elif self.cfg.model_type == "mlp":
            x = self.trunk(x)

        mean = self.mean_head(x)

        if self.cfg.predict_variance:
            if self.cfg.stop_grad_variance:
                x = x.detach()
            if self.cfg.concat_mean_pred:
                x = torch.cat([x, mean.detach()],dim=-1)
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
                raise ValueError(
                    f"Variance transform {self.cfg.variance_transform} not supported."
                )
        else:
            var = torch.ones_like(mean) * self.cfg.fixed_variance

        ret = {
            "mean": mean,  # shape: (batch, out_dim)
            "var": var,  # shape: (batch, out_dim)
        }
        return ret


class Ensemble(pl.LightningModule):
    def __init__(self, cfg):

        super(Ensemble, self).__init__()
        self.full_cfg = cfg
        self.cfg = cfg.model

        # This is a native pytorch-lightning variable
        #   setting to false will enable manual optimization.
        self.automatic_optimization = False

        # create models
        model_list = [
            BaseModel(self.cfg.base_model) for _ in range(self.cfg.num_models)
        ]
        self.models = nn.ModuleList(model_list)

        # keep track of validation predictions
        self.val_y = []
        self.val_mean = []
        self.val_var = []

    def configure_optimizers(self):
        """
        Create optimizers for each model.
        """
        optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
            for model in self.models
        ]
        return optimizers

    def split_batch(self, x):
        """
        Split batch into equal parts for each model.
        """
        # NOTE: this is not currently being used
        batch_size = x.size(0)
        assert (
            batch_size % self.cfg.num_models == 0
        ), "Batch size must be divisible by number of models."
        split_batch_size = batch_size // self.cfg.num_models
        x_list = [
            x[i * split_batch_size : (i + 1) * split_batch_size]
            for i in range(self.cfg.num_models)
        ]
        return x_list

    def gaussNLL(self, mean, var, gt, reduction="mean", var_clip=1e-6):
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

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def parameter_regularization(self, model, reg_order, reg_lambda):
        """
        Regularize model parameters.
        """
        all_params = parameters_to_vector(model.parameters())
        norm = (torch.abs(all_params)**reg_order).sum()
        return norm * reg_lambda

    def calc_loss(self, gt, output, model):
        """
        Calculate loss.
        """
        mean = output["mean"]
        var = output["var"]
        
        mse_weight = self.cfg.loss.mse_weight
        mse_loss = F.mse_loss(mean, gt, reduction="mean")

        nll_weight = self.cfg.loss.nll_weight
        nll_loss = self.gaussNLL(mean, var, gt, reduction="mean")
        
        reg_order = self.cfg.loss.regularization_order
        reg_lambda = self.cfg.loss.regularization_lambda
        reg_loss = self.parameter_regularization(model, reg_order, reg_lambda)

        loss = mse_loss*mse_weight + nll_loss*nll_weight + reg_loss

        loss_dict = {
            "mse": mse_loss,
            "nll": nll_loss,
            f"regularization_l{reg_order}": reg_loss,
            "loss": loss,
            }

        return loss, loss_dict

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

        to_log = {}
        mean_loss = 0
        for n in range(self.cfg.num_models):
            model = self.models[n]
            opt = optimizers[n]
            opt.zero_grad()
            output = model(x.float())
            loss, loss_dict = self.calc_loss(y, output, model)
            self.manual_backward(loss)
            opt.step()

            to_log.update({f"train/{k}_model_{n+1}":v for k,v in loss_dict.items()})
            mean_loss += loss.item() / len(self.models)

        to_log["train/mean_loss"] = mean_loss
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
            loss, loss_dict = self.calc_loss(y, output, model)

            output_list.append(output)
            to_log.update({f"val/{k}_model_{n+1}":v for k,v in loss_dict.items()})
            mean_loss += loss.item() / len(self.models)

        to_log["val/mean_loss"] = mean_loss

        self.val_y.append(y.cpu())

        mean_stack = torch.stack([o["mean"].cpu() for o in output_list])
        self.val_mean.append(
            mean_stack.permute(1, 0, 2)
        )  # permute to (batch, model, out_dim)

        var_stack = torch.stack([o["var"].cpu() for o in output_list])
        self.val_var.append(
            var_stack.permute(1, 0, 2)
        ) # permute to (batch, model, out_dim)

        self.log_dict(to_log)

    def on_validation_epoch_end(self):
        """
        Compute ensemble metrics at end of validation epoch.
        """
        gt = torch.cat(self.val_y, dim=0).squeeze()
        mean_stack = torch.cat(self.val_mean, dim=0)
        var_stack = torch.cat(self.val_var, dim=0)
        mean_pred, std_pred = self.mix_gaussians(mean_stack, var_stack)

        to_log = {}
        to_log["val/std_mean"] = std_pred.mean().item()
        to_log["val/std_min"] = std_pred.min().item()
        to_log["val/std_max"] = std_pred.max().item()

        to_log["val/nll"] = self.gaussNLL(
            mean_pred, std_pred**2, gt, reduction="mean"
        )

        # compute pearsonr correlation
        pearsonr_corr, _ = stats.pearsonr(gt, mean_pred)
        to_log["val/pearsonr"] = pearsonr_corr

        if not self.full_cfg.debug:
            # save plot of ground truth vs. prediction
            plt.figure(dpi=150)
            sns.scatterplot(x=gt, y=mean_pred, alpha=0.5)
            plt.xlabel("Ground Truth")
            plt.ylabel("Predicted Mean")
            plt.title(f"Ground Truth vs. Prediction | pearsonR {pearsonr_corr:.3f}")
            plt.savefig(f"{self.cfg.ckpt_dir}/val_gt_pred_scatter.png")
            plt.close()

        # compute spearmanr correlation
        spearmanr_corr, _ = stats.spearmanr(gt, mean_pred)
        to_log["val/spearmanr"] = spearmanr_corr

        # compute correlation between variance and squared error
        se = (gt - mean_pred) ** 2
        se_var_corr, _ = stats.pearsonr(se, std_pred**2)
        to_log["val/se_var_corr"] = se_var_corr

        if not self.full_cfg.debug:
            # save plot for variance vs. squared error
            plt.figure(dpi=150)
            sns.scatterplot(x=se, y=std_pred**2, alpha=0.5)
            plt.xlabel("Squared Error")
            plt.ylabel("Predicted Variance")
            plt.title(
                f"Squared Error vs. Predicted Variance | pearsonR {se_var_corr:.3f}"
            )
            plt.savefig(f"{self.cfg.ckpt_dir}/val_se_var_scatter.png")
            plt.close()

        if not self.full_cfg.debug:
            # save plot for mu vs. sigma
            plt.figure(dpi=150)
            sns.scatterplot(x=mean_pred, y=std_pred, alpha=0.5)
            plt.xlabel("Mu pred")
            plt.ylabel("Sigma pred")
            plt.title(
                "Correlation between mu and sigma"
            )
            plt.savefig(f"{self.cfg.ckpt_dir}/val_mu_sigma_scatter.png")
            plt.close()

        to_log["val/mse"] = se.mean().item()

        # compute top-k accuracy
        k_list = self.cfg.top_k_accuracy
        for k in k_list:
            if k > gt.size(0):
                break
            gt_topk_values, gt_topk_indices = gt.topk(
                k, dim=0, largest=True, sorted=True
            )
            pred_topk_values, pred_topk_indices = mean_pred.topk(
                k, dim=0, largest=True, sorted=True
            )
            correct_topk = torch.isin(pred_topk_indices, gt_topk_indices)
            topk_accuracy = correct_topk.sum().float() / gt_topk_indices.size(0)

            to_log[f"val/top{k}_accuracy"] = topk_accuracy.item()

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

        mean_stack = torch.stack([o["mean"] for o in output_list]).permute(1, 0, 2)
        var_stack = None
        if self.cfg.base_model.predict_variance:
            var_stack = torch.stack([o["var"] for o in output_list]).permute(1, 0, 2)
            mu, sigma = self.mix_gaussians(mean_stack, var_stack)
        else:
            mu = mean_stack.mean(dim=1).squeeze()
            sigma = mean_stack.std(dim=1).squeeze()

        to_return = {
            "mu": mu,  # shape: (batch, out_dim)
            "sigma": sigma,  # shape: (batch, out_dim)
            "mean": mean_stack,  # shape: (batch, model, out_dim)
            "var": var_stack,  # shape: (batch, model, out_dim)
        }
        return to_return
