import sys, os
import json
import torch
import numpy as np
from tqdm import tqdm
import time
import hydra
import wandb
from omegaconf import OmegaConf
from policy_utils import PolicyMPNN

sys.path.append("/software/lab/mpnn/fused_mpnn")
from data_utils import featurize, parse_PDB
from model_utils import ProteinMPNN

PROTEIN_MPNN_CKPT_PATH = "/databases/mpnn/vanilla_model_weights/v_48_020.pt"
LIGAND_MPNN_CKPT_PATH = "/databases/mpnn/ligand_mpnn_model_weights/s25_r010_t300_p.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define mpnn constants
restype_1to3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', 'X': 'UNK'}
restype_STRtoINT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}
restype_INTtoSTR = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X'}
alphabet = list(restype_STRtoINT)


class PolicyMPNNvDAPO(PolicyMPNN):
    def __init__(self, cfg):
        super().__init__(cfg)

        # load reference MPNN model
        if hasattr(self.cfg, "use_ref_kl") and self.cfg.use_ref_kl:
            self.ref_mpnn = self.load_ref_mpnn_model()

    def load_ref_mpnn_model(self):
        """
        Load the MPNN model based on the configuration.
        """

        model_type = self.cfg.model_type

        if model_type == "protein_mpnn":
            self.atom_context_num = 1
            k_neighbors = 48
            self.ligand_mpnn_use_side_chain_context = 0
            ckpt_path = PROTEIN_MPNN_CKPT_PATH

        elif model_type == "ligand_mpnn":
            self.atom_context_num = 25
            k_neighbors = 32
            self.ligand_mpnn_use_side_chain_context = 0
            ckpt_path = LIGAND_MPNN_CKPT_PATH

        else:
            raise ValueError("Invalid model type specified. Choose 'ligand_mpnn' or 'protein_mpnn'.")


        # load model
        model = ProteinMPNN(node_features=128,
                        edge_features=128,
                        hidden_dim=128,
                        num_encoder_layers=3,
                        num_decoder_layers=3,
                        k_neighbors=k_neighbors,
                        device=self.device,
                        atom_context_num=self.atom_context_num,
                        model_type=model_type,
                        ligand_mpnn_use_side_chain_context=self.ligand_mpnn_use_side_chain_context)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return model.to(self.device)

    def train_step(self, step, init_state, feature_dict):
        """
        Single training step given featurized example

        EDITED: for DAPO algortihm - http://arxiv.org/pdf/2503.14476
        """

        # Collect Experience
        to_log = {}
        with torch.no_grad():
            # run the policy
            h_V_in, h_E_in, E_idx_in = init_state
            out = self.rollout(feature_dict, h_V_in, h_E_in, E_idx_in)

            # compute reward
            batched_rewards, metrics = self.reward_fn(step, out, feature_dict, self.device)
            to_log.update(metrics)
            to_log["reward"] = batched_rewards.mean().cpu().item()


        # create a big batch of all rollouts
        all_log_probs = torch.clone(out["log_probs"]) #[B, L, 21]
        all_S = torch.clone(out["S"])
        all_decoding_order = torch.clone(out["decoding_order"]) #[B, L]
        all_batched_rewards = torch.clone(batched_rewards) #[B, L]
        B = all_S.shape[0] # total number of sequences in the batch


        # Update the policy
        print(f"Training step {step} with {self.cfg.N_updates} updates")
        dapo_losses = []                      # <-- track loss over updates
        for i in range(self.cfg.N_updates):

            # sample a random rollout from the collected data
            sample_idx = torch.randperm(B)[:self.cfg.update_batch_size]
            S = all_S[sample_idx] #[B, L]
            decoding_order = all_decoding_order[sample_idx] #[B, L]
            old_batched_log_probs = all_log_probs[sample_idx] #[B, L, 21]
            batched_rewards = all_batched_rewards[sample_idx] #[B, L]

            self.optimizer.zero_grad()

            h_V_in, h_E_in, E_idx_in = init_state

            # turn on grads for state features
            h_V_in.requires_grad = True
            h_E_in.requires_grad = True

            # add if using KL here then run through the reference model
            if hasattr(self.cfg, "use_ref_kl") and self.cfg.use_ref_kl:
                with torch.no_grad():
                    ref_out = self.rollout(feature_dict, h_V_in, h_E_in, E_idx_in,
                                           decoding_order=decoding_order, sampled_actions=S, model=self.ref_mpnn)
                    ref_batched_log_probs = ref_out["log_probs"] #[B, L, 21]

            out = self.rollout(feature_dict, h_V_in, h_E_in, E_idx_in,
                                    decoding_order=decoding_order, sampled_actions=S, model=self.model)

            # mask for what was actually decoded in the sequence
            seq_mask = torch.nn.functional.one_hot(out["S"], num_classes=len(alphabet)).float()

            # apply mask and take sum over each seq in the batch
            # compute ratio r = P_new / P_old
            batched_log_probs = (out["log_probs"] * seq_mask).sum(dim=(-1)) # new
            old_batched_log_probs = (old_batched_log_probs * seq_mask).sum(dim=(-1)) # old
            r = torch.exp(batched_log_probs - old_batched_log_probs) # ratio P_new / P_old

            # compute advantage
            A = (batched_rewards - torch.mean(batched_rewards)) / (torch.std(batched_rewards) + 1e-8)

            if A.shape != r.shape:
                # if reward is not per residue, we need to sum over log probs
                min_term1 = r.sum(dim=-1) * A
                min_term2 = torch.clamp(r, 1-self.cfg.clip_eps_low, 1+self.cfg.clip_eps_high).sum(dim=-1) * A
            else:
                # else reward is per residue, so apply per residue reward, 
                #   dont sum and instead take mean in next step over all fragments
                min_term1 = r * A
                min_term2 = torch.clamp(r, 1-self.cfg.clip_eps_low, 1+self.cfg.clip_eps_high) * A

            # DAPO objective
            obj = torch.min(min_term1, min_term2).mean()

            if hasattr(self.cfg, "use_ref_kl") and self.cfg.use_ref_kl:
                # add KL penalty between current policy and reference MPNN
                ref_batched_log_probs = (ref_batched_log_probs * seq_mask).sum(dim=(-1))  # [B]
                kl_ratio = torch.exp(ref_batched_log_probs - batched_log_probs)  # safe because logp_ref - logp_theta is small-ish
                kl = kl_ratio - (ref_batched_log_probs - batched_log_probs) - 1
                obj = obj - self.cfg.kl_weight * kl.mean()
                to_log["kl_penalty"] = kl.mean().cpu().item()

            # DAPO objective
            # want to maximize this objective, so we take the negative
            dapo_loss = -obj

            # optimizer update
            dapo_loss.backward()
            self.optimizer.step()

            dapo_losses.append(dapo_loss.detach().cpu().item())


        # ---------- metric aggregation ----------
        if dapo_losses:                       # policy metrics
            to_log["policy/dapo_loss_mean"] = float(np.mean(dapo_losses))
            to_log["policy/dapo_loss_last"] = dapo_losses[-1]

        # reward statistics (from the original rollout batch)
        to_log.update({
            "rewards/mean": all_batched_rewards.mean().cpu().item(),
            "rewards/std":  all_batched_rewards.std().cpu().item() \
                             if all_batched_rewards.numel() > 1 else 0.0,
            "rewards/min":  all_batched_rewards.min().cpu().item(),
            "rewards/max":  all_batched_rewards.max().cpu().item(),
        })

        if wandb.run:
            # ---------- Weights & Biases logging ----------
            wandb_log = {}
            if metrics:                       # reward-fn specific metrics
                wandb_log.update({f"reward_metrics/{k}": v for k, v in metrics.items()})
            wandb_log.update(to_log)          # include policy + reward stats
            wandb.log(wandb_log, step=step)   # don't finish the row yet
            
        # self._log_metrics_to_csv(step, to_log)

        return to_log
