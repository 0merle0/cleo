import sys, os
import json
import torch
import numpy as np
from tqdm import tqdm
import time
import hydra
from omegaconf import OmegaConf
import pandas as pd  # pandas for CSV fallback logging
import pickle
from pathlib import Path
import torch.nn.functional as F

sys.path.append("/software/lab/mpnn/fused_mpnn")
from data_utils import featurize, parse_PDB
from model_utils import ProteinMPNN
import wandb

PROTEIN_MPNN_CKPT_PATH = "/databases/mpnn/vanilla_model_weights/v_48_020.pt"
LIGAND_MPNN_CKPT_PATH = "/databases/mpnn/ligand_mpnn_model_weights/s25_r010_t300_p.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define mpnn constants
restype_1to3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', 'X': 'UNK'}
restype_STRtoINT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}
restype_INTtoSTR = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X'}
alphabet = list(restype_STRtoINT)


class PolicyMPNN:
    def __init__(self, cfg, eval_mode=False):

        self.cfg = cfg
        self.eval_mode = eval_mode
        self.device = DEVICE
        self.run_name = cfg.run_name
        self.output_dir = os.path.join(cfg.output_dir, cfg.run_name)
        
        # create output directory if it does not exist
        os.makedirs(self.output_dir, exist_ok=True)

        # save config in output directory
        OmegaConf.save(config=cfg, f=os.path.join(self.output_dir, f"{self.run_name}_config.yaml"))

        # log reward history
        self.reward_history = [0]

        # load model
        self.model = self.load_mpnn_model()
        self.model.to(self.device)

        if self.cfg.eval:
            self.model.eval()
            self.cfg.reward = self.cfg.evaluate.reward

        # load optimizer
        self.optimizer = self.get_optimizer()

        # defaults from mpnn
        self.ligand_mpnn_use_atom_context = 1
        self.ligand_mpnn_cutoff_for_score = 8.0

        # get reward function
        self.reward_fn = hydra.utils.instantiate(cfg.reward)

        # checkpointing utils
        self.checkpoint_every_n_steps = self.cfg.checkpoint_every_n_steps
        self.best_seen_reward = 0
        self.step_at_best_seen_reward = 0


    def load_mpnn_model(self):
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

        
        # load checkpoint if provided
        if self.eval_mode:
            if hasattr(self.cfg, 'evaluate') and self.cfg.evaluate.get('chkpt_path'):
                print(f"Loading evaluation checkpoint from {self.cfg.evaluate.chkpt_path}")
                ckpt_path = self.cfg.evaluate.chkpt_path
            else:
                # Auto-construct path to last checkpoint from training run
                ckpt_path = os.path.join(self.output_dir, f"{self.cfg.run_name}_last.pt")
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist")
        else:
            if self.cfg.get('checkpoint_path'):
                print(f"Loading training checkpoint from {self.cfg.checkpoint_path}")
                ckpt_path = self.cfg.checkpoint_path
            else:
                print(f"Using default MPNN checkpoint: {ckpt_path}")

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

        # Determine if we're loading a default MPNN checkpoint or a custom checkpoint
        is_default_checkpoint = ckpt_path in [PROTEIN_MPNN_CKPT_PATH, LIGAND_MPNN_CKPT_PATH]        
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=is_default_checkpoint)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return model.to(self.device)

    def get_optimizer(self):
        """
        Define optimizer over decoder params
        """
        
        # turn off encoder weights
        for name, param in self.model.named_parameters():
            if "decoder_layers" in name or "W_out" in name:
                continue
            else:
                param.requires_grad = False


        # only provide optimizer with unfrozen params
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.lr
        )
        return optimizer

    def featurize_pdb(self, pdb):
        """
        Get MPNN features from PDB file.
        """

        #parse PDB file
        protein_dict, backbone, other_atoms, icodes, water_atoms = parse_PDB(pdb,
                                                                            device=self.device, 
                                                                            atom_context_num=self.atom_context_num, 
                                                                            chains="",
                                                                            parse_all_atoms=self.ligand_mpnn_use_side_chain_context)

        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
        chain_letters_list = list(protein_dict["chain_letters"])
        encoded_residues = []
        for i in range(len(R_idx_list)):
            tmp = str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
            encoded_residues.append(tmp)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(zip(list(range(len(encoded_residues))), encoded_residues))


        chain_mask = torch.tensor(np.array([True for item in protein_dict["chain_letters"]],dtype=np.int32), device=self.device)
        protein_dict["chain_mask"] = chain_mask

        # fixed residues
        fixed_residues = [item for item in self.cfg.fixed_residues.split()]
        fixed_positions = torch.tensor([int(item not in fixed_residues) for item in encoded_residues], device=self.device)
        if fixed_residues:
            protein_dict["chain_mask"] = protein_dict["chain_mask"] * fixed_positions

            
        protein_dict["side_chain_mask"] = protein_dict["chain_mask"]

        # also from mpnn args
        omit_AA_list = self.cfg.omit_AA if self.cfg.omit_AA is not None else []
        omit_AA = torch.tensor(np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32), device=self.device)

        bias_AA_per_residue = torch.zeros([len(encoded_residues),21], device=self.device, dtype=torch.float32)
        omit_AA_per_residue = torch.zeros([len(encoded_residues),21], device=self.device, dtype=torch.float32)

        feature_dict = featurize(protein_dict,
                                cutoff_for_score=self.ligand_mpnn_cutoff_for_score, 
                                use_atom_context=self.ligand_mpnn_use_atom_context,
                                number_of_ligand_atoms=self.atom_context_num,
                                model_type=self.cfg.model_type)
        feature_dict["batch_size"] = self.cfg.batch_size
        B, L, _, _ = feature_dict["X"].shape #batch size should be 1 for now.
        #----

        #add additional keys to the feature dictionary
        feature_dict["temperature"] = self.cfg.temperature
        bias_AA = torch.zeros([21], device=self.device, dtype=torch.float32)
        feature_dict["bias"] = (-1e8*omit_AA[None,None,:]+bias_AA).repeat([1,L,1])+bias_AA_per_residue[None]-1e8*omit_AA_per_residue[None]

        feature_dict["symmetry_residues"] = [[]]
        feature_dict["symmetry_weights"] = [[]]
        #----

        feature_dict["randn"] = torch.randn([feature_dict["batch_size"], feature_dict["mask"].shape[1]], device=self.device)

        return feature_dict

    def encode_initial_state(self, feature_dict):
        """
        Run the MPNN model without gradient tracking.
        """
        with torch.no_grad():
            h_V, h_E, E_idx = self.model.encode(feature_dict)
        return h_V, h_E, E_idx

    def gather_nodes(self, nodes, neighbor_idx):
        """
        Copy from MPNN Utils
        """
        # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
        # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
        neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
        neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
        # Gather and re-pack
        neighbor_features = torch.gather(nodes, 1, neighbors_flat)
        neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
        return neighbor_features

    def cat_neighbors_nodes(self, h_nodes, h_neighbors, E_idx):
        """
        Copy from MPNN Utils
        """
        h_nodes = self.gather_nodes(h_nodes, E_idx)
        h_nn = torch.cat([h_neighbors, h_nodes], -1)
        return h_nn


    def rollout(self, 
                feature_dict, 
                h_V, 
                h_E, 
                E_idx,
                decoding_order=None, # B, L
                sampled_actions=None, # B, L
                model=None
            ):
        """
        Ripped from fused MPNN decoding, modified to allow grads to flow through this pass
        """

        # decode
        if model is None:
            model = self.model

        if sampled_actions is None:
            B_decoder = feature_dict["batch_size"]
            S_true = feature_dict["S"] #[B,L] - integer proitein sequence encoded using "restype_STRtoINT
            mask = feature_dict["mask"] #[B,L] - mask for missing regions - should be removed! all ones most of the time
            chain_mask = feature_dict["chain_mask"] #[B,L] - mask for which residues need to be fixed; 0.0 - fixed; 1.0 - will be designed
            bias = feature_dict["bias"] #[B,L,21] - amino acid bias per position
            randn = feature_dict["randn"] #[B,L] - random numbers for decoding order; only the first entry is used since decoding within a batch needs to match for symmetry
            temperature = feature_dict["temperature"] #float - sampling temperature; prob = softmax(logits/temperature)
        
        else:
            # when providing sampled actions, they may not be the same size as the featurized batch
            B_decoder = sampled_actions.shape[0]
            S_true = sampled_actions #[B,L] - sampled sequence
            mask = feature_dict["mask"][:1].repeat(B_decoder, 1) #[B_decoder,L]
            chain_mask = feature_dict["chain_mask"][:1].repeat(B_decoder, 1) #[B_decoder,L]
            bias = feature_dict["bias"][:1].repeat(B_decoder, 1, 1) #[B_decoder,L,21]
            randn = feature_dict["randn"][:1].repeat(B_decoder, 1) #[B_decoder,L]
            temperature = feature_dict["temperature"] #float - sampling temperature; prob = softmax(logits/temperature)

        B, L = S_true.shape

        # else use the provided decoding order
        if decoding_order is None:
            decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]

        E_idx = E_idx.repeat(B_decoder, 1, 1)
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=L).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(L,L, device=self.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([B, L, 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        #repeat for decoding
        S_true = S_true.repeat(B_decoder, 1)
        h_V = h_V.repeat(B_decoder, 1, 1)
        h_E = h_E.repeat(B_decoder, 1, 1, 1)
        chain_mask = chain_mask.repeat(B_decoder, 1)
        mask = mask.repeat(B_decoder, 1)
        bias = bias.repeat(B_decoder, 1, 1)

        #-----

        all_probs = torch.zeros((B_decoder, L, 20), device=self.device, dtype=torch.float32)
        all_log_probs = torch.zeros((B_decoder, L, 21), device=self.device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=self.device)
        S = 20*torch.ones((B_decoder, L), dtype=torch.int64, device=self.device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=self.device) for _ in range(len(model.decoder_layers))]

        h_EX_encoder = self.cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = self.cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder


        for t_ in range(L):

            t = decoding_order[:,t_] #[B]
            chain_mask_t = torch.gather(chain_mask, 1, t[:,None])[:,0] #[B]
            mask_t = torch.gather(mask, 1, t[:,None])[:,0] #[B]
            bias_t = torch.gather(bias, 1, t[:,None,None].repeat(1,1,21))[:,0,:] #[B,21]


            E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
            h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
            h_ES_t = self.cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))

            mask_bw_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1]))

            for l, layer in enumerate(model.decoder_layers):
                h_ESV_decoder_t = self.cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
            

                # JG: replaced mask_V with None, could be mask_t
                # This line is causing issues with backprop because was an in-place operation
                h_V_stack[l+1] = h_V_stack[l+1].scatter(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=None))


            h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]


            logits = model.W_out(h_V_t) #[B,21]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1) #[B,21]


            # JG need to add code here to pick out specific samples
            probs = torch.nn.functional.softmax((logits.detach()+bias_t) / temperature, dim=-1) #[B,21]

            probs_sample = probs[:,:20]/torch.sum(probs[:,:20], dim=-1, keepdim=True) #hard omit X #[B,20]
            
            # if you are already provided with sampled sequence just grab what you need
            if sampled_actions is None:
                S_t = torch.multinomial(probs_sample, 1)[:,0] #[B]
            else:
                S_t = sampled_actions[torch.arange(B), t]

            all_probs.scatter_(1, t[:,None,None].repeat(1,1,20), (chain_mask_t[:,None,None]*probs_sample[:,None,:]).float())
            all_log_probs.scatter_(1, t[:,None,None].repeat(1,1,21), (chain_mask_t[:,None,None]*log_probs[:,None,:]).float())

            with torch.no_grad():
                S_true_t = torch.gather(S_true, 1, t[:,None])[:,0]
                S_t = (S_t*chain_mask_t+S_true_t*(1.0-chain_mask_t)).long()
                h_S.scatter_(1, t[:,None,None].repeat(1,1,h_S.shape[-1]), model.W_s(S_t)[:,None,:])
                S.scatter_(1, t[:,None], S_t[:,None])


        output_dict = {
            "S": S, 
            "sampling_probs": all_probs, 
            "log_probs": all_log_probs, 
            "decoding_order": decoding_order, 
            "state_features": h_V_stack[-1].detach(),
            "chain_mask": chain_mask,
        }

        return output_dict

    def train_step(self, step, init_state, feature_dict):
        """
        Single training step given featurized example
        """

        to_log = {}
        self.optimizer.zero_grad()

        h_V_in, h_E_in, E_idx_in = init_state

        # turn on grads for state features
        h_V_in.requires_grad = True
        h_E_in.requires_grad = True

        # run the policy
        out = self.rollout(feature_dict, h_V_in, h_E_in, E_idx_in)

        # mask for what was actually decoded in the sequence
        seq_mask = torch.nn.functional.one_hot(out["S"], num_classes=len(alphabet)).float()

        # apply mask and take sum over each seq in the batch
        batched_log_probs = (out["log_probs"] * seq_mask).sum(dim=(-1))

        batched_reward, metrics = self.reward_fn(step, out, feature_dict, self.device, evaluate=False)
        to_log.update(metrics)

        # get baseline first
        # baseline = torch.stack(self.reward_history).mean()
        baseline = torch.tensor(self.reward_history, dtype=torch.float32, device=self.device).mean()
        self.reward_history.append(batched_reward.mean().item())

        # baseline subtracted reward
        baseline_subtracted_reward = batched_reward - baseline

        # batched_log_probs [B, L]
        if baseline_subtracted_reward.shape != batched_log_probs.shape:
            # if reward is not per residue, we need to sum over log probs
            loss = -(batched_log_probs.sum(dim=-1) * baseline_subtracted_reward).mean()
        else:
            # else reward is per residue, so apply per residue reward and then sum
            loss = -(batched_log_probs * baseline_subtracted_reward).sum(dim=-1).mean()

        # optimizer update
        loss.backward()
        self.optimizer.step()

        to_log["loss"] = loss.detach().cpu().item()
        to_log["reward"] = batched_reward.mean().cpu().item()
        
        # Add wandb logging for PolicyMPNN specific metrics
        if wandb.run:
            wandb_log = {
                "loss": to_log["loss"],
                "reward": to_log["reward"],
                "baseline": baseline.cpu().item(),
                "baseline_subtracted_reward": baseline_subtracted_reward.mean().cpu().item(),
                "policy/log_probs_mean": batched_log_probs.mean().cpu().item(),
                "policy/log_probs_std": batched_log_probs.std().cpu().item(),
            }
            
            # Add reward function metrics if available
            if metrics:
                wandb_log.update({f"reward_metrics/{k}": v for k, v in metrics.items()})
            
            # Add reward statistics
            wandb_log.update({
                "rewards/mean": batched_reward.mean().cpu().item(),
                "rewards/std": batched_reward.std().cpu().item() if batched_reward.numel() > 1 else 0.0,
                "rewards/min": batched_reward.min().cpu().item(),
                "rewards/max": batched_reward.max().cpu().item(),
            })
            
            wandb.log(wandb_log, step=step)
        
        return to_log

    def train(self):
        """
        Run the main training loop
        """
        self.model.train()

        # featurize from input pdb (in future maybe policy is trained with a variety of pdbs)
        feature_dict = self.featurize_pdb(self.cfg.pdb)

        # encode initial state (run mpnn encoder)
        h_V, h_E, E_idx = self.encode_initial_state(feature_dict)

        # train loop
        start_time = time.time()
        for step in tqdm(range(self.cfg.N_steps), desc="Training"):
            
            # clone initial state variables
            init_state = (h_V.clone(), h_E.clone(), E_idx.clone())
            
            # train step
            to_log = self.train_step(step, init_state, feature_dict)

            # metric logging
            runtime = time.time() - start_time
            
            if wandb.run:
                # Log runtime and step separately to avoid conflicts with train_step logging
                wandb.log({"runtime": runtime, "training/step": step}, step=step)
            else:
                self.log_metrics(step, runtime, to_log)

            # model checkpointing
            if step > 0 and  step % self.checkpoint_every_n_steps == 0:
                self.checkpoint_model(step, to_log)
        
        print("Training complete.")
        print(f"Best reward seen: {self.best_seen_reward:.4f} at step {self.step_at_best_seen_reward}")

    def log_metrics(self, step, runtime, to_log):
        """
        Log training metrics
        """
        metrics_to_log = [k for k,v in to_log.items() if isinstance(v, float)]
        log_path = os.path.join(self.output_dir, f"{self.run_name}_train_metrics.csv")
        if not os.path.exists(log_path):
            with open(log_path,'w') as f:
                f.write("step,runtime,"\
                        +",".join([f"{m}" for m in metrics_to_log])\
                        +'\n')
        with open(log_path,'a') as f:
            f.write(f"{step},{runtime:.4f},"\
                    +",".join([f"{to_log[m]:.4f}" for m in metrics_to_log])\
                    +'\n')
    
    def checkpoint_model(self, step, to_log):
        """
        Checkpoint model locally only
        """
        curr_reward = to_log["reward"]
        ckpt = {
                "config": dict(self.cfg),
                "step": step,
                "reward": curr_reward,
                "model_state_dict": self.model.state_dict(),
            }

        ckpt_path = os.path.join(self.output_dir, f"{self.run_name}_last.pt")
        torch.save(ckpt, ckpt_path)

        if curr_reward > self.best_seen_reward:
            self.best_seen_reward = curr_reward
            self.step_at_best_seen_reward = step

            best_ckpt_path = os.path.join(self.output_dir, f"{self.run_name}_best.pt")
            torch.save(ckpt, best_ckpt_path)
            
            # Log best reward metrics to wandb without saving the file
            if wandb.run:
                wandb.log({"best/reward": curr_reward, "best/step": step})


    def evaluate(self):
        """
        Run evaluation on the trained policy to generate sequences and compute rewards.
        """
        # Set model to evaluation mode
        self.model.eval()
        print("Starting evaluation...")
        
        # Featurize PDB once (same as in training)
        feature_dict = self.featurize_pdb(self.cfg.pdb)
        
        # Encode initial state once (same as in training)
        h_V, h_E, E_idx = self.encode_initial_state(feature_dict)
        
        # Collect rollouts for all batches first
        outs = []
        for _ in tqdm(range(self.cfg.evaluate.num_batches), desc="Evaluation batches"):
            # Set batch size for this evaluation batch
            feature_dict['batch_size'] = self.cfg.evaluate.batch_size
            
            # Clone initial state variables for this batch
            init_state = (h_V.clone(), h_E.clone(), E_idx.clone())
            
            with torch.no_grad():
                outs.append(self.rollout(feature_dict, *init_state))
        
        # Concatenate outputs along the batch dimension
        def _cat(key):
            return torch.cat([o[key] for o in outs], dim=0)
        
        combined_out = {
            "S": _cat("S"),
            "sampling_probs": _cat("sampling_probs"),
            "log_probs": _cat("log_probs"),
            "decoding_order": _cat("decoding_order"),
            "state_features": _cat("state_features"),
            "chain_mask": _cat("chain_mask"),
        }
        
        # Single reward call over all sequences
        with torch.no_grad():
            rewards, metrics = self.reward_fn(0, combined_out, feature_dict, self.device, evaluate=True)
            metrics['batch_rewards'] = rewards
        
        print(f"Evaluation complete. Processed {len(outs)} batches.")
        
        # Check if fragment-based reward was used
        has_fragment_rewards = hasattr(self.cfg.reward, 'frag_cfg') and self.cfg.reward.frag_cfg is not None
        
        # Process and save results
        print("Processing evaluation results...")
        self._process_evaluation_results(combined_out, metrics, has_fragment_rewards)

    def _process_evaluation_results(self, output, metrics, has_fragment_rewards: bool):
        """Aggregate evaluation outputs into CSVs under the latest evaluation run directory.
        Creates: all_sequences.csv and per-fragment CSVs when fragment rewards are enabled."""
        import re
        
        # Find latest evaluation run index from existing batch run dirs
        pipeline_output_dir = Path(self.output_dir) / "pipeline_output"
        pipeline_output_dir.mkdir(parents=True, exist_ok=True)
        eval_batch_dirs = [
            d for d in pipeline_output_dir.glob("pipeline_output_eval_*_batch_*") if d.is_dir()
        ]
        eval_idx = "0"
        if eval_batch_dirs:
            latest_dir = max(eval_batch_dirs, key=lambda p: p.stat().st_mtime)
            m = re.match(r"pipeline_output_eval_(\d+)_batch_\d{4}$", latest_dir.name)
            if m:
                eval_idx = m.group(1)
        CSV_DIR = pipeline_output_dir / f"pipeline_output_eval_{eval_idx}"
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        
        # Helpers
        def _make_pos_mask(L: int, B: int, chain_mask: torch.Tensor, fragment_bounds=None, device=None):
            if device is None:
                device = chain_mask.device
            if chain_mask.dim() > 1:
                cm = chain_mask[0].bool()
            else:
                cm = chain_mask.bool()
            base = cm.to(device).unsqueeze(0).expand(B, -1)
            if fragment_bounds is None:
                frag = torch.ones((1, L), dtype=torch.bool, device=device)
            else:
                s, e = int(fragment_bounds[0]), int(fragment_bounds[1])
                frag = torch.zeros((1, L), dtype=torch.bool, device=device)
                if e > s:
                    frag[:, s:e] = True
            return base & frag.expand(B, -1)
        
        def _calculate_sequence_log_probs(sequences, sampling_probs, pos_mask):
            logp = torch.log(sampling_probs.clamp_min(1e-12))
            oh = F.one_hot(sequences.long(), 21)[:, :, :20].to(logp.dtype)
            seq_logp = (logp * oh).sum(-1)
            return (seq_logp * pos_mask.float()).sum(1)
        
        # Aggregate
        rows_main = []
        frag_rows = {}
        
        S = output["S"]
        P = output["sampling_probs"]
        chain_mask = output.get("chain_mask")
        if chain_mask is None:
            return
        B, L = S.shape
        
        pos_mask_main = _make_pos_mask(L, B, chain_mask, fragment_bounds=None, device=S.device)
        cum_log = _calculate_sequence_log_probs(S, P, pos_mask_main)
        
        batch_rewards = metrics.get("batch_rewards")
        if batch_rewards is None:
            return
        pmr = pos_mask_main.to(batch_rewards.device).float()
        denom = pmr.sum(1).clamp_min(1e-12)
        if batch_rewards.ndim == 1:
            rewards = batch_rewards
        else:
            rewards = (batch_rewards * pmr).sum(1) / denom
        
        seq_idx_strs = [" ".join(map(str, row)) for row in S.tolist()]
        seq_strs = ["".join([alphabet[int(s)] for s in seq]) for seq in S.tolist()]
        
        for i in range(B):
            rows_main.append({
                "sequence_idx": seq_idx_strs[i],
                "sequence_str": seq_strs[i],
                "cumulative_log_prob": float(cum_log[i].detach().cpu()),
                "reward_mean": float((rewards[i].detach().cpu() if torch.is_tensor(rewards) else rewards[i])),
            })
        
        # Per-fragment CSVs
        if has_fragment_rewards and ("fragment_bounds" in metrics) and ("fragment_dict" in metrics):
            fragment_bounds = metrics["fragment_bounds"]
            fragment_dict = metrics["fragment_dict"]
            for i_fb, (start, end) in enumerate(fragment_bounds):
                key = f"fragment_{i_fb+1}"
                frag_mask = _make_pos_mask(L, B, chain_mask, fragment_bounds=[start, end], device=S.device)
                cum_log_f = _calculate_sequence_log_probs(S, P, frag_mask)
                
                fmr = frag_mask.to(batch_rewards.device).float()
                denom_f = fmr.sum(1).clamp_min(1e-12)
                if batch_rewards.ndim == 1:
                    rewards_f = batch_rewards
                else:
                    rewards_f = (batch_rewards * fmr).sum(1) / denom_f
                
                if key not in frag_rows:
                    frag_rows[key] = []
                for frag_name, frag_seq in fragment_dict.get(key, []):
                    try:
                        j = int(str(frag_name).split(".")[0])
                    except Exception:
                        continue
                    frag_rows[key].append({
                        "fragment": key,
                        "fragment_name": frag_name,
                        "fragment_seq": frag_seq,
                        "cumulative_log_prob_fragment": float(cum_log_f[j].detach().cpu()),
                        "reward_mean_fragment": float((rewards_f[j].detach().cpu() if torch.is_tensor(rewards_f) else rewards_f[j])),
                    })
        
        # Write CSVs
        df_seqs = pd.DataFrame(rows_main)
        df_seqs.to_csv(CSV_DIR / "all_sequences.csv", index=False)
        if has_fragment_rewards and frag_rows:
            all_frag_rows = []
            for key, rows in frag_rows.items():
                all_frag_rows.extend(rows)
            pd.DataFrame(all_frag_rows).to_csv(CSV_DIR / "fragments.csv", index=False)
        
    