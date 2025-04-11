import sys, os
import json
import torch
import numpy as np
from tqdm import tqdm
import time
sys.path.append("/software/lab/mpnn/fused_mpnn")

from data_utils import featurize, parse_PDB
from model_utils import ProteinMPNN
import hydra

PROTEIN_MPNN_CKPT_PATH = "/databases/mpnn/vanilla_model_weights/v_48_020.pt"
LIGAND_MPNN_CKPT_PATH = "/databases/mpnn/ligand_mpnn_model_weights/s25_r010_t300_p.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define mpnn constants
restype_1to3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', 'X': 'UNK'}
restype_STRtoINT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}
restype_INTtoSTR = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X'}
alphabet = list(restype_STRtoINT)


class PolicyMPNN:
    def __init__(self, cfg):

        self.cfg = cfg
        self.device = DEVICE
        self.run_name = cfg.run_name
        self.output_dir = cfg.output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)

        # log reward history
        self.reward_history = [torch.tensor(0., dtype=torch.float32, device=self.device)]

        # load model
        self.model = self.load_mpnn_model()
        self.model.to(self.device)

        if self.cfg.eval:
            self.model.eval()

        # load optimizer
        self.optimizer = self.get_optimizer()

        # defaults from mpnn
        self.ligand_mpnn_use_atom_context = 1
        self.ligand_mpnn_cutoff_for_score = 8.0

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
        if self.cfg.checkpoint_path:
            ckpt_path = self.cfg.checkpoint_path

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
        omit_AA_list = self.cfg.omit_AA
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


    def rollout(self, feature_dict, h_V, h_E, E_idx):
        """
        Ripped from fused MPNN decoding, modified to allow grads to flow through this pass
        """

        # decode
        B_decoder = feature_dict["batch_size"]
        S_true = feature_dict["S"] #[B,L] - integer proitein sequence encoded using "restype_STRtoINT
        #R_idx = feature_dict["R_idx"] #[B,L] - primary sequence residue index
        mask = feature_dict["mask"] #[B,L] - mask for missing regions - should be removed! all ones most of the time
        chain_mask = feature_dict["chain_mask"] #[B,L] - mask for which residues need to be fixed; 0.0 - fixed; 1.0 - will be designed
        bias = feature_dict["bias"] #[B,L,21] - amino acid bias per position

        #chain_labels = feature_dict["chain_labels"] #[B,L] - integer labels for chain letters
        randn = feature_dict["randn"] #[B,L] - random numbers for decoding order; only the first entry is used since decoding within a batch needs to match for symmetry
        temperature = feature_dict["temperature"] #float - sampling temperature; prob = softmax(logits/temperature)
        symmetry_list_of_lists = feature_dict["symmetry_residues"] #[[0, 1, 14], [10,11,14,15], [20, 21]] #indices to select X over length - L
        symmetry_weights_list_of_lists = feature_dict["symmetry_weights"] #[[1.0, 1.0, 1.0], [-2.0,1.1,0.2,1.1], [2.3, 1.1]]
        B, L = S_true.shape

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
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=self.device) for _ in range(len(self.model.decoder_layers))]

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

            for l, layer in enumerate(self.model.decoder_layers):
                h_ESV_decoder_t = self.cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
            

                # JG: replaced mask_V with None, could be mask_t
                # This line is causing issues with backprop because was an in-place operation
                h_V_stack[l+1] = h_V_stack[l+1].scatter(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=None))


            h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
            logits = self.model.W_out(h_V_t) #[B,21]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1) #[B,21]


            # JG need to add code here to pick out specific samples
            probs = torch.nn.functional.softmax((logits.detach()+bias_t) / temperature, dim=-1) #[B,21]

            probs_sample = probs[:,:20]/torch.sum(probs[:,:20], dim=-1, keepdim=True) #hard omit X #[B,20]
            S_t = torch.multinomial(probs_sample, 1)[:,0] #[B]

            all_probs.scatter_(1, t[:,None,None].repeat(1,1,20), (chain_mask_t[:,None,None]*probs_sample[:,None,:]).float())
            all_log_probs.scatter_(1, t[:,None,None].repeat(1,1,21), (chain_mask_t[:,None,None]*log_probs[:,None,:]).float())

            with torch.no_grad():
                S_true_t = torch.gather(S_true, 1, t[:,None])[:,0]
                S_t = (S_t*chain_mask_t+S_true_t*(1.0-chain_mask_t)).long()
                h_S.scatter_(1, t[:,None,None].repeat(1,1,h_S.shape[-1]), self.model.W_s(S_t)[:,None,:])
                S.scatter_(1, t[:,None], S_t[:,None])


        output_dict = {"S": S, "sampling_probs": all_probs, "log_probs": all_log_probs, "decoding_order": decoding_order}

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
        batched_log_probs = (out["log_probs"] * seq_mask).sum(dim=(-1,-2))

        # batched_reward = (out["S"] == aa_index_of_interest).sum(dim=-1).float()
        batched_reward, metrics = self.reward_fn(step, out, feature_dict, self.device)
        to_log.update(metrics)

        # get baseline first
        baseline = torch.stack(self.reward_history).mean()
        self.reward_history.append(batched_reward.mean())

        # baseline subtracted reward
        baseline_subtracted_reward = batched_reward - baseline

        # compute loss
        loss = -(batched_log_probs * baseline_subtracted_reward).mean()

        # optimizer update
        loss.backward()
        self.optimizer.step()

        to_log["loss"] = loss.detach().cpu().item()
        to_log["reward"] = batched_reward.mean().cpu().item()
        
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
            self.log_metrics(step, runtime, to_log)

            # model checkpointing
            if step > 0 and self.checkpoint_every_n_steps % step == 0:
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
        Checkpoint model
            - save last model
            - save best model
        """
        
        curr_reward = to_log["reward"]
        ckpt = {
                "config":dict(self.cfg),
                "step":step,
                "reward":curr_reward,
                "model_state_dict":self.model.state_dict(),
            }

        ckpt_path = os.path.join(self.output_dir, f"{self.run_name}_last.pt")
        torch.save(ckpt, ckpt_path)

        if curr_reward > self.best_seen_reward:
            self.best_seen_reward = curr_reward
            self.step_at_best_seen_reward = step

            best_ckpt_path = os.path.join(self.output_dir, f"{self.run_name}_best.pt")
            torch.save(ckpt, best_ckpt_path)