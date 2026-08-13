"""
Vanilla REINFORCE (policy gradient) fine-tuning of ProteinMPNN.

Provides :class:`PolicyMPNN`, which wraps a ProteinMPNN model with an
autoregressive rollout that keeps gradients through the decoder so that
the REINFORCE loss can flow back. A running-mean baseline is subtracted
from the reward to reduce variance.

Amino-acid encoding constants used across the codebase (``alphabet``,
``restype_STRtoINT``, etc.) are defined here and should be imported from
this module rather than duplicated elsewhere.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import time
import hydra
from omegaconf import OmegaConf

from cleo.design.protein_mpnn_utils.data_utils import featurize, parse_PDB
from cleo.design.protein_mpnn_utils.model_utils import ProteinMPNN


PROTEIN_MPNN_UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "protein_mpnn_utils")
PROTEIN_MPNN_CKPT_PATH = os.path.join(PROTEIN_MPNN_UTILS_DIR, "vanilla_protein_mpnn_weights.pt")
LIGAND_MPNN_CKPT_PATH = os.path.join(PROTEIN_MPNN_UTILS_DIR, "ligand_protein_mpnn_weights.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ProteinMPNN amino-acid encoding tables
restype_1to3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', 'X': 'UNK'}
restype_STRtoINT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}
restype_INTtoSTR = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X'}
alphabet = list(restype_STRtoINT)


class PolicyMPNN:
    """ProteinMPNN wrapper for RL fine-tuning with vanilla REINFORCE."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = DEVICE
        self.run_name = cfg.run_name
        self.output_dir = os.path.join(cfg.output_dir, cfg.run_name)

        os.makedirs(self.output_dir, exist_ok=True)
        OmegaConf.save(config=cfg, f=os.path.join(self.output_dir, f"{self.run_name}_config.yaml"))

        self.reward_history = [0]

        self.model = self.load_mpnn_model()
        self.model.to(self.device)
        self.optimizer = self.get_optimizer()

        self.ligand_mpnn_use_atom_context = 1
        self.ligand_mpnn_cutoff_for_score = 8.0

        self.reward_fn = hydra.utils.instantiate(cfg.reward)

        self.checkpoint_every_n_steps = self.cfg.checkpoint_every_n_steps
        # -inf, not 0: with `mode: min` or a z-scored reward the tracked quantity
        # is routinely negative, and a floor of 0 would reject every checkpoint.
        self.best_seen_reward = float("-inf")
        self.step_at_best_seen_reward = 0
        # Restores the pre-fix scoring distribution, for A/B only. See rollout.
        self.legacy_logprobs = bool(self.cfg.get("legacy_logprobs", False))
        self.checkpoint_metric = self.cfg.get("checkpoint_metric", "reward")
        self.checkpoint_metric_mode = self.cfg.get("checkpoint_metric_mode", "max")
        self._ckpt_metric_history = []


    def load_mpnn_model(self):
        """Load and return a ProteinMPNN model initialised from checkpoint weights."""

        model_type = self.cfg.model_type

        # Side-chain context: the fixed residues' side chains are shown to the
        # model as context (designed residues' side chains are masked out in
        # ProteinFeaturesLigand). For motif scaffolding this is what lets the
        # policy design a pocket around the *rotamer* rather than only around
        # the motif backbone. The shipped LigandMPNN weights contain all the
        # required parameters, so it costs nothing but is off by default to
        # preserve existing runs' behaviour. ProteinMPNN has no ligand feature
        # module and so cannot use it.
        use_sc = int(self.cfg.get("ligand_mpnn_use_side_chain_context", 0) or 0)

        if model_type == "protein_mpnn":
            self.atom_context_num = 1
            k_neighbors = 48
            if use_sc:
                raise ValueError(
                    "ligand_mpnn_use_side_chain_context requires model_type='ligand_mpnn'"
                )
            self.ligand_mpnn_use_side_chain_context = 0
            ckpt_path = PROTEIN_MPNN_CKPT_PATH
        elif model_type == "ligand_mpnn":
            self.atom_context_num = 25
            k_neighbors = 32
            self.ligand_mpnn_use_side_chain_context = use_sc
            ckpt_path = LIGAND_MPNN_CKPT_PATH
        else:
            raise ValueError("Invalid model type specified. Choose 'ligand_mpnn' or 'protein_mpnn'.")

        if self.cfg.get('checkpoint_path'):
            print(f"Loading training checkpoint from {self.cfg.checkpoint_path}")
            ckpt_path = self.cfg.checkpoint_path
        else:
            print(f"Using default MPNN checkpoint: {ckpt_path}")

        model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k_neighbors,
            device=self.device,
            atom_context_num=self.atom_context_num,
            model_type=model_type,
            ligand_mpnn_use_side_chain_context=self.ligand_mpnn_use_side_chain_context,
        )

        is_default_checkpoint = ckpt_path in [PROTEIN_MPNN_CKPT_PATH, LIGAND_MPNN_CKPT_PATH]
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=is_default_checkpoint)
        incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        # strict=False is kept so checkpoints with extra keys still load, but a
        # *missing* key means that submodule silently keeps its random
        # initialisation -- which looks like a merely mediocre model rather than
        # a broken one. Never let that pass quietly.
        if incompatible.missing_keys:
            raise RuntimeError(
                f"{ckpt_path} is missing {len(incompatible.missing_keys)} parameter(s) "
                f"required by this model configuration, which would leave them randomly "
                f"initialised: {incompatible.missing_keys[:10]}"
                + (" ..." if len(incompatible.missing_keys) > 10 else "")
            )
        return model.to(self.device)

    def get_optimizer(self):
        """Freeze encoder parameters and return an Adam optimizer over the decoder."""

        for name, param in self.model.named_parameters():
            if "decoder_layers" not in name and "W_out" not in name:
                param.requires_grad = False

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.lr,
        )
        return optimizer

    def featurize_pdb(self, pdb):
        """Parse a PDB file and build the feature dictionary consumed by :meth:`rollout`."""

        protein_dict, backbone, other_atoms, icodes, water_atoms = parse_PDB(
            pdb,
            device=self.device,
            atom_context_num=self.atom_context_num,
            chains="",
            parse_all_atoms=self.ligand_mpnn_use_side_chain_context,
        )

        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
        chain_letters_list = list(protein_dict["chain_letters"])
        encoded_residues = [
            str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
            for i in range(len(R_idx_list))
        ]

        chain_mask = torch.tensor(
            np.array([True for _ in protein_dict["chain_letters"]], dtype=np.int32),
            device=self.device,
        )
        protein_dict["chain_mask"] = chain_mask

        fixed_residues = [item for item in self.cfg.fixed_residues.split()]
        fixed_positions = torch.tensor(
            [int(item not in fixed_residues) for item in encoded_residues],
            device=self.device,
        )
        if fixed_residues:
            protein_dict["chain_mask"] = protein_dict["chain_mask"] * fixed_positions

        protein_dict["side_chain_mask"] = protein_dict["chain_mask"]

        omit_AA_list = self.cfg.omit_AA if self.cfg.omit_AA is not None else []
        omit_AA = torch.tensor(
            np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32),
            device=self.device,
        )

        bias_AA_per_residue = torch.zeros([len(encoded_residues), 21], device=self.device, dtype=torch.float32)
        omit_AA_per_residue = torch.zeros([len(encoded_residues), 21], device=self.device, dtype=torch.float32)

        feature_dict = featurize(
            protein_dict,
            cutoff_for_score=self.ligand_mpnn_cutoff_for_score,
            use_atom_context=self.ligand_mpnn_use_atom_context,
            number_of_ligand_atoms=self.atom_context_num,
            model_type=self.cfg.model_type,
        )
        feature_dict["batch_size"] = self.cfg.batch_size
        B, L, _, _ = feature_dict["X"].shape

        feature_dict["temperature"] = self.cfg.temperature
        bias_AA = torch.zeros([21], device=self.device, dtype=torch.float32)
        feature_dict["bias"] = (
            (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
            + bias_AA_per_residue[None]
            - 1e8 * omit_AA_per_residue[None]
        )

        feature_dict["symmetry_residues"] = [[]]
        feature_dict["symmetry_weights"] = [[]]
        feature_dict["randn"] = torch.randn(
            [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
            device=self.device,
        )

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
        Copy from proteinMPNN Utils
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
        Copy from proteinMPNN Utils
        """
        h_nodes = self.gather_nodes(h_nodes, E_idx)
        h_nn = torch.cat([h_neighbors, h_nodes], -1)
        return h_nn

    def rollout(
        self,
        feature_dict,
        h_V,
        h_E,
        E_idx,
        decoding_order=None,
        sampled_actions=None,
        model=None,
    ):
        """Autoregressive decoding with gradients through the decoder.

        Adapted from the fused ProteinMPNN decoding loop. Gradients flow
        through the decoder layers so the RL loss can update decoder weights.
        ``scatter`` is used instead of in-place indexing to keep the
        computation graph intact.

        Args:
            feature_dict: Feature dictionary from :meth:`featurize_pdb`.
            h_V: Node features from encoder [1, L, C].
            h_E: Edge features from encoder [1, L, K, C].
            E_idx: Edge indices [1, L, K].
            decoding_order: Optional pre-defined decoding order [B, L].
            sampled_actions: Optional pre-sampled sequence integers [B, L].
                When provided, log-probs are computed for these actions
                instead of sampling new ones (used by GRPO for importance
                ratio computation).
            model: Model to decode with (defaults to ``self.model``).

        Returns:
            dict with keys ``S``, ``sampling_probs``, ``log_probs``,
            ``decoding_order``, ``state_features``, ``chain_mask``.
        """

        if model is None:
            model = self.model

        if sampled_actions is None:
            B_decoder = feature_dict["batch_size"]
            S_true = feature_dict["S"]
            mask = feature_dict["mask"]
            chain_mask = feature_dict["chain_mask"]
            bias = feature_dict["bias"]
            randn = feature_dict["randn"]
            temperature = feature_dict["temperature"]
        else:
            B_decoder = sampled_actions.shape[0]
            S_true = sampled_actions
            mask = feature_dict["mask"][:1].repeat(B_decoder, 1)
            chain_mask = feature_dict["chain_mask"][:1].repeat(B_decoder, 1)
            bias = feature_dict["bias"][:1].repeat(B_decoder, 1, 1)
            randn = feature_dict["randn"][:1].repeat(B_decoder, 1)
            temperature = feature_dict["temperature"]

        B, L = S_true.shape

        if decoding_order is None:
            decoding_order = torch.argsort((chain_mask + 0.0001) * (torch.abs(randn)))

        E_idx = E_idx.repeat(B_decoder, 1, 1)
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=L).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(L, L, device=self.device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([B, L, 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        S_true = S_true.repeat(B_decoder, 1)
        h_V = h_V.repeat(B_decoder, 1, 1)
        h_E = h_E.repeat(B_decoder, 1, 1, 1)
        chain_mask = chain_mask.repeat(B_decoder, 1)
        mask = mask.repeat(B_decoder, 1)
        bias = bias.repeat(B_decoder, 1, 1)

        all_probs = torch.zeros((B_decoder, L, 20), device=self.device, dtype=torch.float32)
        all_log_probs = torch.zeros((B_decoder, L, 21), device=self.device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=self.device)
        S = 20 * torch.ones((B_decoder, L), dtype=torch.int64, device=self.device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=self.device) for _ in range(len(model.decoder_layers))]

        omit_num = torch.zeros(B_decoder, device=self.device)
        omit_den = torch.zeros(B_decoder, device=self.device)

        h_EX_encoder = self.cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = self.cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for t_ in range(L):
            t = decoding_order[:, t_]
            chain_mask_t = torch.gather(chain_mask, 1, t[:, None])[:, 0]
            mask_t = torch.gather(mask, 1, t[:, None])[:, 0]
            bias_t = torch.gather(bias, 1, t[:, None, None].repeat(1, 1, 21))[:, 0, :]

            E_idx_t = torch.gather(E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
            h_E_t = torch.gather(h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]))
            h_ES_t = self.cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
            mask_bw_t = torch.gather(mask_bw, 1, t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2], mask_bw.shape[-1]))

            for l, layer in enumerate(model.decoder_layers):
                h_ESV_decoder_t = self.cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                h_V_t = torch.gather(h_V_stack[l], 1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]))
                h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t

                # Uses scatter (not in-place indexing) so gradients flow through the decoder
                h_V_stack[l+1] = h_V_stack[l+1].scatter(
                    1, t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                    layer(h_V_t, h_ESV_t, mask_V=None),
                )

            h_V_t = torch.gather(h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]))[:, 0]

            logits = model.W_out(h_V_t)

            probs = torch.nn.functional.softmax((logits.detach() + bias_t) / temperature, dim=-1)
            probs_sample = probs[:, :20] / torch.sum(probs[:, :20], dim=-1, keepdim=True)

            if self.legacy_logprobs:
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            else:
                # Score the distribution actions were actually drawn from.
                #
                # Sampling uses softmax((logits + bias)/T) renormalised over the
                # 20 residue tokens; scoring under log_softmax(logits) over all
                # 21 is a different distribution. The two differ by
                # log P(allowed), which depends on theta and so does NOT cancel
                # between pi_new and pi_old -- the true importance ratio is the
                # implemented one times P_old(allowed)/P_new(allowed). It also
                # made `temperature` silently wrong for anything but T = 1.
                #
                # bias is finite (-1e8 at omitted residues), so the log_softmax
                # stays finite and the masked-out channel 20 can be padded with
                # zero rather than -inf, which would poison the 0 * x product at
                # fixed positions.
                lp20 = torch.nn.functional.log_softmax(
                    ((logits + bias_t) / temperature)[:, :20], dim=-1
                )
                log_probs = torch.nn.functional.pad(lp20, (0, 1), value=0.0)

            # Drift monitor: how much probability mass the *model* puts on tokens
            # sampling is forbidden from choosing. Nothing in the objective
            # discourages this mass from growing -- omitted tokens are simply
            # never drawn -- so without this it grows unobserved, and it is
            # exactly the quantity that makes the ratio correction above bite.
            with torch.no_grad():
                p_omit_t = (torch.softmax(logits, dim=-1)
                            * (bias_t <= -1e7).to(logits.dtype)).sum(dim=-1)
                omit_num = omit_num + p_omit_t * chain_mask_t
                omit_den = omit_den + chain_mask_t

            if sampled_actions is None:
                S_t = torch.multinomial(probs_sample, 1)[:, 0]
            else:
                S_t = sampled_actions[torch.arange(B), t]

            all_probs.scatter_(1, t[:, None, None].repeat(1, 1, 20), (chain_mask_t[:, None, None] * probs_sample[:, None, :]).float())
            all_log_probs.scatter_(1, t[:, None, None].repeat(1, 1, 21), (chain_mask_t[:, None, None] * log_probs[:, None, :]).float())

            with torch.no_grad():
                S_true_t = torch.gather(S_true, 1, t[:, None])[:, 0]
                S_t = (S_t * chain_mask_t + S_true_t * (1.0 - chain_mask_t)).long()
                h_S.scatter_(1, t[:, None, None].repeat(1, 1, h_S.shape[-1]), model.W_s(S_t)[:, None, :])
                S.scatter_(1, t[:, None], S_t[:, None])

        output_dict = {
            "S": S,
            "sampling_probs": all_probs,
            "log_probs": all_log_probs,
            "decoding_order": decoding_order,
            "state_features": h_V_stack[-1].detach(),
            # Sliced to B_decoder: in the teacher-forced branch `chain_mask`,
            # `mask`, `bias` and `S_true` are each repeated twice -- once when
            # unpacked from feature_dict and again below -- so they carry
            # B_decoder**2 rows. Every use inside the decode loop is a
            # torch.gather with a (B_decoder, 1) index, which legally reads just
            # the leading rows, and since all rows are identical copies the
            # decode has always been correct. Only the tensor handed back was
            # the wrong shape, which went unnoticed until something consumed it.
            "chain_mask": chain_mask[:B_decoder],
            "p_omit": omit_num / omit_den.clamp(min=1.0),
        }

        return output_dict
    
    def get_sequences(self, policy_output, chain_mask=None):
        """Decode integer sequences from a rollout output into amino-acid strings."""
        sampled_sequences = policy_output["S"]
        if chain_mask is not None:
            sampled_sequences = sampled_sequences[:, chain_mask]

        sequences = []
        for i in range(sampled_sequences.shape[0]):
            seq_str = "".join([alphabet[int(s)] for s in sampled_sequences[i]])
            sequences.append(seq_str)
        return sequences

    def train_step(self, step, init_state, feature_dict):
        """Run one REINFORCE update: rollout -> reward -> policy gradient."""

        to_log = {}
        self.optimizer.zero_grad()

        h_V_in, h_E_in, E_idx_in = init_state
        h_V_in.requires_grad = True
        h_E_in.requires_grad = True

        out = self.rollout(feature_dict, h_V_in, h_E_in, E_idx_in)

        seq_mask = torch.nn.functional.one_hot(out["S"], num_classes=len(alphabet)).float()
        batched_log_probs = (out["log_probs"] * seq_mask).sum(dim=(-1))

        batched_reward, metrics = self.reward_fn(step, out, feature_dict, self.device)
        to_log.update(metrics)

        baseline = torch.tensor(self.reward_history, dtype=torch.float32, device=self.device).mean()
        self.reward_history.append(batched_reward.mean().item())
        baseline_subtracted_reward = batched_reward - baseline

        if baseline_subtracted_reward.shape != batched_log_probs.shape:
            loss = -(batched_log_probs.sum(dim=-1) * baseline_subtracted_reward).mean()
        else:
            loss = -(batched_log_probs * baseline_subtracted_reward).sum(dim=-1).mean()

        loss.backward()
        self.optimizer.step()

        to_log["loss"] = loss.detach().cpu().item()
        to_log["reward"] = batched_reward.mean().cpu().item()

        return to_log

    def train(self):
        """Run the main training loop for ``N_steps`` iterations."""

        start_time = time.time()
        for step in tqdm(range(self.cfg.N_steps), desc="Training"):
            self.model.train()
            feature_dict = self.featurize_pdb(self.cfg.pdb)
            h_V, h_E, E_idx = self.encode_initial_state(feature_dict)
            init_state = (h_V.clone(), h_E.clone(), E_idx.clone())

            to_log = self.train_step(step, init_state, feature_dict)

            runtime = time.time() - start_time
            self.log_metrics(step, runtime, to_log)

            if step > 0 and step % self.checkpoint_every_n_steps == 0:
                self.checkpoint_model(step, to_log)

        # The final policy is the one the run actually produced, and it is not
        # generally at a multiple of checkpoint_every_n_steps. Writing _last only
        # inside the interval branch above silently discarded every step after
        # the last multiple -- a 75-step run left _last at step 50, throwing away
        # the 24 steps that a still-improving run cares most about.
        self.checkpoint_model(step, to_log, final=True)

        print("Training complete.")
        print(f"Best {self.checkpoint_metric} seen: {self.best_seen_reward:.4f} "
              f"at step {self.step_at_best_seen_reward}")
        if len(set(self._ckpt_metric_history)) <= 1 and self._ckpt_metric_history:
            print(
                f"WARNING: '{self.checkpoint_metric}' never varied across "
                f"{len(self._ckpt_metric_history)} checkpoints, so _best.pt is "
                "simply the first one saved and carries no selection signal. "
                "This is what a single rank-normalised metric looks like: rank "
                "maps each batch onto the same fixed distribution, so the mean "
                "is constant by construction. Set `checkpoint_metric` to a raw "
                "quantity (e.g. ame_motif_rmsd_batch_mean) to select on."
            )

    def log_metrics(self, step, runtime, to_log):
        """Append one row to the CSV training log."""
        metrics_to_log = [k for k, v in to_log.items() if isinstance(v, float)]
        log_path = os.path.join(self.output_dir, f"{self.run_name}_train_metrics.csv")
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                f.write("step,runtime," + ",".join(metrics_to_log) + '\n')
        with open(log_path, 'a') as f:
            f.write(f"{step},{runtime:.4f}," + ",".join([f"{to_log[m]:.4f}" for m in metrics_to_log]) + '\n')
    
    def checkpoint_model(self, step, to_log, final=False):
        """Save ``_last``, ``_step_NNNN``, and (if improved) ``_best`` checkpoints.

        ``final`` marks the end-of-training call, which refreshes ``_last`` and
        may update ``_best`` but does not write a numbered checkpoint (the step
        is whatever the run ended on, not a checkpoint interval).

        Selection for ``_best`` uses ``cfg.checkpoint_metric``, defaulting to the
        aggregate reward. The default is only meaningful when the reward varies
        between steps, which a rank-normalised single-metric reward does not --
        see the warning emitted at the end of `train`.
        """
        curr_reward = to_log.get(self.checkpoint_metric)
        if curr_reward is None:
            raise KeyError(
                f"checkpoint_metric {self.checkpoint_metric!r} not in the step log; "
                f"available: {sorted(k for k, v in to_log.items() if isinstance(v, float))}"
            )
        if self.checkpoint_metric_mode == "min":
            curr_reward = -curr_reward
        self._ckpt_metric_history.append(curr_reward)

        ckpt = {
            "config": dict(self.cfg),
            "step": step,
            "reward": to_log.get("reward"),
            self.checkpoint_metric: to_log.get(self.checkpoint_metric),
            "model_state_dict": self.model.state_dict(),
        }

        torch.save(ckpt, os.path.join(self.output_dir, f"{self.run_name}_last.pt"))
        if not final:
            torch.save(ckpt, os.path.join(self.output_dir, f"{self.run_name}_step_{step:04}.pt"))

        if curr_reward > self.best_seen_reward:
            self.best_seen_reward = curr_reward
            self.step_at_best_seen_reward = step
            torch.save(ckpt, os.path.join(self.output_dir, f"{self.run_name}_best.pt"))

    def sample_from_policy(self, num_batches):
        """Sample sequences from the current policy without gradients."""

        with torch.no_grad():
            output_list = []
            for _ in tqdm(range(num_batches)):
                feature_dict = self.featurize_pdb(self.cfg.pdb)
                h_V, h_E, E_idx = self.encode_initial_state(feature_dict)
                out = self.rollout(feature_dict, h_V, h_E, E_idx)
                output_list.append(out)

        chain_mask = feature_dict["chain_labels"] == 0
        chain_mask = chain_mask[0]

        sequences = []
        for out in output_list:
            sequences.extend(self.get_sequences(out, chain_mask))

        return sequences