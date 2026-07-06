"""
Group Relative Policy Optimization (GRPO) fine-tuning of ProteinMPNN.

Extends :class:`PolicyMPNN` with a clipped surrogate objective
(following DAPO — https://arxiv.org/pdf/2503.14476) and an optional
KL penalty to a frozen reference ProteinMPNN model. Advantages are
computed relative to the batch mean (group-relative), and multiple
gradient updates are performed per rollout batch.
"""

import torch
from cleo.design.utils.policy import (
    PolicyMPNN,
    alphabet,
    PROTEIN_MPNN_CKPT_PATH,
    LIGAND_MPNN_CKPT_PATH,
)
from cleo.design.protein_mpnn_utils.model_utils import ProteinMPNN as ProteinMPNNModel


class PolicyMPNNvGRPO(PolicyMPNN):
    """GRPO/DAPO variant of PolicyMPNN with clipped surrogate loss and optional KL penalty."""

    def __init__(self, cfg):
        super().__init__(cfg)

        if hasattr(self.cfg, "use_ref_kl") and self.cfg.use_ref_kl:
            self.ref_mpnn = self.load_ref_mpnn_model()

        self.avg_reward_history = []

    def load_ref_mpnn_model(self):
        """Load a frozen copy of the base ProteinMPNN as a KL reference."""

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

        model = ProteinMPNNModel(
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

        ckpt = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return model.to(self.device)

    def train_step(self, step, init_state, feature_dict, reward_fn=None):
        """GRPO/DAPO update: collect one rollout batch, then perform
        ``N_updates`` clipped-surrogate gradient steps on random sub-batches.

        Uses asymmetric clipping (clip_eps_low / clip_eps_high) as proposed
        in DAPO (https://arxiv.org/pdf/2503.14476). Advantages are either
        group-relative (normalised within the current batch) or computed
        against a running reward history, controlled by the ``use_avg_reward``
        config flag.
        """

        reward_fn = reward_fn if reward_fn is not None else self.reward_fn
        to_log = {}

        # --- Collect experience (no gradients) ---
        with torch.no_grad():
            # init_state=None (step 6): encode fresh under no_grad (also computes/caches the epitope
            # embeddings so this rollout sees the SAME conditioning as the grad-tracked updates).
            if init_state is None:
                h_V_in, h_E_in, E_idx_in = self.encode_initial_state(feature_dict)
            else:
                h_V_in, h_E_in, E_idx_in = init_state
            out = self.rollout(feature_dict, h_V_in, h_E_in, E_idx_in)

            batched_rewards, metrics = reward_fn(step, out, feature_dict, self.device)
            to_log.update(metrics)
            to_log["reward"] = batched_rewards.mean().cpu().item()

        all_log_probs = torch.clone(out["log_probs"])
        all_S = torch.clone(out["S"])
        all_decoding_order = torch.clone(out["decoding_order"])
        all_batched_rewards = torch.clone(batched_rewards)
        B = all_S.shape[0]

        # --- Policy updates ---
        for i in range(self.cfg.N_updates):
            sample_idx = torch.randperm(B)[:self.cfg.update_batch_size]
            S = all_S[sample_idx]
            decoding_order = all_decoding_order[sample_idx]
            old_batched_log_probs = all_log_probs[sample_idx]
            batched_rewards = all_batched_rewards[sample_idx]

            self.optimizer.zero_grad()

            # init_state=None (step 6): re-encode grad-tracked each update so the framework +
            # epitope encoders receive gradients. Else use the cached detached leaf (decoder-only).
            if init_state is None:
                h_V_in, h_E_in, E_idx_in = self.encode_initial_state(feature_dict, grad=True)
            else:
                h_V_in, h_E_in, E_idx_in = init_state
                h_V_in.requires_grad = True
                h_E_in.requires_grad = True

            if hasattr(self.cfg, "use_ref_kl") and self.cfg.use_ref_kl:
                with torch.no_grad():
                    ref_out = self.rollout(
                        feature_dict, h_V_in, h_E_in, E_idx_in,
                        decoding_order=decoding_order, sampled_actions=S, model=self.ref_mpnn,
                    )
                    ref_batched_log_probs = ref_out["log_probs"]

            out = self.rollout(
                feature_dict, h_V_in, h_E_in, E_idx_in,
                decoding_order=decoding_order, sampled_actions=S, model=self.model,
            )

            seq_mask = torch.nn.functional.one_hot(out["S"], num_classes=len(alphabet)).float()

            batched_log_probs = (out["log_probs"] * seq_mask).sum(dim=-1)
            old_batched_log_probs = (old_batched_log_probs * seq_mask).sum(dim=-1)
            r = torch.exp(batched_log_probs - old_batched_log_probs)

            self.avg_reward_history.append(batched_rewards.mean().cpu().item())

            if "use_avg_reward" in self.cfg and self.cfg.use_avg_reward:
                reward_history = torch.tensor(self.avg_reward_history[-self.cfg.avg_reward_window:]).to(self.device)
                mean_history = reward_history.mean()
                std_history = reward_history.std() if reward_history.numel() > 1 else 1.0
                A = (batched_rewards - mean_history) / (std_history + 1e-3)
            else:
                A = (batched_rewards - torch.mean(batched_rewards)) / (torch.std(batched_rewards) + 1e-3)

            r_clipped = torch.clamp(r, 1 - self.cfg.clip_eps_low, 1 + self.cfg.clip_eps_high)

            if A.shape != r.shape:
                min_term1 = r.sum(dim=-1) * A
                min_term2 = r_clipped.sum(dim=-1) * A
            else:
                min_term1 = r * A
                min_term2 = r_clipped * A

            obj = torch.min(min_term1, min_term2).mean()

            if hasattr(self.cfg, "use_ref_kl") and self.cfg.use_ref_kl:
                ref_batched_log_probs = (ref_batched_log_probs * seq_mask).sum(dim=-1)
                kl_ratio = torch.exp(ref_batched_log_probs - batched_log_probs)
                kl = kl_ratio - (ref_batched_log_probs - batched_log_probs) - 1
                obj = obj - self.cfg.kl_weight * kl.mean()
                to_log["kl_penalty"] = kl.mean().cpu().item()

            loss = -obj
            loss.backward()
            self.optimizer.step()

        return to_log
