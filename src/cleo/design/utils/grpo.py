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

        if self.use_ref_kl:
            self.ref_mpnn = self.load_ref_mpnn_model()

        self.avg_reward_history = []

    @property
    def legacy_surrogate(self):
        """Reproduce the pre-fix objective, for A/B comparison only.

        Two deviations from GRPO/DAPO were found by audit and corrected. This
        flag restores both so a run can be compared directly against a corrected
        one; it exists to measure the fix, not because the old path is defensible.

        1. The clipped surrogate took ``min`` of the *sums* over tokens rather
           than the sum of per-token ``min``s, collapsing 180 independent clip
           decisions into one per sequence -- so a single token past the clip
           boundary zeroed the gradient of every other token in that sequence.
        2. The advantage was recomputed inside the update loop on each random
           sub-batch, so the group-relative baseline -- the one thing GRPO
           replaces a value network with -- was estimated from half the group,
           and a sequence's advantage changed between inner updates on the same
           rollout.
        """
        return bool(getattr(self.cfg, "legacy_surrogate", False))

    @property
    def use_ref_kl(self):
        """Whether to apply the KL penalty to the frozen reference model.

        A non-zero ``kl_weight`` is enough to turn it on. Gating solely on a
        separate ``use_ref_kl`` flag meant a config could set ``kl_weight: 0.02``,
        load no reference model, compute no penalty, and report nothing amiss --
        an anchored run and an unanchored one were indistinguishable from their
        configs. Requiring both flags to agree is a trap with no upside, so the
        weight now implies the flag.
        """
        if float(getattr(self.cfg, "kl_weight", 0.0) or 0.0) > 0.0:
            return True
        return bool(getattr(self.cfg, "use_ref_kl", False))

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

    def collect_rollout(self, init_state, feature_dict):
        """Sample one rollout batch, optionally oversampling and selecting from it.

        Sampling from MPNN is nearly free; folding is essentially the entire
        cost of a step. So we can draw ``oversample x batch_size`` designs and
        fold only ``batch_size`` of them, choosing which by one of the rules in
        :mod:`cleo.design.utils.selection`. Configure with::

            selection:
              rule: maxmin          # or random / anchor_band / cluster_rep / ...
              oversample: 4
              ref_seqs: [...]       # required by the anchor-based rules
              w_anchor: 1.0

        Absent a ``selection`` block this is an ordinary rollout, so existing
        configs are unaffected.

        IMPORTANT -- this biases the gradient, deliberately. GRPO's advantage is
        group-relative: the baseline is the mean reward over a batch drawn from
        the current policy. A selected batch is not drawn from the policy, and
        because the selection depends on the sampled actions themselves, the
        importance ratios apply to a distribution we did not sample from. This
        is a heuristic, not a correction, and the ``random`` rule at the same
        oversample factor is the control that separates "diversity selection
        helped" from "oversampling helped" -- without it the arm is
        uninterpretable. Retrospective scoring on a finished run found every
        diversity rule *worse than random* at recovering passing designs, so
        the prior here is not favourable; the reason to run it anyway is that
        GRPO wants reward variance within the group, which is a different
        objective from library quality.
        """
        from cleo.design.utils.selection import RULES, Distances, consensus

        h_V_in, h_E_in, E_idx_in = init_state
        sel = getattr(self.cfg, "selection", None)
        over = int(getattr(sel, "oversample", 1)) if sel else 1
        rule = str(getattr(sel, "rule", "random")) if sel else None
        if not sel or over <= 1:
            return self.rollout(feature_dict, h_V_in, h_E_in, E_idx_in), {}
        if rule not in RULES:
            raise ValueError(f"unknown selection rule {rule!r}; have {sorted(RULES)}")

        B = self.cfg.batch_size
        pool = B * over
        # Only batch_size and randn are batch-shaped here; the rest are size-1
        # and repeated inside rollout. Copy so the caller's dict is untouched.
        fd = dict(feature_dict)
        fd["batch_size"] = pool
        fd["randn"] = torch.randn(
            [pool, feature_dict["mask"].shape[1]], device=self.device
        )
        out = self.rollout(fd, h_V_in, h_E_in, E_idx_in)

        chain_mask = (feature_dict["chain_labels"] == 0)[0]
        seqs = self.reward_fn.get_sequences(out, chain_mask)

        # Reseed per step: the stochastic rules pick an arbitrary first point,
        # and a fixed seed would start every step's traversal from the same
        # place in an otherwise different pool.
        kw = {"seed": int(torch.randint(0, 2**31 - 1, (1,)).item())}
        D = Distances(seqs)
        if rule != "random":
            refs = list(getattr(sel, "ref_seqs", []) or [])
            if refs:
                kw["anchor_d"] = D.to_seq(consensus(refs))
            elif "anchor" in rule:
                raise ValueError(f"selection rule {rule!r} needs `ref_seqs`")
            kw["w_anchor"] = float(getattr(sel, "w_anchor", 1.0))
        idx = torch.as_tensor(RULES[rule](D, B, **kw), device=self.device).long()

        # Subset every batch-shaped tensor; leave scalars and size-1 entries.
        out = {
            k: (v[idx] if torch.is_tensor(v) and v.dim() and v.shape[0] == pool else v)
            for k, v in out.items()
        }
        return out, {"selection_pool": pool, "selection_rule_is_random": rule == "random"}

    def advantage(self, rewards):
        """Standardised advantage for a group of rollouts. -> tensor like `rewards`.

        Either group-relative (the GRPO default: centre and scale within the
        group) or against a running history of past batch means, per the
        ``use_avg_reward`` config flag.
        """
        if "use_avg_reward" in self.cfg and self.cfg.use_avg_reward:
            hist = torch.tensor(
                self.avg_reward_history[-self.cfg.avg_reward_window:]
            ).to(self.device)
            centre = hist.mean()
            scale = hist.std() if hist.numel() > 1 else 1.0
        else:
            centre = torch.mean(rewards)
            scale = torch.std(rewards)
        return (rewards - centre) / (scale + 1e-3)

    def train_step(self, step, init_state, feature_dict):
        """GRPO/DAPO update: collect one rollout batch, then perform
        ``N_updates`` clipped-surrogate gradient steps on random sub-batches.

        Uses asymmetric clipping (clip_eps_low / clip_eps_high) as proposed
        in DAPO (https://arxiv.org/pdf/2503.14476). Advantages are either
        group-relative (normalised within the current batch) or computed
        against a running reward history, controlled by the ``use_avg_reward``
        config flag.
        """

        to_log = {}

        # --- Collect experience (no gradients) ---
        with torch.no_grad():
            out, sel_log = self.collect_rollout(init_state, feature_dict)
            to_log.update(sel_log)

            batched_rewards, metrics = self.reward_fn(step, out, feature_dict, self.device)
            to_log.update(metrics)
            to_log["reward"] = batched_rewards.mean().cpu().item()

        all_log_probs = torch.clone(out["log_probs"])
        all_S = torch.clone(out["S"])
        all_decoding_order = torch.clone(out["decoding_order"])
        all_batched_rewards = torch.clone(batched_rewards)
        B = all_S.shape[0]

        # Group-relative advantage over the WHOLE rollout, computed once. GRPO's
        # baseline is the mean reward of the group sampled from one prompt, so it
        # has to see the whole group; and a rollout's advantage must not change
        # between the gradient steps taken on it.
        self.avg_reward_history.append(all_batched_rewards.mean().cpu().item())
        if not self.legacy_surrogate:
            all_A = self.advantage(all_batched_rewards)

        # --- Policy updates ---
        for i in range(self.cfg.N_updates):
            sample_idx = torch.randperm(B)[:self.cfg.update_batch_size]
            S = all_S[sample_idx]
            decoding_order = all_decoding_order[sample_idx]
            old_batched_log_probs = all_log_probs[sample_idx]
            batched_rewards = all_batched_rewards[sample_idx]

            self.optimizer.zero_grad()

            h_V_in, h_E_in, E_idx_in = init_state
            h_V_in.requires_grad = True
            h_E_in.requires_grad = True

            if self.use_ref_kl:
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

            if self.legacy_surrogate:
                self.avg_reward_history.append(batched_rewards.mean().cpu().item())
                A = self.advantage(batched_rewards)
            else:
                A = all_A[sample_idx]

            r_clipped = torch.clamp(r, 1 - self.cfg.clip_eps_low, 1 + self.cfg.clip_eps_high)

            if self.legacy_surrogate and A.shape != r.shape:
                obj = torch.min(r.sum(dim=-1) * A, r_clipped.sum(dim=-1) * A).mean()
            else:
                # Per-token min, then a masked mean. Taking the min first is what
                # makes the clip a trust region: a token whose ratio has left the
                # band stops receiving gradient while its neighbours keep theirs.
                # The mean is over *designed* tokens only -- `chain_mask` is zero
                # at fixed residues, whose log-probs the rollout already zeroes,
                # so including them would contribute r = 1 each and make the step
                # size depend on how much of the protein is held fixed.
                A_ = A.unsqueeze(-1) if A.dim() < r.dim() else A
                per_token = torch.min(r * A_, r_clipped * A_)
                m = out["chain_mask"]
                obj = (per_token * m).sum() / m.sum().clamp(min=1.0)

            if self.use_ref_kl:
                ref_batched_log_probs = (ref_batched_log_probs * seq_mask).sum(dim=-1)
                kl_ratio = torch.exp(ref_batched_log_probs - batched_log_probs)
                kl = kl_ratio - (ref_batched_log_probs - batched_log_probs) - 1
                obj = obj - self.cfg.kl_weight * kl.mean()
                to_log["kl_penalty"] = kl.mean().cpu().item()

            loss = -obj
            loss.backward()
            self.optimizer.step()

        return to_log
