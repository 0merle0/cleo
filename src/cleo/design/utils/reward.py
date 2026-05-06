"""
Universal reward function for RL fine-tuning of ProteinMPNN.

:class:`UniversalReward` runs a configurable pipeline of metric steps
(e.g. structure prediction, distance calculations) on sampled sequences
and aggregates the results into a scalar reward via normalised
weighted summation. Steps and aggregation weights are defined in the
training config under ``reward.steps`` and ``reward.reward_aggregation``.

Each step also appends **batch** sequence stats to the training log (floats in
``{run_name}_train_metrics.csv``): mean pairwise Hamming / fraction differing,
``batch_unique_mutation_sites_wrt_ref`` (union of distinct (position, AA) vs
``ref_seq`` / first ``ref_seqs`` in the steps), and
``batch_unique_mutation_sites_wrt_consensus`` vs a per-position mode
consensus of the current batch.
"""

import os
import shutil
import numpy as np
import torch
import pandas as pd
from hydra.utils import get_method
from omegaconf import OmegaConf

from cleo.design.utils.mutation_diversity import batch_diversity_log_metrics
from cleo.design.utils.policy import alphabet


def _ref_seq_for_batch_metrics_from_steps(steps) -> str | None:
    """Get VHH parent sequence from a ``dist_to_ref_seqs`` or ``mutation_diversity`` step."""
    for s in steps or []:
        try:
            cfg = s.cfg
        except Exception:  # pragma: no cover
            continue
        if cfg is None:
            continue
        if OmegaConf.is_config(cfg):
            rs = OmegaConf.select(cfg, "ref_seq", default=None)
            if rs is not None and str(rs).strip():
                return str(rs)
            rlist = OmegaConf.select(cfg, "ref_seqs", default=None)
            if rlist is not None and len(rlist) > 0:
                return str(rlist[0])
        if isinstance(cfg, dict):
            if cfg.get("ref_seq"):
                return str(cfg["ref_seq"])
            rs = cfg.get("ref_seqs")
            if rs:
                first = rs[0] if isinstance(rs, (list, tuple)) and len(rs) > 0 else rs
                return str(first)
        if getattr(cfg, "ref_seq", None) is not None and str(getattr(cfg, "ref_seq")).strip():
            return str(cfg.ref_seq)
        rlist = getattr(cfg, "ref_seqs", None)
        if rlist is not None and len(rlist) > 0:
            return str(rlist[0])
    return None


class UniversalReward():
    """Configurable reward function for RL fine-tuning of ProteinMPNN.

    Executes a pipeline of metric steps (e.g. structure prediction, distance
    calculations) on sampled sequences, then aggregates per-metric scores
    into a single scalar reward via normalised weighted summation. Steps and
    weights are defined in the training config.
    """

    def __init__(
            self,  
            output_dir=None, 
            run_name=None,
            steps=None,
            reward_aggregation=None,
        ):

        self.output_dir = output_dir
        self.run_name = run_name
        self.steps = steps
        self.reward_aggregation = reward_aggregation
    
    def get_sequences(self, policy_output, chain_mask=None):
        """Decode integer token sequences from policy output into amino acid strings.

        Args:
            policy_output: Dict with key ``S`` containing sampled sequence
                tensors of shape ``(B, L)``.
            chain_mask: Optional boolean tensor of shape ``(L,)`` to select
                only the designed chain positions.

        Returns:
            List of amino acid strings, one per batch element.
        """
        sampled_sequences = policy_output["S"]

        if chain_mask is not None:
            sampled_sequences = sampled_sequences[:, chain_mask]

        B = sampled_sequences.shape[0]
        
        sequences = [] 
        for i in range(B):
            seq = sampled_sequences[i]
            seq_str = "".join([alphabet[int(s)] for s in seq])
            sequences.append(seq_str)
        return sequences

    def get_input_df(self, sequences):
        """Build a DataFrame with sequence names for the metric pipeline."""
        return pd.DataFrame({
            "sequence": sequences,
            "name": [f"seq_{i:04}" for i in range(len(sequences))],
            "origin.path": [f"seq_{i:04}.path" for i in range(len(sequences))],
        })
    
    
    @torch.no_grad()
    def __call__(self, step, policy_output, feature_dict, device):
        """Run the full reward pipeline: decode → metric steps → aggregate.

        Args:
            step: Current training step number (used for output directory naming).
            policy_output: Dict from the policy's ``sample`` method.
            feature_dict: Structural feature dict with ``chain_labels``.
            device: Torch device for the returned reward tensor.

        Returns:
            Tuple of (reward_tensor, log_dict) where reward_tensor has shape
            ``(B,)`` and log_dict contains per-metric batch statistics.
        """

        rundir = os.path.join(
            self.output_dir, 
            self.run_name,
            "outputs", 
            f"step_{step:04}"
        )

        if os.path.exists(rundir):
            shutil.rmtree(rundir, ignore_errors=True)
        os.makedirs(rundir, exist_ok=True)

        chain_mask = feature_dict["chain_labels"] == 0
        chain_mask = chain_mask[0]

        sequences = self.get_sequences(policy_output, chain_mask=chain_mask)
        df = self.get_input_df(sequences)

        print("********* running metric steps *********")
        for _s in self.steps:
            fn = get_method(_s.target_fn)
            _name = _s.name
            print(f"Running step: {_name} using function: {_s.target_fn}")
            _s.cfg.rundir = rundir
            _s.cfg.step = _name
            df = fn(df, _s.cfg, step_name=_name)


        # Clean up structure prediction outputs to save disk space
        af3_out_dir = os.path.join(rundir, "af3/outputs")
        if os.path.exists(af3_out_dir):
            shutil.rmtree(af3_out_dir, ignore_errors=True)

        rewards = []
        weights = []
        print("********* aggregating rewards *********")
        for m in self.reward_aggregation:
            print(f"Processing metric: {m.metric} with mode: {m.mode} and weight: {m.weight}")
            _r = torch.tensor(df[m.metric].tolist())

            _r_clamped = torch.clamp(_r, min=m.lower_bound, max=m.upper_bound)
            _r_norm = (_r_clamped - m.lower_bound) / (m.upper_bound - m.lower_bound + 1e-3)

            if m.mode == "max":
                _reward = _r_norm
            elif m.mode == "min":
                _reward = 1 - _r_norm
            elif m.mode == "avg":
                _reward = 1 - torch.abs(_r_norm - 0.5) * 2  # reward highest at 0.5
            else:
                raise ValueError(f"Unknown mode {m.mode} for metric {m.metric}")

            rewards.append(_reward*m.weight)
            weights.append(m.weight)

        # get final rewards
        reward = torch.stack(rewards, dim=1).sum(dim=1) / sum(weights)

        # make sure reward is properly padded when designing multiple chains
        if len(reward.shape) == 2 and reward.shape[1] != chain_mask.shape[0]:
            padding = torch.zeros(chain_mask.shape[0] - reward.shape[1]).unsqueeze(0).repeat(reward.shape[0], 1)
            reward = torch.cat([reward, padding], dim=1)

        # save df to output dir
        df.to_csv(os.path.join(rundir, f"metrics.csv"), index=False)

        to_log = {}
        for col in df.columns:

            # if the column is numeric, log mean, min, max
            if pd.api.types.is_numeric_dtype(df[col]):
                col_tensor = np.array(df[col].tolist())
                to_log[f"{col}_batch_mean"] = col_tensor.mean().item()
                to_log[f"{col}_batch_min"] = col_tensor.min().item()
                to_log[f"{col}_batch_max"] = col_tensor.max().item()

        # Batch-level sequence diversity: pairwise Hamming, and union of unique (pos,AA)
        # w.r.t. the template ref and w.r.t. a per-position consensus (“average” batch design).
        _ref = _ref_seq_for_batch_metrics_from_steps(self.steps)
        to_log.update(batch_diversity_log_metrics(sequences, _ref))

        return reward.to(device), to_log