"""
Universal reward function for RL fine-tuning of ProteinMPNN.

:class:`UniversalReward` runs a configurable pipeline of metric steps
(e.g. structure prediction, distance calculations) on sampled sequences
and aggregates the results into a scalar reward via normalised
weighted summation. Steps and aggregation weights are defined in the
training config under ``reward.steps`` and ``reward.reward_aggregation``.
"""

import os
import shutil
import numpy as np
import torch
import pandas as pd
from hydra.utils import get_method

from cleo.design.utils.policy import alphabet


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
    
    
    def _aggregate_rewards(self, df):
        """Normalized weighted sum of the configured metrics -> per-design reward tensor.

        Fails fast on non-finite metrics: a single NaN/Inf would poison the whole GRPO group
        (mean/std -> NaN -> all advantages NaN), so surface it loudly instead of masking — we
        should not be seeing NaNs; a failed structure prediction or a buggy reward step is the
        usual cause and is worth investigating, not silently zeroing.
        """
        rewards, weights = [], []
        for m in self.reward_aggregation:
            print(f"Processing metric: {m.metric} with mode: {m.mode} and weight: {m.weight}")
            _r = torch.tensor(df[m.metric].tolist())

            _bad = ~torch.isfinite(_r)
            if _bad.any():
                _names = df["name"].tolist() if "name" in df.columns else list(range(len(_r)))
                _offenders = [_names[i] for i in _bad.nonzero().flatten().tolist()]
                raise ValueError(
                    f"reward metric {m.metric!r} produced non-finite values (NaN/Inf) for designs "
                    f"{_offenders}: {_r[_bad].tolist()}. This poisons the GRPO batch — investigate "
                    f"(failed structure prediction? buggy reward step?) rather than masking.")

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

            rewards.append(_reward * m.weight)
            weights.append(m.weight)

        return torch.stack(rewards, dim=1).sum(dim=1) / sum(weights)

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

        print("********* aggregating rewards *********")
        reward = self._aggregate_rewards(df)

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

        
        return reward.to(device), to_log