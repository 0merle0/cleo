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
        self._check_best_of_n(steps)

    @staticmethod
    def _check_best_of_n(steps):
        """Refuse a pipeline that folds N times and never reduces the N.

        A reward step must return one row per sampled sequence. An oracle run at
        ``num_diffusion_samples > 1`` with ``per_sample_rows`` returns N rows per
        sequence, so something has to collapse them, and the reduction has to be
        over the benchmark's own criterion rather than over AF3 confidence --
        that is what ``best_of_n_from_df`` is for.

        This is checked rather than auto-inserted. ``best_of_n_from_df`` needs to
        know which metric prefix to reduce, and guessing it from step order would
        silently pick the wrong column in any pipeline carrying two metric steps.

        Since the oracle default became best-of-5, omitting the reduction is the
        easy mistake, and left unchecked it fails far downstream as a length
        mismatch between the reward tensor and the batch.
        """
        if not steps:
            return
        names = [getattr(s, "name", "?") for s in steps]
        has_reduction = any(
            str(getattr(s, "target_fn", "")).endswith("best_of_n_from_df")
            for s in steps
        )
        if has_reduction:
            return
        for s in steps:
            cfg = getattr(s, "cfg", None)
            if cfg is None:
                continue
            # af3_from_df is the only step that implements per-sample rows, so
            # it is the only one carrying the best-of-5 default. boltz_from_df
            # reads neither key and always returns one row per sequence, and
            # applying the default to every step with a cfg would flag steps
            # that cannot produce the rows this check is about.
            is_oracle = str(getattr(s, "target_fn", "")).endswith("af3_from_df")
            if "num_diffusion_samples" not in cfg and not is_oracle:
                continue
            n = int(cfg.get("num_diffusion_samples", 5 if is_oracle else 1))
            if n > 1 and bool(cfg.get("per_sample_rows", n > 1)):
                raise ValueError(
                    f"reward step '{getattr(s, 'name', '?')}' folds {n} samples "
                    f"per sequence, but no best_of_n_from_df step follows it, so "
                    f"the {n} rows per sequence are never collapsed.\n"
                    f"Pipeline is: {names}\n"
                    f"Add after the metric step:\n"
                    f"  - name: bo{n}\n"
                    f"    target_fn: cleo.design.utils.rfd2_benchmark.best_of_n_from_df\n"
                    f"    cfg:\n"
                    f"      group_col: name\n"
                    f"      metric_prefix: <name of the metric step, e.g. ame>\n"
                    f"Or set 'num_diffusion_samples: 1' on that step to opt out."
                )


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
            _norm_mode = str(getattr(m, "normalize", "") or "")

            if _norm_mode == "rank":
                # Within-batch rank, ties averaged, mapped to [0,1]. Bounds are
                # deliberately ignored.
                #
                # Fixed bounds cannot survive training. They must be calibrated
                # to the metric's range, but that range moves by an order of
                # magnitude: a step-0 T=1.0 batch on a 4-chain target had motif
                # RMSD 9-37 A, so a 6 A upper bound clipped 100% of it, leaving
                # the term exactly constant and the reward driven entirely by
                # the other metric. Rank is invariant to scale and to outliers,
                # so it yields usable gradient at every stage of training, which
                # is all GRPO's group-relative advantage needs.
                _finite = torch.isfinite(_r)
                _r_norm = torch.full_like(_r, 0.5, dtype=torch.float32)
                if _finite.sum() > 1:
                    vals = _r[_finite]
                    # average ranks so tied values receive identical credit
                    order = vals.argsort()
                    ranks = torch.empty_like(order, dtype=torch.float32)
                    ranks[order] = torch.arange(len(vals), dtype=torch.float32)
                    for v in torch.unique(vals):
                        tie = vals == v
                        if tie.sum() > 1:
                            ranks[tie] = ranks[tie].mean()
                    _r_norm[_finite] = ranks / max(len(vals) - 1, 1)
            else:
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

            # Optional per-metric batch standardisation.
            #
            # `weight` alone does not control influence. GRPO consumes
            # *within-batch* differences, so a metric that barely varies across
            # a batch contributes almost nothing however it is weighted. On a
            # 16-sequence AME batch the motif-RMSD term has sd ~0.33 while batch
            # mutation diversity has sd ~0.03: nominally equal weights give the
            # diversity term under a fifth of the actual influence, and no
            # choice of fixed bounds fixes that robustly, since the spreads move
            # as training proceeds.
            #
            # With `normalize: zscore` each term is standardised over the batch
            # before weighting, so equal weight means equal influence by
            # construction. Off by default: it changes reward semantics, and
            # single-metric rewards do not need it (GRPO already standardises
            # the aggregate).
            if str(getattr(m, "normalize", "") or "") == "zscore":
                _sd = _reward.std()
                if _sd > 1e-8:
                    _reward = (_reward - _reward.mean()) / _sd
                else:
                    # Degenerate batch: every sequence identical on this metric.
                    # Contribute nothing rather than amplifying float noise.
                    _reward = torch.zeros_like(_reward)

            rewards.append(_reward*m.weight)
            weights.append(m.weight)

        # get final rewards
        reward = torch.stack(rewards, dim=1).sum(dim=1) / sum(weights)

        # make sure reward is properly padded when designing multiple chains
        if len(reward.shape) == 2 and reward.shape[1] != chain_mask.shape[0]:
            padding = torch.zeros(chain_mask.shape[0] - reward.shape[1]).unsqueeze(0).repeat(reward.shape[0], 1)
            reward = torch.cat([reward, padding], dim=1)

        # Persist the aggregated scalar alongside the metrics it came from.
        # Without it the per-step record shows what was measured but not what
        # was optimized, so reward shaping cannot be audited after the fact.
        # Guarded: in the multi-chain case `reward` has been padded to a 2D
        # per-position tensor and no longer aligns with the rows.
        if reward.dim() == 1 and reward.shape[0] == len(df):
            df["reward"] = reward.tolist()

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