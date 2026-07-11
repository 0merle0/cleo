"""Reward registry — bind a dataset Example to a per-example UniversalReward (SPEC 6.4).

CLEO builds ONE ``UniversalReward`` from ``cfg.reward`` for a whole run. The
dataset-driven harness instead needs a reward that is **specialized per example**:
same target-agnostic pipeline (``steps`` + ``reward_aggregation`` from the example's
reward package), but with the example's resolved inputs (``${native.seq.T}``,
``${row.params.*}``, ...) seeded as **df columns** so the existing step functions
(``protenix_from_df`` reads ``antigen_sequence`` / ``framework_msa_dir`` / ... off
the df) consume them unchanged.

``RewardRegistry.bind(example)`` returns a ``BoundReward`` — a ``UniversalReward``
subclass whose ``get_input_df`` injects those columns. ``requires`` / ``inputs`` are
registry metadata and are **not** passed to ``UniversalReward`` (``inputs`` are
stripped from the steps here; they've already been resolved into columns).
"""
from __future__ import annotations

from omegaconf import OmegaConf

from cleo.design.data.binding import DESIGN_SEQ
from cleo.design.utils.reward import UniversalReward


class BoundReward(UniversalReward):
    """A UniversalReward that seeds per-example resolved inputs as df columns."""

    def __init__(self, resolved_inputs, **kwargs):
        super().__init__(**kwargs)
        self._resolved_inputs = resolved_inputs

    def get_input_df(self, sequences):
        df = super().get_input_df(sequences)
        n = len(df)
        for col, val in self._resolved_inputs.items():
            if val is DESIGN_SEQ:
                df[col] = list(df["sequence"])
            else:
                # broadcast one target value to every sampled sequence; wrapping in a
                # list keeps dict/list values (cdr_spans, epitope_residues) intact per cell.
                df[col] = [val] * n
        return df


class RewardRegistry:
    """Builds a per-example ``BoundReward`` from a dataset's reward packages."""

    def __init__(self, dataset, output_dir, run_name):
        self.dataset = dataset
        self.output_dir = output_dir
        self.run_name = run_name

    def bind(self, example) -> BoundReward:
        package = self.dataset.package(example.reward)
        resolved = self.dataset.bind_inputs(example)

        # steps minus the `inputs` metadata (already resolved into df columns)
        steps = [{k: v for k, v in step.items() if k != "inputs"} for step in package["steps"]]

        return BoundReward(
            resolved_inputs=resolved,
            output_dir=self.output_dir,
            run_name=self.run_name,
            steps=OmegaConf.create(steps),
            reward_aggregation=OmegaConf.create(package["reward_aggregation"]),
        )
