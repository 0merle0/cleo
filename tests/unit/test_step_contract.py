"""Contract tests for reward step functions.

Every `*_from_df` step in the reward pipeline must follow the same contract:
    fn(df_input, cfg, step_name=...) -> DataFrame
where the returned frame contains all input columns plus new columns prefixed
with `{step_name}_`. These tests pin that contract for the lightweight steps
and assert signature shape for the heavy ones (boltz, experimental_predictor)
that need real models / subprocesses to run.
"""
import inspect

import pytest

from cleo.design.utils import (
    experimental_predictor,
    mutation_diversity,
    oracle,
    sequence,
)

ALL_STEP_FNS = [
    sequence.compute_dist_to_ref_seqs_from_df,
    mutation_diversity.mutation_diversity_from_df,
    oracle.boltz_from_df,
    experimental_predictor.experimental_predictor_from_df,
]


class TestStepSignature:
    @pytest.mark.parametrize("fn", ALL_STEP_FNS)
    def test_three_params(self, fn):
        params = list(inspect.signature(fn).parameters.values())
        assert len(params) == 3, f"{fn.__name__} should take (df_input, cfg, step_name)"

    @pytest.mark.parametrize("fn", ALL_STEP_FNS)
    def test_step_name_has_default(self, fn):
        params = inspect.signature(fn).parameters
        assert "step_name" in params
        assert params["step_name"].default is not inspect.Parameter.empty

    @pytest.mark.parametrize("fn", ALL_STEP_FNS)
    def test_step_name_default_matches_module_convention(self, fn):
        # Reward pipeline lets users override step_name in the config, but the
        # default should be a non-empty string so naive callers still get a prefix.
        default = inspect.signature(fn).parameters["step_name"].default
        assert isinstance(default, str) and default


class TestSequenceStepContract:
    def test_returns_dataframe(self, tiny_sequence_df, cfg_ns):
        cfg = cfg_ns(ref_seqs=["ACDE"])
        out = sequence.compute_dist_to_ref_seqs_from_df(tiny_sequence_df, cfg)
        assert hasattr(out, "columns")

    def test_preserves_input_columns(self, tiny_sequence_df, cfg_ns):
        cfg = cfg_ns(ref_seqs=["ACDE"])
        out = sequence.compute_dist_to_ref_seqs_from_df(tiny_sequence_df, cfg)
        for col in tiny_sequence_df.columns:
            assert col in out.columns

    def test_preserves_row_count(self, tiny_sequence_df, cfg_ns):
        cfg = cfg_ns(ref_seqs=["ACDE"])
        out = sequence.compute_dist_to_ref_seqs_from_df(tiny_sequence_df, cfg)
        assert len(out) == len(tiny_sequence_df)

    def test_new_columns_are_prefixed(self, tiny_sequence_df, cfg_ns):
        cfg = cfg_ns(ref_seqs=["ACDE"])
        out = sequence.compute_dist_to_ref_seqs_from_df(
            tiny_sequence_df, cfg, step_name="my_step"
        )
        new_cols = set(out.columns) - set(tiny_sequence_df.columns)
        assert new_cols, "step_fn must add at least one new column"
        for col in new_cols:
            assert col.startswith("my_step_"), f"new column {col!r} should be prefixed"

    def test_step_name_changes_prefix(self, tiny_sequence_df, cfg_ns):
        cfg = cfg_ns(ref_seqs=["ACDE"])
        out_a = sequence.compute_dist_to_ref_seqs_from_df(
            tiny_sequence_df, cfg, step_name="alpha"
        )
        out_b = sequence.compute_dist_to_ref_seqs_from_df(
            tiny_sequence_df, cfg, step_name="beta"
        )
        new_a = set(out_a.columns) - set(tiny_sequence_df.columns)
        new_b = set(out_b.columns) - set(tiny_sequence_df.columns)
        assert all(c.startswith("alpha_") for c in new_a)
        assert all(c.startswith("beta_") for c in new_b)
        assert new_a.isdisjoint(new_b)


class TestMutationDiversityStepContract:
    def test_returns_dataframe(self, tiny_sequence_df, cfg_ns):
        cfg = cfg_ns(ref_seq="ACDE")
        out = mutation_diversity.mutation_diversity_from_df(tiny_sequence_df, cfg)
        assert hasattr(out, "columns")

    def test_preserves_input_columns(self, tiny_sequence_df, cfg_ns):
        cfg = cfg_ns(ref_seq="ACDE")
        out = mutation_diversity.mutation_diversity_from_df(tiny_sequence_df, cfg)
        for col in tiny_sequence_df.columns:
            assert col in out.columns

    def test_preserves_row_count(self, tiny_sequence_df, cfg_ns):
        cfg = cfg_ns(ref_seq="ACDE")
        out = mutation_diversity.mutation_diversity_from_df(tiny_sequence_df, cfg)
        assert len(out) == len(tiny_sequence_df)

    def test_new_columns_are_prefixed(self, tiny_sequence_df, cfg_ns):
        cfg = cfg_ns(ref_seq="ACDE")
        out = mutation_diversity.mutation_diversity_from_df(
            tiny_sequence_df, cfg, step_name="div"
        )
        new_cols = set(out.columns) - set(tiny_sequence_df.columns)
        assert new_cols
        for col in new_cols:
            assert col.startswith("div_")
