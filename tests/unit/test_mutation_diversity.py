"""Unit tests for cleo.design.utils.mutation_diversity."""
import pandas as pd
import pytest

from cleo.design.utils.mutation_diversity import (
    _get_mutation_sets,
    _mutation_counts,
    mutation_diversity_from_df,
)


class TestMutationSets:
    def test_no_mutations(self):
        sets = _get_mutation_sets(["ACDE", "ACDE"], "ACDE")
        assert sets == [set(), set()]

    def test_single_mutation(self):
        sets = _get_mutation_sets(["ACDF"], "ACDE")
        assert sets == [{(3, "F")}]

    def test_counts_sum_equals_total_mutations(self):
        sequences = ["ACDF", "AGDE", "ACDF"]
        ref = "ACDE"
        sets = _get_mutation_sets(sequences, ref)
        counts = _mutation_counts(sets)
        # total mutation slots across sequences == sum of counts
        assert sum(len(s) for s in sets) == sum(counts.values())


class TestMutationDiversityFromDf:
    def test_identical_batch_no_marginal(self, cfg_ns):
        df = pd.DataFrame({"name": ["a", "b"], "sequence": ["ACDF", "ACDF"]})
        cfg = cfg_ns(ref_seq="ACDE")
        out = mutation_diversity_from_df(df, cfg, step_name="md")
        # shared mutation -> marginal_count is zero for everyone
        assert (out["md_marginal_count"] == 0).all()
        # fractional credit is 1/k = 1/2 for each
        assert out["md_fractional_score"].tolist() == [0.5, 0.5]

    def test_fully_distinct_batch_full_marginal(self, cfg_ns):
        df = pd.DataFrame(
            {"name": ["a", "b", "c"], "sequence": ["BCDE", "ABDE", "ACBE"]}
        )
        cfg = cfg_ns(ref_seq="ACDE")
        out = mutation_diversity_from_df(df, cfg, step_name="md")
        # every mutation is exclusive
        assert out["md_marginal_count"].tolist() == [1, 1, 1]
        assert out["md_marginal_fraction"].tolist() == [1.0, 1.0, 1.0]

    def test_expected_columns(self, cfg_ns):
        df = pd.DataFrame({"name": ["a"], "sequence": ["ACDE"]})
        cfg = cfg_ns(ref_seq="ACDE")
        out = mutation_diversity_from_df(df, cfg, step_name="md")
        for col in (
            "md_total_muts",
            "md_marginal_count",
            "md_marginal_fraction",
            "md_fractional_score",
            "md_fractional_normalized",
        ):
            assert col in out.columns

    def test_zero_mutations_no_divzero(self, cfg_ns):
        df = pd.DataFrame({"name": ["a"], "sequence": ["ACDE"]})
        cfg = cfg_ns(ref_seq="ACDE")
        out = mutation_diversity_from_df(df, cfg, step_name="md")
        # No crash on division when there are zero mutations
        assert out["md_total_muts"].iloc[0] == 0
        assert out["md_marginal_fraction"].iloc[0] == pytest.approx(0.0)
