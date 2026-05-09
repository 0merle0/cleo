"""Unit tests for cleo.design.utils.sequence."""
import pytest

from cleo.design.utils.sequence import (
    compute_dist_to_ref_seqs_from_df,
    get_dist_from_seqs,
)


class TestGetDistFromSeqs:
    def test_identity_is_zero(self):
        result = get_dist_from_seqs("ACDE", ["ACDE"])
        assert result == {"min": 0, "avg": 0.0, "max": 0}

    def test_single_mismatch(self):
        result = get_dist_from_seqs("A", ["B"])
        assert result == {"min": 1, "avg": 1.0, "max": 1}

    def test_aggregates_across_refs(self):
        result = get_dist_from_seqs("ACDE", ["ACDE", "ACDF", "AGDH"])
        assert result["min"] == 0
        assert result["max"] == 2
        assert result["avg"] == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            get_dist_from_seqs("ACDE", ["ACDEF"])


class TestComputeDistToRefSeqsFromDf:
    def test_adds_expected_columns(self, tiny_sequence_df, cfg_ns):
        cfg = cfg_ns(ref_seqs=["ACDE"])
        out = compute_dist_to_ref_seqs_from_df(tiny_sequence_df, cfg, step_name="d2r")
        for suffix in ("min", "avg", "max"):
            assert f"d2r_{suffix}" in out.columns
        # original columns preserved
        assert "sequence" in out.columns
        assert "name" in out.columns

    def test_preserves_row_count(self, tiny_sequence_df, cfg_ns):
        cfg = cfg_ns(ref_seqs=["ACDE"])
        out = compute_dist_to_ref_seqs_from_df(tiny_sequence_df, cfg)
        assert len(out) == len(tiny_sequence_df)

    def test_distance_values_are_correct(self, tiny_sequence_df, cfg_ns):
        cfg = cfg_ns(ref_seqs=["ACDE"])
        out = compute_dist_to_ref_seqs_from_df(tiny_sequence_df, cfg, step_name="d")
        by_name = out.set_index("name")
        # ACDE vs ACDE = 0; ACDF vs ACDE = 1; AGDE vs ACDE = 1
        assert by_name.loc["s0", "d_min"] == 0
        assert by_name.loc["s1", "d_min"] == 1
        assert by_name.loc["s2", "d_min"] == 1
