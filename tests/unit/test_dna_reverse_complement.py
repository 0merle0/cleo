"""Unit tests for DNA helpers that don't require dnachisel solves."""
import pytest

rc_mod = pytest.importorskip("cleo.design.dna_utils.dna_fragment_design")
reverse_complement = rc_mod.reverse_complement


class TestReverseComplement:
    def test_known_pair(self):
        assert reverse_complement("ATCG") == "CGAT"

    def test_involution(self):
        seq = "ACGTACGTTTAGGC"
        assert reverse_complement(reverse_complement(seq)) == seq

    def test_palindrome_returns_itself(self):
        # GAATTC is an EcoRI palindrome
        assert reverse_complement("GAATTC") == "GAATTC"

    def test_lowercase_input_uppercased(self):
        assert reverse_complement("atcg") == "CGAT"

    def test_empty(self):
        assert reverse_complement("") == ""

    def test_unknown_base_raises(self):
        with pytest.raises(KeyError):
            reverse_complement("ATNX")
