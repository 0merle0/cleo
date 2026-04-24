"""Unit tests for cleo.optimize.utils.fragment."""
import pandas as pd
import pytest
import torch

from cleo.optimize.utils.fragment import (
    featurize_fragments,
    get_all_sequences,
    get_fragment_dictionary,
    make_all_sequences,
)


class TestGetFragmentDictionary:
    def test_parses_csv(self, tmp_path):
        df = pd.DataFrame(
            {
                "name": ["1.a", "1.b", "2.a", "2.b"],
                "fragment": [1, 1, 2, 2],
                "seq": ["AA", "AB", "CC", "CD"],
            }
        )
        csv = tmp_path / "frags.csv"
        df.to_csv(csv, index=False)

        frag_dict = get_fragment_dictionary(csv)
        assert set(frag_dict.keys()) == {1, 2}
        assert len(frag_dict[1]) == 2
        assert len(frag_dict[2]) == 2

    def test_entries_alphabetically_sorted(self, tmp_path):
        df = pd.DataFrame(
            {
                "name": ["1.z", "1.a", "1.m"],
                "fragment": [1, 1, 1],
                "seq": ["Z", "A", "M"],
            }
        )
        csv = tmp_path / "frags.csv"
        df.to_csv(csv, index=False)

        frag_dict = get_fragment_dictionary(csv)
        names = [n for n, _ in frag_dict[1]]
        assert names == sorted(names)


class TestEnumerateAllSequences:
    def test_count_is_product_of_region_sizes(self, tiny_fragment_dictionary):
        all_seqs = get_all_sequences(tiny_fragment_dictionary)
        expected = 1
        for v in tiny_fragment_dictionary.values():
            expected *= len(v)
        assert len(all_seqs) == expected

    def test_sequences_are_unique(self, tiny_fragment_dictionary):
        all_seqs = get_all_sequences(tiny_fragment_dictionary)
        names = [n for n, _ in all_seqs]
        sequences = [s for _, s in all_seqs]
        assert len(set(names)) == len(names)
        assert len(set(sequences)) == len(sequences)

    def test_sequence_is_concatenation_of_fragments(self, tiny_fragment_dictionary):
        # Use a non-default delimiter to stay clean of real fragment names
        seqs = make_all_sequences(tiny_fragment_dictionary, to_join_on=":")
        name_to_seq = dict(seqs)
        # "1.a:2.a:3.a" -> "AA" + "CC" + "EE" = "AACCEE"
        assert name_to_seq["1.a:2.a:3.a"] == "AACCEE"
        # "1.b:2.b:3.b" -> "AB" + "CD" + "EF" = "ABCDEF"
        assert name_to_seq["1.b:2.b:3.b"] == "ABCDEF"

    def test_single_region(self):
        frag_dict = {1: [("1.a", "X"), ("1.b", "Y"), ("1.c", "Z")]}
        all_seqs = get_all_sequences(frag_dict)
        assert len(all_seqs) == 3


class TestFeaturizeFragments:
    def test_shape_and_dtype(self, tiny_fragment_dictionary):
        names = ["1.a:2.a:3.a", "1.b:2.b:3.b"]
        out = featurize_fragments(names, tiny_fragment_dictionary, to_split_on=":")
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 3)
        assert out.dtype == torch.int64

    def test_indices_match_dictionary_order(self, tiny_fragment_dictionary):
        # dict entries are sorted in the fixture; "1.a" is index 0, "1.b" is index 1
        out = featurize_fragments(
            ["1.a:2.a:3.a", "1.b:2.b:3.b"],
            tiny_fragment_dictionary,
            to_split_on=":",
        )
        assert out[0].tolist() == [0, 0, 0]
        assert out[1].tolist() == [1, 1, 1]

    def test_raises_on_wrong_fragment_count(self, tiny_fragment_dictionary):
        # names have only 2 tokens but dict has 3 regions
        with pytest.raises(AssertionError):
            featurize_fragments(["1.a:2.a"], tiny_fragment_dictionary, to_split_on=":")
