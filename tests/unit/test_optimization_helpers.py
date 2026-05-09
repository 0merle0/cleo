"""Unit tests for small pure helpers in cleo.optimize.utils.optimization."""
import torch

from cleo.optimize.utils.optimization import get_feasible_mask, get_one_hot


class TestGetFeasibleMask:
    def test_uniform_sizes(self, tiny_fragment_dictionary):
        mask = get_feasible_mask(tiny_fragment_dictionary)
        assert mask.shape == (3, 2)
        assert mask.dtype == torch.bool
        assert mask.all()

    def test_ragged_regions_padded_with_false(self):
        frag_dict = {
            1: [("1.a", "A"), ("1.b", "B"), ("1.c", "C")],
            2: [("2.a", "D")],
            3: [("3.a", "E"), ("3.b", "F")],
        }
        mask = get_feasible_mask(frag_dict)
        assert mask.shape == (3, 3)
        # Row 0: 3 feasible entries
        assert mask[0].tolist() == [True, True, True]
        # Row 1: 1 feasible, 2 masked
        assert mask[1].tolist() == [True, False, False]
        # Row 2: 2 feasible, 1 masked
        assert mask[2].tolist() == [True, True, False]


class TestGetOneHot:
    def test_list_input(self):
        out = get_one_hot([0, 1, 2], num_classes=4)
        assert out.shape == (3, 4)
        # Each row sums to 1 (valid one-hot)
        assert torch.equal(out.sum(dim=-1), torch.ones(3, dtype=out.dtype))

    def test_tensor_input(self):
        x = torch.tensor([0, 3])
        out = get_one_hot(x, num_classes=4)
        assert torch.equal(out[0], torch.tensor([1, 0, 0, 0]))
        assert torch.equal(out[1], torch.tensor([0, 0, 0, 1]))

    def test_default_20_classes(self):
        out = get_one_hot([5])
        assert out.shape == (1, 20)
