"""Unit tests for cleo.design.utils.geom."""
import numpy as np
import pytest
import torch

from cleo.design.utils.geom import (
    angle_between_three_points,
    compute_dihedral,
    torch_get_rmsd,
)


class TestRMSD:
    def test_identical_coords_zero(self):
        a = torch.randn(10, 3)
        rmsd, _ = torch_get_rmsd(a.clone(), a.clone())
        # torch_get_rmsd has an eps=1e-6 floor -> sqrt gives ~1e-3 for identical inputs
        assert float(rmsd) == pytest.approx(0.0, abs=5e-3)

    def test_translation_invariant(self):
        a = torch.randn(10, 3)
        b = a + torch.tensor([5.0, -2.0, 7.0])
        rmsd, _ = torch_get_rmsd(a, b)
        # torch_get_rmsd has an eps=1e-6 floor -> sqrt gives ~1e-3 for identical inputs
        assert float(rmsd) == pytest.approx(0.0, abs=5e-3)

    def test_rotation_recovered(self):
        theta = torch.tensor(0.7)
        R = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta), 0.0],
                [torch.sin(theta), torch.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        a = torch.randn(10, 3)
        b = a @ R.T
        rmsd, _ = torch_get_rmsd(a, b)
        # torch_get_rmsd has an eps=1e-6 floor -> sqrt gives ~1e-3 for identical inputs
        assert float(rmsd) == pytest.approx(0.0, abs=5e-3)

    def test_shape_mismatch_raises(self):
        a = torch.randn(10, 3)
        b = torch.randn(8, 3)
        with pytest.raises(AssertionError):
            torch_get_rmsd(a, b)


class TestAngle:
    def test_right_angle(self):
        A = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 0.0, 0.0])
        C = np.array([0.0, 1.0, 0.0])
        assert angle_between_three_points(A, B, C) == pytest.approx(90.0, abs=1e-6)

    def test_straight_line(self):
        A = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 0.0, 0.0])
        C = np.array([-1.0, 0.0, 0.0])
        assert angle_between_three_points(A, B, C) == pytest.approx(180.0, abs=1e-6)

    def test_coincident_rays(self):
        A = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 0.0, 0.0])
        C = np.array([2.0, 0.0, 0.0])
        assert angle_between_three_points(A, B, C) == pytest.approx(0.0, abs=1e-6)


class TestDihedral:
    def test_planar_cis(self):
        # All four points on the x-y plane, cis configuration -> 0 degrees
        p0 = np.array([0.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        assert compute_dihedral(p0, p1, p2, p3) == pytest.approx(0.0, abs=1e-6)

    def test_planar_trans(self):
        # All four points on the x-y plane, trans configuration -> 180 degrees
        p0 = np.array([0.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, -1.0, 0.0])
        assert abs(compute_dihedral(p0, p1, p2, p3)) == pytest.approx(180.0, abs=1e-6)

    def test_plus_ninety(self):
        p0 = np.array([0.0, 1.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 1.0])
        assert compute_dihedral(p0, p1, p2, p3) == pytest.approx(90.0, abs=1e-6)
