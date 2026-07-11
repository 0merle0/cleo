"""Slice-2 step 3 (SPEC 4.1): coordinate-free CDR edges in ProteinFeatures.

The surgery must (a) leave the stock path byte-identical when off, and (b) when on, make the
edge features + connectivity **completely invariant to CDR coordinates** (stem-borrowed kNN +
learned unknown-distance token on every CDR-incident edge, both directions) — which also makes
it NaN-safe against garbage/gapped CDR coords. Tested directly on ProteinFeatures (tensor-pure,
no MPNN weights beyond the feature extractor itself).
"""
import pytest

torch = pytest.importorskip("torch")

from cleo.design.protein_mpnn_utils.model_utils import ProteinFeatures

L = 12
CDR = [5, 6, 7]                      # interior designable span; stems 4 / 8
FRAME = [i for i in range(L) if i not in CDR]


def _features(seed=0, cdr=CDR):
    torch.manual_seed(seed)
    X = torch.randn(1, L, 4, 3) * 5.0 + 10.0        # backbone N, CA, C, O
    mask = torch.ones(1, L)
    R_idx = torch.arange(L)[None]                    # residue numbers
    chain_labels = torch.zeros(1, L, dtype=torch.long)
    chain_mask = torch.zeros(1, L)
    chain_mask[0, cdr] = 1.0                         # 1 = designable = CDR
    return {"X": X, "mask": mask, "R_idx": R_idx, "chain_labels": chain_labels, "chain_mask": chain_mask}


# --- baseline: flag OFF is untouched --------------------------------------- #

def test_flag_off_has_no_token_param():
    pf = ProteinFeatures(128, 128)
    assert pf.coord_free_cdr_edges is False
    assert not hasattr(pf, "unknown_rbf")
    assert all("unknown" not in n for n, _ in pf.named_parameters())


def test_flag_off_ignores_chain_mask():
    """With the flag off, presence/absence of chain_mask must not change anything."""
    pf = ProteinFeatures(128, 128)
    fd = _features()
    E1, Eidx1 = pf(fd)
    E2, Eidx2 = pf({k: v for k, v in fd.items() if k != "chain_mask"})
    assert torch.equal(E1, E2) and torch.equal(Eidx1, Eidx2)


def test_flag_off_still_depends_on_cdr_coords():
    """Sanity: the stock path DOES use CDR coordinates (so the on-path invariance is meaningful)."""
    pf = ProteinFeatures(128, 128)
    fd = _features()
    E1, _ = pf(fd)
    fd2 = dict(fd); fd2["X"] = fd["X"].clone(); fd2["X"][0, CDR] += 100.0
    E2, _ = pf(fd2)
    assert not torch.allclose(E1, E2)


# --- flag ON: coordinate-free CDR edges ------------------------------------ #

def test_flag_on_has_token_param():
    pf = ProteinFeatures(128, 128, coord_free_cdr_edges=True)
    assert pf.coord_free_cdr_edges is True
    assert pf.unknown_rbf.shape == (16 * 25,)


def test_flag_on_invariant_to_cdr_coordinates():
    """The whole point: edge features AND connectivity are independent of CDR coords."""
    pf = ProteinFeatures(128, 128, coord_free_cdr_edges=True)
    fd = _features()
    E1, Eidx1 = pf(fd)
    fd2 = dict(fd); fd2["X"] = fd["X"].clone()
    fd2["X"][0, CDR] = torch.randn(len(CDR), 4, 3) * 50.0 - 30.0   # wildly different CDR coords
    E2, Eidx2 = pf(fd2)
    assert torch.equal(Eidx1, Eidx2)                # connectivity invariant (stem-borrowed kNN)
    assert torch.allclose(E1, E2, atol=1e-5)        # edge features invariant (token-replaced)


def test_flag_on_is_nan_safe():
    """Gapped/garbage CDR coords (NaN) must not leak into any edge feature."""
    pf = ProteinFeatures(128, 128, coord_free_cdr_edges=True)
    fd = _features()
    fd["X"][0, CDR] = float("nan")
    E, Eidx = pf(fd)
    assert torch.isfinite(E).all()
    assert torch.isfinite(Eidx.float()).all()


def test_flag_on_framework_only_matches_when_no_cdr():
    """With an all-framework chain_mask the surgery is a no-op vs. running without it."""
    pf = ProteinFeatures(128, 128, coord_free_cdr_edges=True)
    fd = _features(cdr=[])                           # nothing designable
    E1, Eidx1 = pf(fd)
    E2, Eidx2 = pf({k: v for k, v in fd.items() if k != "chain_mask"})
    assert torch.equal(Eidx1, Eidx2)
    assert torch.allclose(E1, E2, atol=1e-6)


def test_unknown_token_receives_gradient():
    pf = ProteinFeatures(128, 128, coord_free_cdr_edges=True)
    E, _ = pf(_features())
    E.sum().backward()
    assert pf.unknown_rbf.grad is not None
    assert pf.unknown_rbf.grad.abs().sum() > 0
