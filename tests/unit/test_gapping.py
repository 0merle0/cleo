"""Slice-2 step 3.5 (SPEC 4.5 / 6.8): CDR length sampling + gap-synthesis.

Two pure, fast units:
- ``sample_cdr_lengths`` — fixed override wins, ranges sample in-bounds + deterministic under a
  seeded RNG, span-width fallback, error paths.
- ``apply_cdr_gaps`` — excises native CDR residues and splices sampled-length gap nodes into a
  parsed ``protein_dict``: correct new length, chain_mask marks *only* the gap nodes, framework
  rows preserved verbatim, gap sequence = unknown, design chain renumbered contiguously.
"""
import random

import pytest

torch = pytest.importorskip("torch")

from cleo.design.data.gapping import _UNK_AA, apply_cdr_gaps
from cleo.design.data.mask import MaskError, as_chain_list, sample_cdr_lengths


# --- length sampler -------------------------------------------------------- #

def test_fixed_override_wins():
    out = sample_cdr_lengths({"cdr_lengths": {"H3": 12}, "cdr_length_ranges": {"H3": [1, 3]}})
    assert out == {"H3": 12}


def test_range_sampling_in_bounds_and_deterministic():
    params = {"cdr_length_ranges": {"H1": [5, 9], "H3": [8, 20]}}
    a = sample_cdr_lengths(params, rng=random.Random(0))
    b = sample_cdr_lengths(params, rng=random.Random(0))
    assert a == b                                       # same seed => same draw
    assert 5 <= a["H1"] <= 9 and 8 <= a["H3"] <= 20


def test_span_width_fallback():
    """A CDR with neither override nor range falls back to its native span width."""
    out = sample_cdr_lengths({"cdr_spans": {"H2": [51, 57]}})
    assert out == {"H2": 6}


def test_reversed_range_raises():
    with pytest.raises(MaskError, match="reversed"):
        sample_cdr_lengths({"cdr_length_ranges": {"H1": [9, 5]}})


def test_nonpositive_length_raises():
    with pytest.raises(MaskError):
        sample_cdr_lengths({"cdr_lengths": {"H1": 0}})


# --- gap-synthesis --------------------------------------------------------- #

def _protein_dict(L=10, chain="A"):
    """A minimal parsed protein_dict (single chain, un-batched) for gapping."""
    g = torch.Generator().manual_seed(0)
    return {
        "X": torch.randn(L, 4, 3, generator=g),
        "mask": torch.ones(L, dtype=torch.int32),
        "R_idx": torch.arange(100, 100 + L, dtype=torch.int32),   # non-contiguous native numbering
        "chain_labels": torch.zeros(L, dtype=torch.int32),
        "S": torch.arange(L, dtype=torch.int32) % 20,             # anything < 20 (not the UNK token)
        "xyz_37": torch.randn(L, 37, 3, generator=g),
        "xyz_37_m": torch.ones(L, 37, dtype=torch.int32),
        "chain_letters": [chain] * L,
    }


def test_gap_synthesis_length_and_mask():
    pd = _protein_dict(L=10)
    out = apply_cdr_gaps(pd, "A", {"H1": [3, 6]}, {"H1": 5})   # excise 3 natives, insert 5 gaps

    assert out["X"].shape[0] == 12                             # 10 - 3 + 5
    cm = out["chain_mask"]
    assert cm.sum().item() == 5                                # only the gap nodes are designable
    assert cm.nonzero().flatten().tolist() == [3, 4, 5, 6, 7]  # spliced where native H1 started


def test_gap_synthesis_preserves_framework_and_marks_gaps():
    pd = _protein_dict(L=10)
    out = apply_cdr_gaps(pd, "A", {"H1": [3, 6]}, {"H1": 5})

    # framework rows carried verbatim (0,1,2 then native 6..9 land at 8..11)
    assert torch.equal(out["X"][0:3], pd["X"][0:3])
    assert torch.equal(out["X"][8:12], pd["X"][6:10])
    assert torch.equal(out["S"][0:3], pd["S"][0:3])
    assert torch.equal(out["S"][8:12], pd["S"][6:10])
    # gap sequence = unknown token, gap backbone copied from the N-side stem (row 2) => finite
    assert torch.all(out["S"][3:8] == _UNK_AA)
    assert torch.isfinite(out["X"][3:8]).all()
    assert torch.equal(out["X"][3], pd["X"][2])                # N-side stem coordinate


def test_cdrs_are_non_existent_in_mask_mode():
    """SPEC 4.1 mask-mode (initial training): the parser resolves CDR positions as *non-existent* —
    they carry no native identity, only UNK tokens, so the policy cannot copy the native loop. (The
    future post-training 'un-mask' mode that keeps native CDRs + docks is a separate path.)"""
    pd = _protein_dict(L=10)
    out = apply_cdr_gaps(pd, "A", {"H1": [3, 6]}, {"H1": 5})
    cm = out["chain_mask"].bool()
    assert torch.all(out["S"][cm] == _UNK_AA)                     # native CDR sequence is gone
    # the masked region is independent of whatever native residues occupied the CDR columns
    pd2 = _protein_dict(L=10)
    pd2["S"][3:6] = (pd2["S"][3:6] + 7) % 20
    out2 = apply_cdr_gaps(pd2, "A", {"H1": [3, 6]}, {"H1": 5})
    assert torch.equal(out["S"][cm], out2["S"][out2["chain_mask"].bool()])


def test_gap_synthesis_renumbers_design_chain_contiguously():
    pd = _protein_dict(L=10)
    out = apply_cdr_gaps(pd, "A", {"H1": [3, 6]}, {"H1": 5})
    assert out["R_idx"].tolist() == list(range(12))           # gap nodes adjacent to their stems


def test_gap_at_chain_start_uses_cside_stem():
    pd = _protein_dict(L=10)
    out = apply_cdr_gaps(pd, "A", {"H1": [0, 2]}, {"H1": 4})   # CDR at the very start
    # first two natives excised; gaps take the C-side stem (original row 2)
    assert out["X"].shape[0] == 12                            # 10 - 2 + 4
    assert out["chain_mask"].nonzero().flatten().tolist() == [0, 1, 2, 3]
    assert torch.equal(out["X"][0], pd["X"][2])


def test_overlapping_spans_raise():
    pd = _protein_dict(L=10)
    with pytest.raises(MaskError, match="overlap"):
        apply_cdr_gaps(pd, "A", {"H1": [3, 6], "H2": [5, 8]}, {"H1": 3, "H2": 3})


def test_out_of_bounds_span_raises():
    pd = _protein_dict(L=10)
    with pytest.raises(MaskError, match="out of bounds"):
        apply_cdr_gaps(pd, "A", {"H1": [8, 12]}, {"H1": 3})


# --- Fv (two-chain) robustness --------------------------------------------- #

def _fv_protein_dict(Lh=10, Ll=8):
    """A minimal two-chain (H then L) parsed protein_dict for the paired-Fv path."""
    g = torch.Generator().manual_seed(0)
    L = Lh + Ll
    return {
        "X": torch.randn(L, 4, 3, generator=g),
        "mask": torch.ones(L, dtype=torch.int32),
        "R_idx": torch.arange(100, 100 + L, dtype=torch.int32),
        "chain_labels": torch.tensor([0] * Lh + [1] * Ll, dtype=torch.int32),
        "S": torch.arange(L, dtype=torch.int32) % 20,
        "xyz_37": torch.randn(L, 37, 3, generator=g),
        "xyz_37_m": torch.ones(L, 37, dtype=torch.int32),
        "chain_letters": ["H"] * Lh + ["L"] * Ll,
    }


def test_fv_gaps_both_chains():
    """H* spans route to chain 0, L* spans to chain 1; gap nodes land in both chains."""
    pd = _fv_protein_dict(Lh=10, Ll=8)                 # H rows 0..9, L rows 10..17
    out = apply_cdr_gaps(
        pd, ["H", "L"],
        {"H1": [3, 6], "L1": [2, 5]},                  # positional ranges *into each chain*
        {"H1": 4, "L1": 3},
    )
    # new length: (10 - 3 + 4) + (8 - 3 + 3) = 11 + 8 = 19
    assert out["X"].shape[0] == 19
    cm = out["chain_mask"]
    assert cm.sum().item() == 7                        # 4 (H1) + 3 (L1) designable
    letters = [str(c) for c in out["chain_letters"]]
    gap_rows = cm.nonzero().flatten().tolist()
    gap_chains = {letters[i] for i in gap_rows}
    assert gap_chains == {"H", "L"}                    # both chains contributed gap nodes
    assert torch.all(out["S"][cm.bool()] == _UNK_AA)   # every designable node is unknown-seq
    assert torch.isfinite(out["X"]).all()


def test_light_span_without_second_chain_raises():
    """An L* CDR on a single-chain (VHH) design is a hard error, not a silent misroute."""
    pd = _protein_dict(L=10)
    with pytest.raises(MaskError, match="light chain"):
        apply_cdr_gaps(pd, "H", {"L1": [2, 5]}, {"L1": 3})


def test_as_chain_list_normalizes_all_forms():
    assert as_chain_list("H") == ["H"]                 # VHH single chain
    assert as_chain_list("H L") == ["H", "L"]          # Fv serialized as whitespace-joined string
    assert as_chain_list(["H", "L"]) == ["H", "L"]     # already a list
