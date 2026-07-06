"""Composing dataset (SPEC 6.9): independent target x scaffold x CDR-length sampling.

Pure-logic units — no structures parsed except a tiny gemmi-written VD fixture used to
check the post-gap ``design_chains`` length arithmetic end-to-end. The composer must:
  - honor the VHH/Fv mix knob (``vhh_fraction``),
  - pin one CDR-length draw so the featurizer (``cdr_lengths`` override) and the reward
    (``design_chains`` lengths) agree,
  - emit a uniform ``design_chains`` record (1 entry VHH, 2 Fv) whose per-chain ``length``
    == native VD length - native span widths + sampled lengths.
"""
import json
import random

import pytest

from cleo.design.data.composer import (
    ComposerError,
    ComposingDataset,
    _load_scaffolds,
    _load_targets,
    _parse_epitope_residues,
)


# --- pool loading ---------------------------------------------------------- #

def _write_targets(path):
    path.write_text(
        "id,split,antigen_file,msa_dir,epitope_residues,epitope_net_charge\n"
        "t_train,train,/ag/t.pdb,/msa/t,10 11 12,-2\n"
        "t_val,val,/ag/v.pdb,/msa/v,5 6,1\n"
        "t_empty,train,/ag/e.pdb,/msa/e,,0\n"           # no epitope -> dropped
    )


def _write_scaffolds(path):
    path.write_text(
        "scaffold_id,kind,structure_file,design_chain,cdr_spans,msa_dir_H,msa_dir_L\n"
        'v1,vhh,/sc/v1.pdb,H,"{""H1"": [2, 5]}",/msa/h1,\n'
        'f1,fv,/sc/f1.pdb,H L,"{""H1"": [2, 5], ""L1"": [1, 3]}",/msa/h2,/msa/l2\n'
        'v_bad,vhh,/sc/vb.pdb,H,"{""H1"": [2, 5]}",,\n'   # missing MSA -> dropped
    )


def test_parse_epitope_residues():
    assert _parse_epitope_residues("10 11 12") == [10, 11, 12]
    assert _parse_epitope_residues("") == []


def test_load_targets_filters_split_and_empty(tmp_path):
    p = tmp_path / "t.csv"
    _write_targets(p)
    train = _load_targets(str(p), "train")
    assert [r["target_id"] for r in train] == ["t_train"]       # val + empty excluded
    assert train[0]["epitope_residues"] == [10, 11, 12]
    assert train[0]["epitope_net_charge"] == -2
    assert len(_load_targets(str(p), None)) == 2                 # both non-empty splits


def test_load_scaffolds_splits_modality_and_drops_missing_msa(tmp_path):
    p = tmp_path / "s.csv"
    _write_scaffolds(p)
    vhh, fv, dropped = _load_scaffolds(str(p))
    assert [s["scaffold_id"] for s in vhh] == ["v1"]
    assert [s["scaffold_id"] for s in fv] == ["f1"]
    assert dropped == 1                                         # v_bad
    assert fv[0]["msa_dirs"] == {"H": "/msa/h2", "L": "/msa/l2"}


# --- modality mix knob ----------------------------------------------------- #

def _fake_native(lengths):
    """Stub NativeSeqProvider.seq -> a sequence of the requested per-chain length."""
    class _N:
        def seq(self, path, chain):
            return "A" * lengths[chain]
    return _N()


def _composer(tmp_path, vhh_fraction=0.5, ranges=None, native_lengths=None):
    t = tmp_path / "t.csv"; _write_targets(t)
    s = tmp_path / "s.csv"; _write_scaffolds(s)
    targets = _load_targets(str(t), "train")
    vhh, fv, _ = _load_scaffolds(str(s))
    ds = ComposingDataset(
        targets, vhh, fv, "antibody_interface_composed", str(tmp_path),
        vhh_fraction=vhh_fraction, cdr_length_ranges=ranges, rng=random.Random(0),
    )
    if native_lengths is not None:
        ds.native = _fake_native(native_lengths)
    return ds


def test_vhh_fraction_all_vhh(tmp_path):
    ds = _composer(tmp_path, vhh_fraction=1.0, native_lengths={"H": 10, "L": 8})
    kinds = {ds._compose_row()["params"]["kind"] for _ in range(30)}
    assert kinds == {"vhh"}


def test_vhh_fraction_all_fv(tmp_path):
    ds = _composer(tmp_path, vhh_fraction=0.0, native_lengths={"H": 10, "L": 8})
    kinds = {ds._compose_row()["params"]["kind"] for _ in range(30)}
    assert kinds == {"fv"}


def test_vhh_fraction_out_of_range_raises(tmp_path):
    t = tmp_path / "t.csv"; _write_targets(t)
    s = tmp_path / "s.csv"; _write_scaffolds(s)
    targets = _load_targets(str(t), "train")
    vhh, fv, _ = _load_scaffolds(str(s))
    with pytest.raises(ComposerError, match="vhh_fraction"):
        ComposingDataset(targets, vhh, fv, "r", str(tmp_path), vhh_fraction=1.5)


# --- design_chains length arithmetic --------------------------------------- #

def test_vhh_design_chains_length(tmp_path):
    # native H VD length 10; H1 native span [2,5) width 3, fixed sampled length 6 -> 10-3+6=13
    ds = _composer(tmp_path, vhh_fraction=1.0, native_lengths={"H": 10, "L": 8})
    row = ds._compose_row()
    dcs = row["params"]["design_chains"]
    assert len(dcs) == 1
    assert dcs[0]["framework_msa_dir"] == "/msa/h1"
    assert dcs[0]["cdr_spans"] == {"H1": [2, 5]}
    # cdr_lengths pinned into params == what the featurizer will reuse
    sampled = row["params"]["cdr_lengths"]["H1"]
    assert dcs[0]["length"] == 10 - 3 + sampled


def test_fv_emits_two_records_routed_by_prefix(tmp_path):
    ds = _composer(tmp_path, vhh_fraction=0.0, native_lengths={"H": 10, "L": 8})
    row = ds._compose_row()
    dcs = row["params"]["design_chains"]
    assert len(dcs) == 2
    assert [d["framework_msa_dir"] for d in dcs] == ["/msa/h2", "/msa/l2"]  # H then L order
    assert dcs[0]["cdr_spans"] == {"H1": [2, 5]}       # H record gets only H* spans
    assert dcs[1]["cdr_spans"] == {"L1": [1, 3]}       # L record gets only L* spans
    cl = row["params"]["cdr_lengths"]
    assert dcs[0]["length"] == 10 - 3 + cl["H1"]       # H: width 3
    assert dcs[1]["length"] == 8 - 2 + cl["L1"]        # L: width 2


def test_pinned_lengths_make_split_consistent(tmp_path):
    """sum(design_chains.length) must equal the total decoded length the oracle will split,
    i.e. sum over chains of (native_len - width + sampled)."""
    ds = _composer(tmp_path, vhh_fraction=0.0, native_lengths={"H": 10, "L": 8})
    row = ds._compose_row()
    dcs = row["params"]["design_chains"]
    cl = row["params"]["cdr_lengths"]
    total = (10 - 3 + cl["H1"]) + (8 - 2 + cl["L1"])
    assert sum(d["length"] for d in dcs) == total


def test_length_ranges_drive_diversity(tmp_path):
    ds = _composer(
        tmp_path, vhh_fraction=1.0,
        ranges={"H1": [4, 12]}, native_lengths={"H": 10, "L": 8},
    )
    lens = {ds._compose_row()["params"]["cdr_lengths"]["H1"] for _ in range(40)}
    assert len(lens) > 1 and all(4 <= x <= 12 for x in lens)


def test_row_is_reward_compatible_shape(tmp_path):
    ds = _composer(tmp_path, vhh_fraction=0.0, native_lengths={"H": 10, "L": 8})
    row = ds._compose_row()
    assert row["design_chain"] == "H L"
    assert row["antigen_structure"] == "/ag/t.pdb"
    assert row["params"]["antigen_msa_dir"] == "/msa/t"
    assert row["params"]["epitope_residues"] == [10, 11, 12]
    assert row["params"]["epitope_net_charge"] == -2
    # design_chains must be JSON-serializable (broadcast through the reward df)
    json.dumps(row["params"]["design_chains"])
