"""Pure-logic units for the Protenix oracle's N-designed-chain support (VHH + Fv).

No GPU / no Protenix subprocess here — these cover the input builders and summary/overlap parsers
that decide how many designed chains a task has, how the decoded sequence is split per chain, how
each chain's framework MSA is gapped, and how the designed-vs-antigen interface is aggregated.
"""
import json

import pytest

from cleo.design.utils.protenix_oracle import (
    _build_design_fw_msa,
    _build_task,
    _designed_chain_ids,
    _epitope_overlap,
    _gapped_fw_homolog_rows,
    _parse_summary,
    _split_seq,
)


# --- chain-id assignment / sequence split ---------------------------------- #

def test_designed_chain_ids_vhh_and_fv():
    assert _designed_chain_ids(1, "L") == ["A"]           # VHH
    assert _designed_chain_ids(2, "L") == ["A", "B"]      # Fv, disjoint from antigen "L"


def test_designed_chain_ids_antigen_collision_raises():
    with pytest.raises(ValueError, match="collides"):
        _designed_chain_ids(2, "B")                       # antigen "B" == 2nd designed id


def test_split_seq_partitions_in_order():
    assert _split_seq("HHHHLLL", [4, 3]) == ["HHHH", "LLL"]


def test_split_seq_length_mismatch_raises():
    with pytest.raises(ValueError, match="!= sum of chain lengths"):
        _split_seq("HHHHLLL", [4, 2])


# --- task construction ----------------------------------------------------- #

def test_build_task_orders_designed_then_antigen():
    designed = [("HHHH", "A", None), ("LLL", "B", None)]
    task = _build_task("fv0", designed, "AGSEQ", "L", None, use_msa=False)
    ids = [s["proteinChain"]["id"][0] for s in task["sequences"]]
    seqs = [s["proteinChain"]["sequence"] for s in task["sequences"]]
    assert ids == ["A", "B", "L"]                         # designed chains first, antigen last
    assert seqs == ["HHHH", "LLL", "AGSEQ"]
    assert task["name"] == "fv0"


# --- interface aggregation over designed chains ---------------------------- #

def test_parse_summary_single_chain_matches_legacy_index():
    summary = {"iptm": 0.5, "chain_pair_iptm": [[1.0, 0.7], [0.7, 1.0]],
               "chain_pair_pae_mean": [[1.0, 8.0], [8.0, 1.0]]}
    out = _parse_summary(summary, designed_idxs=(0,), ag_idx=1)
    assert out["interface_iptm"] == 0.7                   # == old chain_pair[0][1]
    assert out["interface_pae"] == 8.0


def test_parse_summary_fv_aggregates_best_interface():
    # 3 chains: designed A(0), B(1), antigen(2). B binds antigen better than A.
    cp_iptm = [[1.0, 0.2, 0.3], [0.2, 1.0, 0.8], [0.3, 0.8, 1.0]]
    cp_pae = [[1.0, 20.0, 15.0], [20.0, 1.0, 5.0], [15.0, 5.0, 1.0]]
    summary = {"chain_pair_iptm": cp_iptm, "chain_pair_pae_mean": cp_pae}
    out = _parse_summary(summary, designed_idxs=range(2), ag_idx=2)
    assert out["interface_iptm"] == 0.8                   # max over {A-ag=0.3, B-ag=0.8}
    assert out["interface_pae"] == 5.0                    # min over {A-ag=15, B-ag=5}


# --- per-chain framework MSA gapping (Fv => two independent gappings) ------- #

def test_fv_builds_one_gapped_msa_per_chain(tmp_path):
    # Heavy framework a3m: query + 1 homolog; H CDR span [2,5) gets gapped.
    h_a3m = tmp_path / "H.a3m"
    h_a3m.write_text(">q\nMKAAAWY\n>h1\nMKCDEWY\n")
    l_a3m = tmp_path / "L.a3m"
    l_a3m.write_text(">q\nDIQMTQ\n>h2\nDIVMTQ\n")

    h_dir = _build_design_fw_msa(str(tmp_path / "d0"), str(h_a3m), {"H1": [2, 5]}, "MKQRSWY")
    l_dir = _build_design_fw_msa(str(tmp_path / "d1"), str(l_a3m), {"L1": [1, 3]}, "DIQMTQ")

    h_lines = (open(h_dir + "/non_pairing.a3m").read()).splitlines()
    l_lines = (open(l_dir + "/non_pairing.a3m").read()).splitlines()
    assert h_lines[0] == ">query" and h_lines[1] == "MKQRSWY"   # query row = designed segment
    # homolog H1 columns (2,3,4) gapped, framework columns preserved
    assert h_lines[3] == "MK---WY"
    assert l_lines[1] == "DIQMTQ"
    assert l_lines[3] == "D--MTQ"                                # L1 cols (1,2) gapped


def test_gapped_homolog_rows_independent_per_chain(tmp_path):
    a3m = tmp_path / "fw.a3m"
    a3m.write_text(">q\nABCDEF\n>h\nABCDEF\n")
    rows = _gapped_fw_homolog_rows(str(a3m), ((1, 3),))          # gap cols 1,2
    assert rows == ((">h", "A--DEF"),)                           # header kept verbatim


# --- epitope overlap over multiple designed chains ------------------------- #

def test_epitope_overlap_counts_any_designed_chain(tmp_path):
    gemmi = pytest.importorskip("gemmi")
    # Build a tiny 3-chain model: antigen T with 2 residues, designed A near res 1, B near res 2.
    st = gemmi.Structure()
    st.spacegroup_hm = "P 1"
    m = gemmi.Model("1")

    def _chain(name, resnum, x):
        c = gemmi.Chain(name)
        r = gemmi.Residue()
        r.name = "ALA"
        r.seqid = gemmi.SeqId(resnum, " ")
        a = gemmi.Atom()
        a.name = "CA"
        a.element = gemmi.Element("C")
        a.pos = gemmi.Position(x, 0, 0)
        r.add_atom(a)
        c.add_residue(r)
        return c

    # antigen residues 10 (x=0) and 11 (x=10); designed A at x=1 (near 10), B at x=11 (near 11)
    ag = gemmi.Chain("L")
    for resnum, x in [(10, 0.0), (11, 10.0)]:
        for r in _chain("L", resnum, x):
            ag.add_residue(r)
    m.add_chain(_chain("A", 1, 1.0))     # ~1A from antigen res 10
    m.add_chain(_chain("B", 1, 11.0))    # ~1A from antigen res 11
    m.add_chain(ag)
    st.add_model(m)
    cif = tmp_path / "fv.cif"
    st.make_mmcif_document().write_file(str(cif))

    # intended epitope = {10}. Both antigen residues are predicted-interface (A hits 10, B hits 11),
    # so precision = |{10,11} & {10}| / |{10,11}| = 1/2.
    ov = _epitope_overlap(str(cif), "L", ["A", "B"], [10], cutoff=5.0)
    assert ov == pytest.approx(0.5)
