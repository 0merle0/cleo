"""Unit tests for the dataset-driven training harness data layer (SPEC 6).

Exercises DesignDataset load/validation, ${...} template binding, and CDR->
fixed_residues masking against a small committed 2-chain fixture (chain A =
design, chain T = antigen), plus every hard-error path.
"""
import json
from pathlib import Path

import pytest

from cleo.design.data import (
    BindingError,
    DesignDataset,
    DesignDatasetError,
    MaskError,
    resolve_fixed_residues,
    scan_chain_ids,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_complex.pdb"
CDR = {"H1": [2, 5], "H2": [8, 11]}          # positional into chain A (len 15) -> 6 designed
N_DESIGN = sum(e - s for s, e in CDR.values())


def _reward_dir(tmp_path, name="iface", requires=None, inputs=None):
    requires = requires if requires is not None else ["antigen_sequence", "cdr_spans", "epitope_residues"]
    inputs = inputs if inputs is not None else {
        "antigen_sequence": "${native.seq.T}",
        "cdr_spans": "${row.params.cdr_spans}",
        "epitope_residues": "${row.params.epitope_residues}",
    }
    d = tmp_path / "reward"
    d.mkdir(exist_ok=True)
    yaml_txt = (
        f"requires: {json.dumps(requires)}\n"
        "steps:\n"
        "  - name: protenix\n"
        "    target_fn: cleo.design.utils.protenix_oracle.protenix_from_df\n"
        "    cfg: {use_msa: true}\n"
        "    inputs:\n"
        + "".join(f"      {k}: {v}\n" for k, v in inputs.items())
    )
    (d / f"{name}.yaml").write_text(yaml_txt)
    return d


def _row(**over):
    row = {
        "id": "t0", "task": "nanobody_design", "reward": "iface",
        "structure": str(FIXTURE), "design_chain": "A",
        "design_regions": ["cdr_H1", "cdr_H2"],
        "params": {"cdr_spans": CDR, "epitope_residues": [3, 4, 5]},
    }
    row.update(over)
    return row


def _jsonl(tmp_path, rows):
    p = tmp_path / "ds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_load_and_sample(tmp_path):
    ds = DesignDataset.load(_jsonl(tmp_path, [_row()]), str(_reward_dir(tmp_path)))
    assert len(ds) == 1
    ex = ds.sample()
    assert ex.id == "t0" and ex.design_chain == "A" and ex.cdr_spans == CDR
    assert len(ds.sample(3)) == 3


def test_bind_inputs_resolves_native_and_params(tmp_path):
    ds = DesignDataset.load(_jsonl(tmp_path, [_row()]), str(_reward_dir(tmp_path)))
    resolved = ds.bind_inputs(ds.sample())
    assert resolved["antigen_sequence"] == "NLAFALSELD"     # chain T native seq
    assert resolved["cdr_spans"] == CDR
    assert resolved["epitope_residues"] == [3, 4, 5]


def test_scan_chain_ids():
    assert scan_chain_ids(str(FIXTURE)) == {"A", "T"}


# --------------------------------------------------------------------------- #
# mask: fixed_residues is the complement of the CDR positions
# --------------------------------------------------------------------------- #
def test_mask_complement_matches_parse_pdb(tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841
    from cleo.design.protein_mpnn_utils.data_utils import parse_PDB

    pdct, *_mid, icodes, _ = parse_PDB(
        str(FIXTURE), device="cpu", atom_context_num=1, chains="", parse_all_atoms=False
    )
    cl = list(pdct["chain_letters"])
    ridx = list(pdct["R_idx"].cpu().numpy())
    n_total = len(cl)

    fixed = resolve_fixed_residues(cl, ridx, icodes, "A", CDR).split()
    assert len(fixed) == n_total - N_DESIGN

    fixedset = set(fixed)
    apos = [i for i, c in enumerate(cl) if str(c) == "A"]
    design_tokens = {f"{cl[i]}{int(ridx[i])}{icodes[i]}" for s, e in CDR.values() for i in apos[s:e]}
    assert design_tokens.isdisjoint(fixedset)                       # CDRs are designable
    tchain = [f"T{int(ridx[i])}{icodes[i]}" for i in range(n_total) if str(cl[i]) == "T"]
    assert all(tok in fixedset for tok in tchain)                   # antigen fully fixed


def test_mask_out_of_bounds_span():
    with pytest.raises(MaskError):
        resolve_fixed_residues(["A"] * 5, list(range(5)), [""] * 5, "A", {"H1": [0, 999]})


def test_mask_light_chain_without_second_design_chain():
    with pytest.raises(MaskError):
        resolve_fixed_residues(["A"] * 5, list(range(5)), [""] * 5, "A", {"L1": [0, 2]})


# --------------------------------------------------------------------------- #
# hard-error validation paths (each names the row id)
# --------------------------------------------------------------------------- #
def test_absent_design_chain_errors(tmp_path):
    with pytest.raises(DesignDatasetError, match="design_chain 'Z' absent"):
        DesignDataset.load(_jsonl(tmp_path, [_row(design_chain="Z")]), str(_reward_dir(tmp_path)))


def test_missing_structure_errors(tmp_path):
    with pytest.raises(DesignDatasetError, match="structure not found"):
        DesignDataset.load(_jsonl(tmp_path, [_row(structure="/no/such.pdb")]), str(_reward_dir(tmp_path)))


def test_missing_required_param_errors(tmp_path):
    bad = _row(params={"cdr_spans": CDR})   # no epitope_residues
    with pytest.raises(BindingError, match="epitope_residues"):
        DesignDataset.load(_jsonl(tmp_path, [bad]), str(_reward_dir(tmp_path)))


def test_native_missing_chain_errors(tmp_path):
    rd = _reward_dir(tmp_path, name="badnative", requires=["antigen_sequence"],
                     inputs={"antigen_sequence": "${native.seq.Q}"})
    with pytest.raises(BindingError, match="chain absent"):
        DesignDataset.load(_jsonl(tmp_path, [_row(reward="badnative")]), str(rd))


def test_requires_without_input_errors(tmp_path):
    rd = _reward_dir(tmp_path, name="unbound", requires=["framework_msa_dir"],
                     inputs={"antigen_sequence": "${native.seq.T}"})
    with pytest.raises(BindingError, match="requires 'framework_msa_dir'"):
        DesignDataset.load(_jsonl(tmp_path, [_row(reward="unbound")]), str(rd))


def test_missing_reward_package_errors(tmp_path):
    with pytest.raises(DesignDatasetError, match="reward package 'nope' not found"):
        DesignDataset.load(_jsonl(tmp_path, [_row(reward="nope")]), str(_reward_dir(tmp_path)))
