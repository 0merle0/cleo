"""Slice-2 step 1 (SPEC 6.8): the policy builds the epitope encoder + conditioner.

Constructs the real PolicyMPNN (loads the pretrained base weights, writes to a tmp output
dir) and checks the object graph both ways:
- conditioning OFF (default): no epitope encoder, a zero-param no-op conditioner, policy still built.
- conditioning ON: a SECOND ProteinMPNN as the epitope encoder (distinct from the policy model,
  init from pretrained base weights) + a conditioner carrying trainable params.

Hooks are wired in later steps; here we only verify construction. Kept out of the tensor-level
suite (test_epitope_conditioning.py) because this one actually loads MPNN weights.
"""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from cleo.design.data.dataset import Example
from cleo.design.protein_mpnn_utils.model_utils import ProteinMPNN
from cleo.design.utils.policy import PolicyMPNN

FIXTURE = str(Path(__file__).resolve().parents[1] / "fixtures" / "mini_complex.pdb")
# chain T = residues 8..17 (NLAFALSELD); pick 3 epitope residues by PDB number
EPI_RESIDUES = [9, 11, 13]           # -> enumeration positions 1, 3, 5


def _epitope_example(epitope_residues=EPI_RESIDUES):
    return Example(
        id="epi0", task="nanobody_design", reward="antibody_interface",
        structure=FIXTURE, design_chain="A",
        params={"epitope_residues": epitope_residues},
    )


def _cfg(tmp_path, conditioning=None):
    d = {
        "run_name": "t",
        "output_dir": str(tmp_path),
        "model_type": "protein_mpnn",
        "lr": 1e-4,
        "checkpoint_every_n_steps": 100,
        "omit_AA": None,
        "batch_size": 1,
        "temperature": 1.0,
    }
    if conditioning is not None:
        d["conditioning"] = conditioning
    return OmegaConf.create(d)


def test_disabled_builds_noop_conditioner(tmp_path):
    policy = PolicyMPNN(_cfg(tmp_path))                 # no conditioning key => disabled
    assert policy.epi_encoder is None
    assert not policy.conditioning_cfg.enabled
    assert list(policy.conditioner.parameters()) == []  # exact no-op, zero extra params
    assert policy.model is not None                     # policy itself still built


def test_explicit_disabled_is_also_noop(tmp_path):
    policy = PolicyMPNN(_cfg(tmp_path, {"enabled": False}))
    assert policy.epi_encoder is None
    assert list(policy.conditioner.parameters()) == []


def test_enabled_builds_epitope_encoder_and_conditioner(tmp_path):
    cfg = _cfg(tmp_path, {"enabled": True, "hidden_dim": 128, "n_heads": 4})
    policy = PolicyMPNN(cfg)

    # a real, SEPARATE second ProteinMPNN for the epitope
    assert isinstance(policy.epi_encoder, ProteinMPNN)
    assert policy.epi_encoder is not policy.model
    epi_ids = {id(p) for p in policy.epi_encoder.parameters()}
    pol_ids = {id(p) for p in policy.model.parameters()}
    assert epi_ids.isdisjoint(pol_ids)                  # independent parameter storage

    # the conditioner carries trainable params and holds the encoder
    assert policy.conditioner.epi_encoder is policy.epi_encoder
    cond_params = list(policy.conditioner.parameters())
    assert len(cond_params) > 0
    assert all(p.device.type == policy.device.type for p in cond_params)


def test_epitope_encoder_inits_from_pretrained_base(tmp_path):
    """Encoder weights come from the vanilla base checkpoint, not random init."""
    from cleo.design.utils.policy import PROTEIN_MPNN_CKPT_PATH

    policy = PolicyMPNN(_cfg(tmp_path, {"enabled": True}))
    base = torch.load(PROTEIN_MPNN_CKPT_PATH, map_location="cpu", weights_only=True)["model_state_dict"]
    got = policy.epi_encoder.state_dict()
    key = "W_e.weight"                                   # an encoder edge-embedding weight
    assert torch.allclose(got[key].cpu(), base[key], atol=0)


# --- step 2: featurize_epitope + mode seam + encode_epitope patch mask ------ #

def test_featurize_epitope_builds_patch_mask(tmp_path):
    policy = PolicyMPNN(_cfg(tmp_path, {"enabled": True}))
    fd = policy.featurize_epitope(_epitope_example())

    M = fd["mask"].shape[1]
    assert M == 10                                       # whole chain T encoded
    assert fd["mask"].sum().item() == 10                 # all antigen residues valid (message passing)
    em = fd["epitope_mask"]
    assert em.shape == (1, 10)
    assert em.sum().item() == 3                          # only the 3 epitope residues
    assert em[0].nonzero().flatten().tolist() == [1, 3, 5]  # PDB 9/11/13 -> positions 1/3/5
    assert fd["S"].shape == (1, 10)                      # known antigen sequence present (path-b)


def test_encode_epitope_returns_patch_mask_not_full(tmp_path):
    policy = PolicyMPNN(_cfg(tmp_path, {"enabled": True}))
    fd = policy.featurize_epitope(_epitope_example())
    epi_per_res, cond_mask = policy.conditioner.encode_epitope(fd)

    assert epi_per_res.shape == (1, 10, 128)             # per-residue over the WHOLE antigen
    assert torch.equal(cond_mask, fd["epitope_mask"])    # returns the patch, not the full mask
    assert cond_mask.sum().item() == 3                   # (full mask would be 10)


def test_featurize_example_complex_mode_is_stage2_seam(tmp_path):
    policy = PolicyMPNN(_cfg(tmp_path, {"enabled": True}))
    with pytest.raises(NotImplementedError):
        policy.featurize_example(_epitope_example(), mode="complex")


def test_featurize_epitope_rejects_positional_indices(tmp_path):
    """Guard: positional-looking indices (0,1,2) don't match chain-T PDB numbers 8..17."""
    policy = PolicyMPNN(_cfg(tmp_path, {"enabled": True}))
    with pytest.raises(ValueError, match="epitope_residues"):
        policy.featurize_epitope(_epitope_example(epitope_residues=[0, 1, 2]))


# --- step 3: coord_free_cdr_edges flag threaded into the policy model ------- #

def test_coord_free_flag_threaded_into_policy_model(tmp_path):
    p = PolicyMPNN(_cfg(tmp_path, {"enabled": True, "coord_free_cdr_edges": True}))
    assert p.model.features.coord_free_cdr_edges is True
    assert hasattr(p.model.features, "unknown_rbf")
    assert p.epi_encoder.features.coord_free_cdr_edges is False  # epitope has no CDRs -> no surgery


def test_coord_free_independent_toggle_off(tmp_path):
    """Master enabled but the mechanism toggle off => no surgery on the policy model."""
    p = PolicyMPNN(_cfg(tmp_path, {"enabled": True, "coord_free_cdr_edges": False}))
    assert p.model.features.coord_free_cdr_edges is False
    assert not hasattr(p.model.features, "unknown_rbf")


def test_baseline_policy_model_has_no_surgery(tmp_path):
    p = PolicyMPNN(_cfg(tmp_path))                              # conditioning disabled
    assert p.model.features.coord_free_cdr_edges is False
    assert not hasattr(p.model.features, "unknown_rbf")


# --- step 3.5: gapped-framework featurize + two-file loading ---------------- #

def _gapped_example():
    """Chain A native residues 4,5,6 (positional span [4,7)) excised -> 6 gap nodes decoded in."""
    return Example(
        id="gap0", task="nanobody_design", reward="antibody_interface",
        structure=FIXTURE, design_chain="A",
        params={"cdr_spans": {"H1": [4, 7]}, "cdr_lengths": {"H1": 6},
                "epitope_residues": EPI_RESIDUES},
    )


def test_featurize_example_gaps_cdrs_when_coord_free_on(tmp_path):
    p = PolicyMPNN(_cfg(tmp_path, {"enabled": True, "coord_free_cdr_edges": True}))
    fd = p.featurize_example(_gapped_example())

    # mini_complex = chain A (15) + chain T (10); gapping A: 15 - 3 native + 6 gap = 18 (+10) = 28
    assert fd["X"].shape[1] == 28
    assert fd["chain_mask"].sum().item() == 6                   # exactly the 6 gap nodes designable
    assert torch.isfinite(fd["X"]).all()                       # gap backbone copied from stem


class _StubDataset:
    """Minimal stand-in for the fixed-residues resolver (no JSONL/reward machinery needed)."""

    def fixed_residues(self, example, cl, ri, ic):
        from cleo.design.data.mask import resolve_fixed_residues
        return resolve_fixed_residues(cl, ri, ic, example.design_chain, example.cdr_spans)


def test_featurize_example_no_gap_when_coord_free_off(tmp_path):
    """Master enabled but surgery off => the stock fixed-residues path (native lengths kept)."""
    p = PolicyMPNN(_cfg(tmp_path, {"enabled": True, "coord_free_cdr_edges": False}))
    p.dataset = _StubDataset()
    fd = p.featurize_example(_gapped_example())
    assert fd["X"].shape[1] == 25                               # 15 + 10, nothing spliced
    assert fd["chain_mask"].sum().item() == 3                   # native CDR positions 4,5,6


def test_epitope_source_prefers_separate_antigen_file(tmp_path):
    """Two-file loading: epitope comes from antigen_structure when set, else falls back."""
    combined = _epitope_example()
    assert combined.epitope_source == FIXTURE                   # fallback (single-file fixture)

    split = Example(
        id="split0", task="nanobody_design", reward="antibody_interface",
        structure="/some/framework.pdb", design_chain="A",
        antigen_structure=FIXTURE, params={"epitope_residues": EPI_RESIDUES},
    )
    assert split.epitope_source == FIXTURE                      # separate antigen file wins

    policy = PolicyMPNN(_cfg(tmp_path, {"enabled": True}))
    fd = policy.featurize_epitope(split)                        # reads antigen_structure, not structure
    assert fd["epitope_mask"].sum().item() == 3
