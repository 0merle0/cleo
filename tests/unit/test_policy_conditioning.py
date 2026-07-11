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

import math
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


# --- step 4: the 3 conditioning hooks wired into encode + rollout ----------- #

def test_attach_epitope_populates_embeddings(tmp_path):
    p = PolicyMPNN(_cfg(tmp_path, {"enabled": True, "coord_free_cdr_edges": True}))
    ex = _gapped_example()
    fd = p.attach_epitope(ex, p.featurize_example(ex))
    assert fd["epi_per_res"].shape == (1, 10, 128)             # per-residue over whole chain T
    assert fd["epi_mask"].sum().item() == 3                    # the epitope patch, not all 10


def test_attach_epitope_noop_when_disabled(tmp_path):
    p = PolicyMPNN(_cfg(tmp_path))                             # conditioning off
    p.dataset = _StubDataset()
    ex = _gapped_example()
    fd = p.attach_epitope(ex, p.featurize_example(ex))
    assert "epi_per_res" not in fd and "epi_mask" not in fd


def test_encode_initial_state_conditions_only_cdr_nodes(tmp_path):
    p = PolicyMPNN(_cfg(tmp_path, {"enabled": True, "coord_free_cdr_edges": True}))
    p.model.eval()                                            # deterministic encode (dropout off)
    ex = _gapped_example()
    fd = p.attach_epitope(ex, p.featurize_example(ex))

    with torch.no_grad():
        h_plain, _, _ = p.model.encode(fd)                   # un-conditioned reference
    h_cond, _, _ = p.encode_initial_state(fd)                # applies condition_nodes

    cdr = fd["chain_mask"][0] > 0.5                           # the 6 gap nodes
    assert torch.allclose(h_cond[0][~cdr], h_plain[0][~cdr], atol=1e-6)   # framework untouched
    assert not torch.allclose(h_cond[0][cdr], h_plain[0][cdr])            # CDR nodes conditioned


def test_encode_initial_state_identity_when_disabled(tmp_path):
    """M1 baseline preserved: no epitope embeddings, encode == plain model.encode."""
    p = PolicyMPNN(_cfg(tmp_path))
    p.model.eval()
    p.dataset = _StubDataset()
    ex = _gapped_example()
    fd = p.featurize_example(ex)
    with torch.no_grad():
        h_plain, _, _ = p.model.encode(fd)
    h_cond, _, _ = p.encode_initial_state(fd)
    assert torch.allclose(h_cond, h_plain, atol=1e-6)


def test_conditioner_params_are_optimized_when_enabled(tmp_path):
    p = PolicyMPNN(_cfg(tmp_path, {"enabled": True}))
    opt_ids = {id(pp) for g in p.optimizer.param_groups for pp in g["params"]}
    cond_ids = {id(pp) for pp in p.conditioner.parameters()}
    assert cond_ids                                           # the conditioner has trainable params
    assert cond_ids <= opt_ids                                # all of them are in the optimizer


def test_rollout_runs_end_to_end_with_conditioning(tmp_path):
    p = PolicyMPNN(_cfg(tmp_path, {"enabled": True, "coord_free_cdr_edges": True}))
    p.model.eval()
    ex = _gapped_example()
    fd = p.attach_epitope(ex, p.featurize_example(ex))       # decoder cross-attn active
    h_V, h_E, E_idx = p.encode_initial_state(fd)
    out = p.rollout(fd, h_V, h_E, E_idx)

    L = fd["X"].shape[1]
    assert out["S"].shape == (1, L)                          # full graph (gapped chain A + chain T)
    assert torch.isfinite(out["log_probs"]).all()
    assert out["chain_mask"].sum().item() == 6               # only the 6 gap nodes designable


# --- step 5 wiring: featurize_example stashes per-segment CDR-type ids -------- #

def test_featurize_example_stashes_cdr_ids(tmp_path):
    p = PolicyMPNN(_cfg(tmp_path, {"enabled": True, "coord_free_cdr_edges": True}))
    fd = p.featurize_example(_gapped_example())              # one CDR: H1 -> type id 0
    assert fd["cdr_ids"] == [0]


# --- step 6: unfreeze framework + epitope encoders, grad-tracked re-encode ---- #

def _step6_cfg(tmp_path):
    cfg = _cfg(tmp_path, {"enabled": True, "coord_free_cdr_edges": True})
    return OmegaConf.merge(cfg, {"train_framework_encoder": True})


def test_step4_default_freezes_framework_encoder(tmp_path):
    p = PolicyMPNN(_cfg(tmp_path, {"enabled": True}))        # train_framework_encoder defaults False
    assert not p.train_framework_encoder
    frozen = [p_ for n, p_ in p.model.named_parameters()
              if "decoder_layers" not in n and "W_out" not in n]
    assert frozen and not any(p_.requires_grad for p_ in frozen)   # encoder stays frozen


def test_step6_gated_on_conditioning_enabled(tmp_path):
    cfg = OmegaConf.merge(_cfg(tmp_path), {"train_framework_encoder": True})  # no conditioning
    p = PolicyMPNN(cfg)
    assert not p.train_framework_encoder                    # nothing to train => no-op


def test_step6_unfreezes_all_and_optimizer_covers_both_encoders(tmp_path):
    p = PolicyMPNN(_step6_cfg(tmp_path))
    assert p.train_framework_encoder
    assert all(pp.requires_grad for pp in p.model.parameters())    # framework all-trainable
    opt_ids = {id(pp) for g in p.optimizer.param_groups for pp in g["params"]}
    enc_ids = {id(pp) for n, pp in p.model.named_parameters() if "encoder_layers" in n}
    epi_ids = {id(pp) for pp in p.epi_encoder.parameters()}
    assert enc_ids and enc_ids <= opt_ids                   # framework encoder optimized
    assert epi_ids and epi_ids <= opt_ids                   # epitope encoder optimized


def test_step6_attach_epitope_defers_encoding(tmp_path):
    """Step 6 stashes only the epitope feature dict; embeddings are (re)computed grad-tracked
    in encode_initial_state, not cached under no_grad here."""
    p = PolicyMPNN(_step6_cfg(tmp_path))
    ex = _gapped_example()
    fd = p.attach_epitope(ex, p.featurize_example(ex))
    assert "epi_fd" in fd and "epi_per_res" not in fd


def test_step6_grad_reaches_framework_and_epitope_encoders(tmp_path):
    p = PolicyMPNN(_step6_cfg(tmp_path))
    p.model.eval()
    ex = _gapped_example()
    fd = p.attach_epitope(ex, p.featurize_example(ex))
    h_V, h_E, E_idx = p.encode_initial_state(fd, grad=True)
    assert h_V.requires_grad                                 # grad-tracked (not a detached leaf)
    assert fd["epi_per_res"].requires_grad                   # epitope embeddings recomputed grad-tracked
    h_V.sum().backward()
    enc_grad = [pp for n, pp in p.model.named_parameters() if "encoder_layers" in n and pp.grad is not None]
    epi_grad = [pp for pp in p.epi_encoder.parameters() if pp.grad is not None]
    assert enc_grad                                          # framework encoder received gradient
    assert epi_grad                                          # epitope encoder received gradient


def test_epitope_log_fields_reports_size_and_charge(tmp_path):
    p = PolicyMPNN(_cfg(tmp_path))
    ex = Example(
        id="e", task="nanobody_design", reward="antibody_interface",
        structure=FIXTURE, design_chain="A",
        params={"epitope_residues": EPI_RESIDUES, "epitope_net_charge": -2},
    )
    fields = p._epitope_log_fields(ex)
    assert fields["epitope_size"] == float(len(EPI_RESIDUES))  # 3
    assert fields["epitope_net_charge"] == -2.0
    assert all(isinstance(v, float) for v in fields.values())   # log_metrics only writes floats


def test_epitope_log_fields_nan_charge_when_missing(tmp_path):
    p = PolicyMPNN(_cfg(tmp_path))
    ex = _epitope_example()                                      # params has no epitope_net_charge
    fields = p._epitope_log_fields(ex)
    assert fields["epitope_size"] == 3.0
    assert math.isnan(fields["epitope_net_charge"])             # stable column, NaN when unknown
