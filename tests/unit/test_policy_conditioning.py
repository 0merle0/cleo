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

from cleo.design.protein_mpnn_utils.model_utils import ProteinMPNN
from cleo.design.utils.policy import PolicyMPNN


def _cfg(tmp_path, conditioning=None):
    d = {
        "run_name": "t",
        "output_dir": str(tmp_path),
        "model_type": "protein_mpnn",
        "lr": 1e-4,
        "checkpoint_every_n_steps": 100,
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
