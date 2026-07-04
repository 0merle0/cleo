"""Unit tests for the epitope-conditioning module (SPEC 4.1) — focus on the toggles.

All tensor-level (no MPNN weights / coordinates): verifies the master switch makes
every hook an exact no-op (M1 baseline), that each mechanism only touches CDR
positions, that toggles are independent (ablation-ready), and that gradients flow.
"""
import pytest

torch = pytest.importorskip("torch")

from cleo.design.nanobody import (
    ConditioningConfig,
    EpitopeConditioner,
    cdr_segments_from_chain_mask,
    pool_epitope,
)

H, NH, L, M = 16, 2, 10, 5
# CDRs at [2,3,4] (stems 1/5) and [7,8] (stems 6/9)
CHAIN_MASK = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0]], dtype=torch.float32)
CDR_IDX = [2, 3, 4, 7, 8]
FRAME_IDX = [0, 1, 5, 6, 9]


def _cfg(**over):
    base = dict(enabled=True, hidden_dim=H, n_heads=NH,
                node_init_interpolate=False, node_init_relpos=False,
                node_init_pooled_epitope=False, encoder_cross_attn=False,
                decoder_cross_attn=False, coord_free_cdr_edges=False)
    base.update(over)
    return ConditioningConfig(**base)


def _inputs(B=1):
    torch.manual_seed(0)
    h_V = torch.randn(B, L, H)
    epi = torch.randn(B, M, H)
    epi_mask = torch.ones(B, M)
    return h_V, epi, epi_mask


def test_cdr_segments():
    segs = cdr_segments_from_chain_mask(CHAIN_MASK)
    assert segs == [([2, 3, 4], 1, 5), ([7, 8], 6, 9)]


def test_disabled_is_exact_noop():
    cond = EpitopeConditioner(ConditioningConfig(enabled=False, hidden_dim=H))
    h_V, epi, epi_mask = _inputs()
    out = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask)
    assert out is h_V                                   # identity, not just equal
    h_t = torch.randn(3, H)
    assert cond.decoder_cross_attn(h_t, epi, epi_mask) is h_t
    assert list(cond.parameters()) == []               # baseline carries no extra params


def test_node_init_touches_only_cdr():
    cond = EpitopeConditioner(_cfg(node_init_interpolate=True))
    h_V, epi, epi_mask = _inputs()
    out = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask)
    assert out.shape == h_V.shape
    assert torch.equal(out[:, FRAME_IDX], h_V[:, FRAME_IDX])          # framework untouched
    assert not torch.allclose(out[:, CDR_IDX], h_V[:, CDR_IDX])       # CDRs changed


def test_node_init_add_mode_is_residual():
    cond = EpitopeConditioner(_cfg(node_init_relpos=True, node_init_mode="add"))
    h_V, epi, epi_mask = _inputs()
    out = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask)
    assert torch.equal(out[:, FRAME_IDX], h_V[:, FRAME_IDX])
    # add mode: CDR node = original + nonzero contribution
    assert not torch.allclose(out[:, CDR_IDX], h_V[:, CDR_IDX])


def test_encoder_cross_attn_touches_only_cdr():
    cond = EpitopeConditioner(_cfg(encoder_cross_attn=True))   # node_init all off
    h_V, epi, epi_mask = _inputs()
    out = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask)
    assert torch.equal(out[:, FRAME_IDX], h_V[:, FRAME_IDX])
    assert not torch.allclose(out[:, CDR_IDX], h_V[:, CDR_IDX])


def test_pooled_epitope_changes_with_epitope():
    cond = EpitopeConditioner(_cfg(node_init_pooled_epitope=True))
    h_V, epi, epi_mask = _inputs()
    out_a = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask)
    out_b = cond.condition_nodes(h_V, CHAIN_MASK, epi + 5.0, epi_mask)
    assert not torch.allclose(out_a[:, CDR_IDX], out_b[:, CDR_IDX])   # CDR init depends on epitope
    assert torch.equal(out_a[:, FRAME_IDX], out_b[:, FRAME_IDX])


def test_decoder_cross_attn_changes_query_and_broadcasts_batch():
    cond = EpitopeConditioner(_cfg(decoder_cross_attn=True))
    epi = torch.randn(1, M, H)                     # single-example epitope
    h_t = torch.randn(3, H)                        # B_decoder = 3
    out = cond.decoder_cross_attn(h_t, epi, torch.ones(1, M))
    assert out.shape == (3, H)
    assert not torch.allclose(out, h_t)


def test_toggles_are_independent():
    enc_only = EpitopeConditioner(_cfg(encoder_cross_attn=True))
    dec_only = EpitopeConditioner(_cfg(decoder_cross_attn=True))
    assert hasattr(enc_only, "enc_xattn") and not hasattr(enc_only, "dec_xattn")
    assert hasattr(dec_only, "dec_xattn") and not hasattr(dec_only, "enc_xattn")
    # decoder-only must NOT modify nodes at the encoder hook
    h_V, epi, epi_mask = _inputs()
    assert torch.equal(dec_only.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask), h_V)


def test_gradients_flow_to_conditioner():
    cond = EpitopeConditioner(_cfg(node_init_interpolate=True, node_init_relpos=True,
                                   node_init_pooled_epitope=True, encoder_cross_attn=True))
    h_V, epi, epi_mask = _inputs()
    out = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask)
    out.sum().backward()
    grads = [p.grad for p in cond.parameters()]
    assert len(grads) > 0 and all(g is not None for g in grads)
    assert cond.node_init.W_pool.weight.grad.abs().sum() > 0


def test_pool_epitope_respects_mask():
    epi = torch.randn(1, 4, H)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert torch.allclose(pool_epitope(epi, mask), epi[:, :2].mean(dim=1))


def test_config_from_dict_ignores_unknown():
    cfg = ConditioningConfig.from_dict({"enabled": True, "n_heads": 8, "bogus": 1})
    assert cfg.enabled and cfg.n_heads == 8
