"""Unit tests for the epitope-conditioning module (SPEC 4.1) — focus on the toggles.

All tensor-level (no MPNN weights / coordinates): verifies the master switch makes
every hook an exact no-op (M1 baseline), that each mechanism only touches CDR
positions, that toggles are independent (ablation-ready), and that gradients flow.
"""
import pytest

torch = pytest.importorskip("torch")

from cleo.design.data.epitope import (
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


# --- step 5: CDR-identity embedding, stem-gap geometry, attention-pool ------ #

from cleo.design.data.epitope import (   # noqa: E402
    AttentionPool,
    EpitopeConditioningError,
    ordered_cdr_type_ids,
)


# --- PR#27: routing is strict, per-CDR position encoding, explicit epitope mask ---- #


def test_ordered_cdr_type_ids_raises_on_unroutable():
    # an L* CDR on a single-chain VHH is a heavy/light mismatch, not a silently dropped CDR
    with pytest.raises(EpitopeConditioningError):
        ordered_cdr_type_ids("H", {"H1": [25, 32], "L1": [23, 39]})
    # a key that is neither heavy nor light
    with pytest.raises(EpitopeConditioningError):
        ordered_cdr_type_ids("H", {"X1": [1, 5]})


def test_relpos_per_cdr_writes_cdr_and_depends_on_type():
    cond = EpitopeConditioner(_cfg(node_init_relpos_per_cdr=True))
    h_V, epi, epi_mask = _inputs()
    out_a = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask, cdr_ids=[0, 1])
    out_b = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask, cdr_ids=[2, 3])
    assert torch.equal(out_a[:, FRAME_IDX], h_V[:, FRAME_IDX])         # framework untouched
    assert not torch.allclose(out_a[:, CDR_IDX], h_V[:, CDR_IDX])      # CDRs written
    assert not torch.allclose(out_a[:, CDR_IDX], out_b[:, CDR_IDX])    # different types => different pos enc


def test_resolve_cond_mask_requires_explicit_whole_chain():
    mask = torch.ones(1, 4)
    patch = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # epitope_mask present -> returned as-is
    cond = EpitopeConditioner(_cfg())
    assert torch.equal(cond._resolve_cond_mask({"epitope_mask": patch}, mask), patch)
    # no epitope_mask and no opt-in -> error (no silent whole-chain fallback)
    with pytest.raises(EpitopeConditioningError):
        cond._resolve_cond_mask({}, mask)
    # explicit opt-in -> whole-antigen mask
    cond_whole = EpitopeConditioner(_cfg(allow_whole_epitope=True))
    assert torch.equal(cond_whole._resolve_cond_mask({}, mask), mask)


def test_ordered_cdr_type_ids_vhh_and_fv():
    # VHH: single chain H, sorted by span start -> H1,H2,H3 ids [0,1,2]
    vhh = ordered_cdr_type_ids("H", {"H2": [51, 57], "H1": [25, 32], "H3": [98, 109]})
    assert vhh == [0, 1, 2]
    # Fv: H chain then L chain, each sorted by start -> H1,H2,H3,L1,L2,L3 ids [0..5]
    fv = ordered_cdr_type_ids(
        "H L",
        {"L1": [23, 39], "H1": [25, 32], "H3": [98, 109], "L3": [93, 102],
         "H2": [51, 57], "L2": [54, 61]},
    )
    assert fv == [0, 1, 2, 3, 4, 5]


def test_cdr_identity_only_cdr_and_depends_on_id():
    cond = EpitopeConditioner(_cfg(node_init_cdr_identity=True))
    h_V, epi, epi_mask = _inputs()
    segs = cdr_segments_from_chain_mask(CHAIN_MASK)          # two segments
    out_a = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask, cdr_ids=[0, 1])
    out_b = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask, cdr_ids=[2, 3])
    assert len(segs) == 2
    assert torch.equal(out_a[:, FRAME_IDX], h_V[:, FRAME_IDX])         # framework untouched
    assert not torch.allclose(out_a[:, CDR_IDX], h_V[:, CDR_IDX])      # CDRs changed
    assert not torch.allclose(out_a[:, CDR_IDX], out_b[:, CDR_IDX])    # different CDR ids => different init


def test_cdr_identity_missing_ids_no_error_and_omits_signal():
    """Missing / mismatched cdr_ids must not error; they simply omit the identity term (with
    another node-init mechanism on, the node is still computed)."""
    cond = EpitopeConditioner(_cfg(node_init_interpolate=True, node_init_cdr_identity=True))
    h_V, epi, epi_mask = _inputs()
    out_noids = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask, cdr_ids=None)
    out_ids = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask, cdr_ids=[0, 1])
    assert out_noids.shape == h_V.shape                               # no error
    assert not torch.allclose(out_noids[:, CDR_IDX], out_ids[:, CDR_IDX])  # ids add a signal
    out_bad = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask, cdr_ids=[0])  # wrong length
    assert torch.allclose(out_bad[:, CDR_IDX], out_noids[:, CDR_IDX])  # treated like None


def test_stem_geom_only_cdr_and_depends_on_coords():
    cond = EpitopeConditioner(_cfg(node_init_stem_geom=True))
    h_V, epi, epi_mask = _inputs()
    # segment 1 stems are nodes 1 (N) and 5 (C); vary only the C stem so the span distance changes.
    X_a = torch.zeros(1, L, 4, 3)
    X_a[:, 5, 1, 0] = 3.0                                  # span dist(seg1) = 3
    X_b = torch.zeros(1, L, 4, 3)
    X_b[:, 5, 1, 0] = 9.0                                  # span dist(seg1) = 9
    out_a = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask, X=X_a)
    out_b = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask, X=X_b)
    assert torch.equal(out_a[:, FRAME_IDX], h_V[:, FRAME_IDX])        # framework untouched
    assert not torch.allclose(out_a[:, CDR_IDX], h_V[:, CDR_IDX])     # geometry writes CDR nodes
    assert not torch.allclose(out_a[:, [2, 3, 4]], out_b[:, [2, 3, 4]])  # seg1 depends on stem coords


def test_attention_pool_differs_from_mean_and_is_used():
    cfg = _cfg(node_init_pooled_epitope=True, attention_pool=True)
    cond = EpitopeConditioner(cfg)
    assert hasattr(cond, "attn_pool")
    epi = torch.randn(1, M, H)
    mask = torch.ones(1, M)
    pooled_attn = cond._pool(epi, mask)
    assert pooled_attn.shape == (1, H)
    assert not torch.allclose(pooled_attn, pool_epitope(epi, mask))    # attention != masked mean


def test_attention_pool_module_masks_padding():
    ap = AttentionPool(H, NH)
    epi = torch.randn(1, 4, H)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    out = ap(epi, mask)
    assert out.shape == (1, H) and torch.isfinite(out).all()


# --- step 5b: CDR self-attn <-> CDR-epitope coupler (stacked, cross-chain) --------- #


def test_coupler_absent_when_disabled_present_when_on():
    assert not hasattr(EpitopeConditioner(_cfg()), "coupler")           # off by default
    cond = EpitopeConditioner(_cfg(cdr_epitope_coupler=True, coupler_rounds=2))
    assert hasattr(cond, "coupler") and cond.coupler.rounds == 2


def test_coupler_only_touches_cdr_and_depends_on_epitope():
    cond = EpitopeConditioner(_cfg(cdr_epitope_coupler=True))
    h_V, epi, epi_mask = _inputs()
    out_a = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask)
    out_b = cond.condition_nodes(h_V, CHAIN_MASK, epi + 5.0, epi_mask)
    assert torch.equal(out_a[:, FRAME_IDX], h_V[:, FRAME_IDX])          # framework untouched
    assert not torch.allclose(out_a[:, CDR_IDX], h_V[:, CDR_IDX])       # CDRs updated
    assert not torch.allclose(out_a[:, CDR_IDX], out_b[:, CDR_IDX])     # CDRs depend on epitope


def test_coupler_couples_cdr_segments_across_the_set():
    """Self-attention over the gathered CDR set means one segment's nodes influence another's —
    this is the cross-chain organization (both segments are in one attention set)."""
    cond = EpitopeConditioner(_cfg(cdr_epitope_coupler=True))
    h_V, epi, epi_mask = _inputs()
    out1 = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask)
    h_V2 = h_V.clone()
    h_V2[:, 7] = h_V2[:, 7] + 5.0                                       # perturb a node in segment 2
    out2 = cond.condition_nodes(h_V2, CHAIN_MASK, epi, epi_mask)
    assert not torch.allclose(out1[:, [2, 3, 4]], out2[:, [2, 3, 4]])   # segment-1 output shifts too


def test_coupler_stacks_after_encoder_cross_attn():
    base = EpitopeConditioner(_cfg(encoder_cross_attn=True))
    stacked = EpitopeConditioner(_cfg(encoder_cross_attn=True, cdr_epitope_coupler=True))
    stacked.enc_xattn.load_state_dict(base.enc_xattn.state_dict())      # isolate the coupler's effect
    h_V, epi, epi_mask = _inputs()
    out_base = base.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask)
    out_stacked = stacked.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask)
    assert torch.equal(out_base[:, FRAME_IDX], out_stacked[:, FRAME_IDX])
    assert not torch.allclose(out_base[:, CDR_IDX], out_stacked[:, CDR_IDX])
    out_stacked.sum().backward()
    grads = [p.grad for p in stacked.coupler.parameters() if p.grad is not None]
    assert len(grads) > 0 and any(g.abs().sum() > 0 for g in grads)


def test_step5_gradients_flow():
    cond = EpitopeConditioner(_cfg(node_init_cdr_identity=True, node_init_stem_geom=True,
                                   node_init_pooled_epitope=True, attention_pool=True))
    h_V, epi, epi_mask = _inputs()
    X = torch.randn(1, L, 4, 3)
    out = cond.condition_nodes(h_V, CHAIN_MASK, epi, epi_mask, cdr_ids=[0, 1], X=X)
    out.sum().backward()
    assert cond.node_init.cdr_emb.weight.grad.abs().sum() > 0
    assert cond.node_init.stem_geom.weight.grad.abs().sum() > 0
    assert cond.attn_pool.query.grad.abs().sum() > 0
