"""Epitope-conditioning module for the dual-encoder ProteinMPNN policy (SPEC 4.1).

Everything here is gated so ablations are one config flag each. With
``ConditioningConfig.enabled = False`` every hook is an exact no-op and the policy
is byte-identical to stock ProteinMPNN (the M1 multi-target baseline). Each
mechanism (seq injection, interpolated node-init, rel-pos, pooled epitope, encoder
cross-attn, decoder cross-attn, coordinate-free CDR edges) has its own independent
toggle so we can run encoder-only / decoder-only / pooled-only / none, etc.

Pieces (SPEC 4.1):
- ``EpitopeConditioner.encode_epitope`` — a SEPARATE ProteinMPNN encoder (init from
  the same pretrained weights, independently trainable) run over the epitope with the
  **known sequence injected into the node features before the message-passing layers**
  (path b), so the encoder mixes sequence + structure and every per-residue embedding
  carries both.
- ``condition_nodes`` (post-encode, framework side) — replaces/refines each masked CDR
  node with an interpolated flanking-stem anchor (N->C sweep, weight (i+1)/(L+1)) +
  a rel-pos embedding of ``(i, L)`` + the pooled epitope embedding, then optional
  per-residue **encoder cross-attention** (CDR nodes attend to individual epitope
  residues). Only CDR positions are touched; framework nodes pass through unchanged.
- ``decoder_cross_attn`` (per decode step) — the per-step decoding hidden state attends
  to individual epitope residues; the caller gates it to CDR steps.

Shapes: ``h_V`` node embeddings ``[B, L, H]``; epitope per-residue ``[B, M, H]`` with
``epi_mask`` ``[B, M]`` (1 = real residue). CDR positions are ``chain_mask == 1``
(designable), as built by ``featurize`` / the data-layer mask.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn


@dataclass
class ConditioningConfig:
    """Ablation surface: a master switch + one flag per mechanism (SPEC 8.1 ablations)."""

    enabled: bool = False                 # master OFF => stock MPNN, all hooks no-op (M1 baseline)
    inject_epitope_sequence: bool = True  # path (b): W_s(seq) into epitope node feats pre-message-passing
    node_init_interpolate: bool = True    # flanking-stem anchor interpolation for CDR nodes
    node_init_relpos: bool = True         # (i, L) relative-position embedding on CDR nodes
    node_init_pooled_epitope: bool = True # add pooled epitope embedding to CDR nodes
    node_init_cdr_identity: bool = False  # (step 5) which-CDR embedding (H1/H2/H3, L1-L3) on CDR nodes
    node_init_relpos_per_cdr: bool = False  # (step 5) per-CDR-type position embedding (unique table per H1..L3)
    node_init_stem_geom: bool = False     # (step 5) flanking-stem gap geometry (span dist, dist/len) on CDR nodes
    attention_pool: bool = False          # (step 5) pool the epitope by learned-query attention (else masked mean)
    encoder_cross_attn: bool = True       # per-residue epitope cross-attn at the encoder (CDR nodes)
    decoder_cross_attn: bool = True       # per-residue epitope cross-attn at each decode step
    coord_free_cdr_edges: bool = True     # Option B: coordinate-free CDR graph edges (ProteinFeatures surgery)
    node_init_mode: str = "replace"       # "replace" | "add" at CDR positions
    allow_whole_epitope: bool = False     # explicit opt-in to condition on the whole antigen when no epitope_mask
    hidden_dim: int = 128
    n_heads: int = 4
    max_cdr_len: int = 32                 # rel-pos table size (H3 ~24)

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return cls()
        if isinstance(d, cls):
            return d
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in known})

    @property
    def uses_node_init(self) -> bool:
        return (
            self.node_init_interpolate
            or self.node_init_relpos
            or self.node_init_relpos_per_cdr
            or self.node_init_pooled_epitope
            or self.node_init_cdr_identity
            or self.node_init_stem_geom
        )


class EpitopeConditioningError(ValueError):
    """Raised on malformed conditioning inputs (unroutable CDRs, missing epitope mask)."""


#: canonical CDR-type vocabulary (heavy then light) for the identity embedding (step 5).
CDR_TYPE_IDS = {"H1": 0, "H2": 1, "H3": 2, "L1": 3, "L2": 4, "L3": 5}


def ordered_cdr_type_ids(design_chain, cdr_spans) -> list[int]:
    """CDR-type ids (into ``CDR_TYPE_IDS``) in structure/position order, aligned to the
    segments ``cdr_segments_from_chain_mask`` returns.

    Routing mirrors ``mask._route_spans`` — **positional**, not by literal chain letter: ``H*``
    CDRs belong to the first design chain, ``L*`` to the second (a VHH may name its chain ``"A"``
    while its CDRs are still ``H1/H2/H3``). Segments appear in position order: the first design
    chain before the second, and within a chain sorted by span start — exactly the order the gapped
    structure lays them out. Returns one id per CDR so ``CDRNodeInit`` stamps the right identity.
    """
    from cleo.design.data.mask import _cdr_key, as_chain_list

    n_chains = len(as_chain_list(design_chain))
    per_chain: dict[int, list] = {i: [] for i in range(n_chains)}
    for k, v in cdr_spans.items():
        key = _cdr_key(k)
        idx = 0 if key.startswith("H") else (1 if key.startswith("L") else None)
        # Every framework is heavy (VHH) or heavy+light (Fv); anything else is a data error, not a
        # silently-dropped CDR. An L* CDR on a single-chain VHH means the scaffold/spec disagree.
        if idx is None:
            raise EpitopeConditioningError(f"CDR key {key!r} is neither heavy (H*) nor light (L*)")
        if idx >= n_chains:
            raise EpitopeConditioningError(
                f"CDR {key!r} routes to design chain {idx} but the framework has only "
                f"{n_chains} chain(s) — heavy/light mismatch between scaffold and cdr_spans")
        per_chain[idx].append((int(v[0]), key))

    ids: list[int] = []
    for i in range(n_chains):
        for _start, key in sorted(per_chain[i]):
            ids.append(CDR_TYPE_IDS[key])
    return ids


def pool_epitope(epi_per_res, epi_mask=None):
    """Masked mean over epitope residues -> ``[B, H]``."""
    if epi_mask is None:
        return epi_per_res.mean(dim=1)
    w = epi_mask.to(epi_per_res.dtype).unsqueeze(-1)          # [B, M, 1]
    return (epi_per_res * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)


class AttentionPool(nn.Module):
    """Pool epitope residues into ``[B, H]`` via a single learned query (step 5).

    A content-based alternative to the masked mean: a learned query attends over the epitope
    residues, so the pooled vector can weight the residues that matter for the paratope instead
    of averaging uniformly.
    """

    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

    def forward(self, epi_per_res, epi_mask=None):
        B = epi_per_res.shape[0]
        q = self.query.expand(B, -1, -1)
        kpm = None if epi_mask is None else (epi_mask <= 0.5)
        out, _ = self.attn(q, epi_per_res, epi_per_res, key_padding_mask=kpm, need_weights=False)
        return out[:, 0]                                       # [B, H]


def cdr_segments_from_chain_mask(chain_mask):
    """Contiguous runs of designable (``==1``) positions -> list of (positions, stem_n, stem_c).

    ``stem_n`` / ``stem_c`` are the flanking framework indices (or ``None`` at a chain
    boundary). A masked antibody framework keeps CDRs interior, so both stems exist.
    """
    if chain_mask.dim() > 1:
        chain_mask = chain_mask[0]
    m = (chain_mask > 0.5).tolist()
    L = len(m)
    segments = []
    i = 0
    while i < L:
        if m[i]:
            a = i
            while i < L and m[i]:
                i += 1
            b = i - 1
            stem_n = a - 1 if a - 1 >= 0 else None
            stem_c = b + 1 if b + 1 < L else None
            segments.append((list(range(a, b + 1)), stem_n, stem_c))
        else:
            i += 1
    return segments


class PerResidueCrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to per-residue epitope keys/values.

    Returns the (layer-normed) attention delta; the caller adds it as a gated residual
    so only the intended positions (CDR nodes / CDR decode steps) are updated.
    """

    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, epi_per_res, epi_mask=None):
        # key_padding_mask: True = ignore
        kpm = None if epi_mask is None else (epi_mask <= 0.5)
        out, _ = self.attn(query, epi_per_res, epi_per_res, key_padding_mask=kpm, need_weights=False)
        return self.norm(out)


class CDRNodeInit(nn.Module):
    """Build CDR node embeddings: interpolated flanking-stem anchors + rel-pos + pooled epitope."""

    def __init__(self, cfg: ConditioningConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim
        if cfg.node_init_relpos:
            self.pos_emb = nn.Embedding(cfg.max_cdr_len, H)
            self.len_emb = nn.Embedding(cfg.max_cdr_len + 1, H)
        if cfg.node_init_relpos_per_cdr:
            # a unique position table per CDR type (H1..L3): the within-CDR position embedding is
            # CDR-specific, so H3 position 4 differs from L1 position 4 (SPEC 4.1 rel-pos comment).
            self.pos_emb_per_cdr = nn.Embedding(len(CDR_TYPE_IDS) * cfg.max_cdr_len, H)
        if cfg.node_init_pooled_epitope:
            self.W_pool = nn.Linear(H, H)
        if cfg.node_init_cdr_identity:
            self.cdr_emb = nn.Embedding(len(CDR_TYPE_IDS), H)
        if cfg.node_init_stem_geom:
            self.stem_geom = nn.Linear(3, H)   # [span dist, dist/(len+1), len/max_cdr_len] -> H

    def forward(self, h_V, segments, pooled=None, cdr_ids=None, X=None):
        cfg = self.cfg
        B, L, H = h_V.shape
        device = h_V.device
        init_full = torch.zeros(B, L, H, device=device, dtype=h_V.dtype)
        cdr_mask = torch.zeros(L, dtype=torch.bool, device=device)

        have_ids = cdr_ids is not None and len(cdr_ids) == len(segments)
        use_cdr_id = cfg.node_init_cdr_identity and have_ids
        use_per_cdr_pos = cfg.node_init_relpos_per_cdr and have_ids
        ca = X[..., 1, :] if (cfg.node_init_stem_geom and X is not None) else None  # Cα coords [B, L, 3]

        for seg_i, (positions, stem_n, stem_c) in enumerate(segments):
            Lr = len(positions)
            n_idx = stem_n if stem_n is not None else stem_c
            c_idx = stem_c if stem_c is not None else stem_n

            # per-segment CDR-identity + stem-gap-geometry vectors (constant across the segment)
            cdr_vec = self.cdr_emb.weight[cdr_ids[seg_i]] if use_cdr_id else None
            geom_vec = None
            if ca is not None and stem_n is not None and stem_c is not None:
                d = torch.linalg.norm(ca[:, stem_c] - ca[:, stem_n], dim=-1)          # [B]
                feats = torch.stack(
                    [d, d / (Lr + 1), torch.full_like(d, Lr / cfg.max_cdr_len)], dim=-1
                )                                                                       # [B, 3]
                geom_vec = self.stem_geom(feats)                                        # [B, H]

            for k, p in enumerate(positions):
                cdr_mask[p] = True
                contrib = torch.zeros(B, H, device=device, dtype=h_V.dtype)
                if cfg.node_init_interpolate and n_idx is not None:
                    w = (k + 1) / (Lr + 1)
                    contrib = contrib + (1.0 - w) * h_V[:, n_idx] + w * h_V[:, c_idx]
                if cfg.node_init_relpos:
                    ki = min(k, cfg.max_cdr_len - 1)
                    li = min(Lr, cfg.max_cdr_len)
                    contrib = contrib + self.pos_emb.weight[ki] + self.len_emb.weight[li]
                if use_per_cdr_pos:
                    ki = min(k, cfg.max_cdr_len - 1)
                    contrib = contrib + self.pos_emb_per_cdr.weight[cdr_ids[seg_i] * cfg.max_cdr_len + ki]
                if cfg.node_init_pooled_epitope and pooled is not None:
                    contrib = contrib + self.W_pool(pooled)
                if cdr_vec is not None:
                    contrib = contrib + cdr_vec
                if geom_vec is not None:
                    contrib = contrib + geom_vec
                init_full[:, p] = contrib

        gate = cdr_mask.view(1, L, 1)
        if cfg.node_init_mode == "replace":
            return torch.where(gate, init_full, h_V)
        return h_V + gate.to(h_V.dtype) * init_full


class EpitopeConditioner(nn.Module):
    """Orchestrates epitope conditioning across the 3 integration points (SPEC 4.1).

    ``epi_encoder`` is a separate ProteinMPNN whose encoder produces the epitope
    embeddings (init from pretrained weights, independently trainable). It may be None
    when only the tensor-level hooks are exercised (unit tests / when embeddings are
    supplied directly).
    """

    def __init__(self, cfg: ConditioningConfig, epi_encoder=None):
        super().__init__()
        self.cfg = cfg = ConditioningConfig.from_dict(cfg)
        self.epi_encoder = epi_encoder
        if not cfg.enabled:
            return
        if cfg.uses_node_init:
            self.node_init = CDRNodeInit(cfg)
        if cfg.attention_pool and cfg.node_init_pooled_epitope:
            self.attn_pool = AttentionPool(cfg.hidden_dim, cfg.n_heads)
        if cfg.encoder_cross_attn:
            self.enc_xattn = PerResidueCrossAttention(cfg.hidden_dim, cfg.n_heads)
        if cfg.decoder_cross_attn:
            self.dec_xattn = PerResidueCrossAttention(cfg.hidden_dim, cfg.n_heads)

    def _pool(self, epi_per_res, epi_mask=None):
        """Pool the epitope to ``[B, H]`` — learned-query attention if configured, else masked mean."""
        if hasattr(self, "attn_pool"):
            return self.attn_pool(epi_per_res, epi_mask)
        return pool_epitope(epi_per_res, epi_mask)

    # -- epitope encoding (path b: seq-injected message passing) ------------- #
    def encode_epitope(self, epi_feature_dict):
        """Run the separate MPNN encoder over the WHOLE antigen with the known sequence injected.

        Returns ``(epi_per_res [B, M, H], cond_mask [B, M])``. The sequence is added to the node
        features BEFORE the encoder layers so message passing over the structural edges mixes
        sequence + structure. Message passing uses ``mask`` (all valid antigen residues → full
        structural context), but the returned ``cond_mask`` is the ``epitope_mask`` **patch** so
        that ``pool_epitope`` + cross-attention condition on the epitope residues specifically,
        not the whole antigen (SPEC 4.1, 2026-07-04). An ``epitope_mask`` is required; to condition
        on the whole antigen chain instead, opt in explicitly with ``allow_whole_epitope=true``.
        """
        from cleo.design.protein_mpnn_utils.model_utils import gather_nodes

        m = self.epi_encoder
        E, E_idx = m.features(epi_feature_dict)
        h_E = m.W_e(E)
        mask = epi_feature_dict["mask"]
        if self.cfg.inject_epitope_sequence:
            h_V = m.W_s(epi_feature_dict["S"])
        else:
            h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in m.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        return h_V, self._resolve_cond_mask(epi_feature_dict, mask)

    def _resolve_cond_mask(self, epi_feature_dict, mask):
        """The conditioning mask: the ``epitope_mask`` patch if present, else the whole-antigen
        ``mask`` **only** when ``allow_whole_epitope`` is explicitly set — otherwise error rather
        than silently conditioning on the whole chain (SPEC 4.1)."""
        if epi_feature_dict.get("epitope_mask") is not None:
            return epi_feature_dict["epitope_mask"]
        if self.cfg.allow_whole_epitope:
            return mask
        raise EpitopeConditioningError(
            "no 'epitope_mask' supplied to encode_epitope; set "
            "conditioning.allow_whole_epitope=true to condition on the whole antigen chain")

    # -- hook 1+2: post-encode framework node conditioning ------------------- #
    def condition_nodes(self, h_V, chain_mask, epi_per_res=None, epi_mask=None, cdr_ids=None, X=None):
        """Node-init + encoder cross-attn at CDR positions. Identity when disabled.

        ``cdr_ids`` (one CDR-type id per segment, from :func:`ordered_cdr_type_ids`) and ``X``
        (backbone coords ``[B, L, 4, 3]``) feed the step-5 identity / stem-geometry node signals;
        both are optional and ignored unless their flags are on.
        """
        if not self.cfg.enabled:
            return h_V
        segments = cdr_segments_from_chain_mask(chain_mask)

        if self.cfg.uses_node_init:
            pooled = None
            if self.cfg.node_init_pooled_epitope and epi_per_res is not None:
                pooled = self._pool(epi_per_res, epi_mask)
            h_V = self.node_init(h_V, segments, pooled, cdr_ids=cdr_ids, X=X)

        if self.cfg.encoder_cross_attn and epi_per_res is not None:
            gate = self._cdr_gate(chain_mask, h_V)                 # [1, L, 1]
            delta = self.enc_xattn(h_V, epi_per_res, epi_mask)     # [B, L, H]
            h_V = h_V + gate * delta
        return h_V

    # -- hook 3: per-step decoder cross-attn --------------------------------- #
    def decoder_cross_attn(self, h_V_t, epi_per_res=None, epi_mask=None):
        """Refine a per-step decoding hidden state ``[B, H]`` by epitope cross-attn.

        Identity when disabled. The caller gates this to CDR decode steps.
        """
        if not self.cfg.enabled or not self.cfg.decoder_cross_attn or epi_per_res is None:
            return h_V_t
        B = h_V_t.shape[0]
        if epi_per_res.shape[0] == 1 and B > 1:
            epi_per_res = epi_per_res.expand(B, -1, -1)
            if epi_mask is not None:
                epi_mask = epi_mask.expand(B, -1)
        delta = self.dec_xattn(h_V_t.unsqueeze(1), epi_per_res, epi_mask)  # [B, 1, H]
        return h_V_t + delta[:, 0]

    @staticmethod
    def _cdr_gate(chain_mask, ref):
        cm = chain_mask[0] if chain_mask.dim() > 1 else chain_mask
        return (cm > 0.5).to(ref.dtype).view(1, -1, 1)
