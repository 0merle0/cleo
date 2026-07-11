"""Batch CDR-diversity reward (SPEC §8 — reward diversity of generated CDRs).

Rewards each design for how *different* its CDR loops are from the other designs of the same CDR
type in the same rollout batch, to push the policy toward a diverse proposal distribution over
backbones rather than collapsing onto one loop per epitope. Reference-free and batch-relative, like
:mod:`cleo.design.utils.mutation_diversity`.

Two complementary signals (both computed; weight either/both in ``reward_aggregation``):

- **sequence** (``_cdr_seq_diversity``) — mean normalized string distance between this design's CDR
  of each type and every other design's same-type CDR. Needs only ``sequence`` + the per-chain
  ``cdr_spans`` (no structure), so it always works.
- **structural** (``_cdr_struct_diversity``) — mean pairwise CA-RMSD (Kabsch-superposed) between
  same-type CDR loops, read from the oracle's predicted structures. Only same-length pairs are
  comparable; when the structure is missing / gemmi is unavailable the value is NaN.

Per design, each signal is averaged over that design's CDR types (H1/H2/H3, L1-L3). Both are
"higher = more diverse", so aggregate them with ``mode: max``.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cleo.design.data.mask import _cdr_key


def _as_design_chains(row, sequence):
    """Normalize a df row to ``[{length, cdr_spans}, ...]`` in chain order.

    Prefers the multi-chain ``design_chains`` column (VHH = 1 entry, Fv = 2); falls back to the
    scalar single-chain ``cdr_spans`` column over the whole sequence.
    """
    spec = row.get("design_chains")
    if spec is not None and (isinstance(spec, (list, tuple)) or (isinstance(spec, str) and spec.strip())):
        chains = json.loads(spec) if isinstance(spec, str) else spec
        return [{"length": int(c["length"]), "cdr_spans": c.get("cdr_spans") or {}} for c in chains]
    spans = row.get("cdr_spans")
    if isinstance(spans, str) and spans.strip():
        spans = json.loads(spans)
    return [{"length": len(sequence), "cdr_spans": spans or {}}]


def _iter_cdrs(sequence, design_chains):
    """Yield ``(chain_id, cdr_type, start, end, subseq)`` for every CDR, chain order.

    ``chain_id`` mirrors the oracle's ``_designed_chain_ids`` (``A``, ``B``, ...); ``start``/``end``
    are half-open, chain-local (into that chain's decoded segment).
    """
    offset = 0
    for j, ch in enumerate(design_chains):
        L = ch["length"]
        seg = sequence[offset:offset + L]
        cid = chr(ord("A") + j)
        for key, span in (ch["cdr_spans"] or {}).items():
            s, e = int(span[0]), int(span[1])
            yield cid, _cdr_key(key), s, e, seg[s:e]
        offset += L


def _ca_by_chain(cif_path):
    """``{chain_id: [ [x,y,z] or None per residue ]}`` of CA coords, in residue order. ``{}`` on failure."""
    try:
        import gemmi
    except Exception:
        return {}
    try:
        model = gemmi.read_structure(cif_path)[0]
    except Exception:
        return {}
    out = {}
    for chain in model:
        cas = []
        for res in chain:
            ca = None
            for atom in res:
                if atom.name == "CA":
                    ca = [atom.pos.x, atom.pos.y, atom.pos.z]
                    break
            cas.append(ca)
        out[chain.name] = cas
    return out


def _seq_distance(a, b):
    """Normalized string distance in ``[0, 1]``: differing positions over the overlap plus the
    length gap, all divided by the longer length. Equal strings -> 0; disjoint -> 1."""
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    overlap = min(len(a), len(b))
    mism = sum(1 for i in range(overlap) if a[i] != b[i])
    return (mism + abs(len(a) - len(b))) / n


def _kabsch_rmsd(P, Q):
    """CA-RMSD (Å) between two ``[N, 3]`` point sets after optimal superposition."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = Pc @ R.T - Qc
    return float(np.sqrt((diff ** 2).sum() / P.shape[0]))


def cdr_diversity_from_df(df_input, cfg, step_name="cdr_diversity"):
    """Per-design CDR sequence + structural diversity relative to the rest of the batch.

    cfg:
        structure_path_col: df column with the predicted-structure path for the structural signal
            (default ``protenix_path`` — the oracle step's output). Set to ``null``/absent to skip.

    Output columns (per design): ``{step_name}_cdr_seq_diversity`` and
    ``{step_name}_cdr_struct_diversity`` (NaN when no structure is available).
    """
    if len(df_input) == 0:
        return df_input
    path_col = cfg.get("structure_path_col", "protenix_path") if hasattr(cfg, "get") else "protenix_path"

    # --- gather per-design CDRs (subsequence + CA coords per type) ----------
    designs = []                        # [{name, cdrs: {type: {"seq":str, "ca":np.ndarray|None}}}]
    for _, row in df_input.iterrows():
        seq = row["sequence"]
        chains = _as_design_chains(row, seq)
        ca_by_chain = _ca_by_chain(row[path_col]) if (path_col and path_col in row and isinstance(row[path_col], str)) else {}
        cdrs = {}
        for cid, ctype, s, e, subseq in _iter_cdrs(seq, chains):
            ca = None
            chain_cas = ca_by_chain.get(cid)
            if chain_cas is not None and e <= len(chain_cas):
                sl = chain_cas[s:e]
                if sl and all(p is not None for p in sl):
                    ca = np.asarray(sl, dtype=float)
            cdrs[ctype] = {"seq": subseq, "ca": ca}
        designs.append({"name": row["name"], "cdrs": cdrs})

    # --- per type, index designs that carry it -----------------------------
    by_type = {}
    for idx, d in enumerate(designs):
        for ctype in d["cdrs"]:
            by_type.setdefault(ctype, []).append(idx)

    rows = []
    for idx, d in enumerate(designs):
        seq_scores, struct_scores = [], []
        for ctype, entry in d["cdrs"].items():
            others = [k for k in by_type[ctype] if k != idx]
            if not others:
                continue
            sdists = [_seq_distance(entry["seq"], designs[k]["cdrs"][ctype]["seq"]) for k in others]
            seq_scores.append(float(np.mean(sdists)))
            if entry["ca"] is not None:
                rmsds = [
                    _kabsch_rmsd(entry["ca"], designs[k]["cdrs"][ctype]["ca"])
                    for k in others
                    if designs[k]["cdrs"][ctype]["ca"] is not None
                    and designs[k]["cdrs"][ctype]["ca"].shape == entry["ca"].shape
                ]
                if rmsds:
                    struct_scores.append(float(np.mean(rmsds)))
        rows.append({
            "name": d["name"],
            f"{step_name}_cdr_seq_diversity": float(np.mean(seq_scores)) if seq_scores else 0.0,
            f"{step_name}_cdr_struct_diversity": float(np.mean(struct_scores)) if struct_scores else float("nan"),
        })

    return pd.merge(df_input, pd.DataFrame(rows), on="name", how="inner")
