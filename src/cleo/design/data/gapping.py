"""Excise native CDRs from a parsed framework and splice in gap nodes (SPEC 4.5 / 6.8 step 3.5).

Stage-1 treats the framework as if its CDRs were fully gapped: no positional information leaks
from the template, and CDR **length** is an explicit design variable (sampled per step from
``cdr_length_ranges``; see :func:`cleo.design.data.mask.sample_cdr_lengths`). This module does the
structural surgery on the *parsed* ``protein_dict`` (the output of ``parse_PDB``, before
``featurize`` batches/renumbers it):

- The residues in each native ``cdr_span`` are **excised**.
- ``N`` **gap nodes** are spliced in at that junction, where ``N`` is the sampled length.
- Gap nodes are marked **designable** (``chain_mask = 1``); everything else is fixed (``0``).

Gap-node backbone coordinates are copied from the flanking framework **stem** residue purely so
downstream tensors stay finite — they are never read as features: this path is only valid with
``coord_free_cdr_edges`` on, where every CDR-incident edge's distance block is replaced by a
learned unknown-distance token and CDR kNN is stem-borrowed (SPEC 4.1). The design chain is
renumbered contiguously so positional edge features see the gap nodes adjacent to their stems.
"""
from __future__ import annotations

import torch

from cleo.design.data.mask import MaskError, as_chain_list, _cdr_key

# ProteinMPNN alphabet index of the unknown/mask residue 'X' (last token); gap nodes start here
# and are overwritten during autoregressive decoding (they are chain_mask==1 = designed).
_UNK_AA = 20


def _route_spans_keyed(cdr_spans: dict, design_chains: list[str]):
    """``[(chain, key, s, e)]`` routing each CDR span to a design chain (H*->0, L*->1)."""
    routed = []
    for key, span in cdr_spans.items():
        s, e = int(span[0]), int(span[1])
        if e <= s:
            raise MaskError(f"CDR span {key}={list(span)} is empty or reversed (need s < e, half-open)")
        k = _cdr_key(key)
        if k.startswith("H"):
            chain = design_chains[0]
        elif k.startswith("L"):
            if len(design_chains) < 2:
                raise MaskError(
                    f"CDR span {key!r} targets a light chain but design_chain={design_chains} "
                    "has no second chain"
                )
            chain = design_chains[1]
        else:
            raise MaskError(f"CDR span key {key!r} must start with H or L")
        routed.append((chain, k, s, e))
    return routed


def apply_cdr_gaps(protein_dict: dict, design_chain, cdr_spans: dict, cdr_lengths: dict) -> dict:
    """Return a new ``protein_dict`` with native CDRs excised and gap nodes spliced in.

    Args:
        protein_dict: parsed structure from ``parse_PDB`` (un-batched; per-residue ``X`` [L,4,3],
            ``mask`` [L], ``R_idx`` [L], ``chain_labels`` [L], ``S`` [L], ``xyz_37`` [L,37,3],
            ``xyz_37_m`` [L,37], and the ``chain_letters`` list).
        design_chain: chain letter (VHH ``"A"``) or list (Fab ``["A", "B"]``).
        cdr_spans: ``{"H1": [s, e], ...}`` half-open positional ranges into the design chain
            (the **native** residues to remove).
        cdr_lengths: ``{"H1": n, ...}`` how many gap nodes to splice at each CDR junction.

    Returns:
        A shallow-copied ``protein_dict`` with rebuilt per-residue tensors, a fresh ``chain_mask``
        (1 = gap/designable, 0 = fixed), and rebuilt ``mask_c`` / ``chain_list``.
    """
    design_chains = as_chain_list(design_chain)
    chain_letters = list(protein_dict["chain_letters"])
    L = len(chain_letters)
    device = protein_dict["X"].device

    routed = _route_spans_keyed(cdr_spans, design_chains)

    # Map each CDR to its native flat indices; validate against the actual chain lengths.
    excised: set[int] = set()
    gap_at: dict[int, int] = {}          # first-native-flat-index -> number of gap nodes
    for chain, key, s, e in routed:
        pos = [i for i, c in enumerate(chain_letters) if str(c) == chain]
        if not pos:
            raise MaskError(f"design_chain {chain!r} has no residues in the structure")
        if s < 0 or e > len(pos):
            raise MaskError(f"CDR span [{s},{e}) out of bounds for chain {chain!r} (length {len(pos)})")
        n = int(cdr_lengths[key])
        if n <= 0:
            raise MaskError(f"CDR {key!r} has non-positive gap length {n}")
        native = pos[s:e]
        if excised & set(native):
            raise MaskError(f"CDR span {key!r} overlaps another CDR span")
        excised.update(native)
        gap_at[native[0]] = n

    # Nearest framework stem (same chain) whose backbone we copy into gap nodes, purely for finite
    # tensors (never read as a feature under coord_free_cdr_edges).
    def _stem_flat(first: int) -> int:
        chain = chain_letters[first]
        if first - 1 >= 0 and str(chain_letters[first - 1]) == str(chain) and (first - 1) not in excised:
            return first - 1
        # CDR at chain start: use the C-side stem (first residue after the excised run).
        j = first
        while j in excised:
            j += 1
        return j if j < L else first  # degenerate all-excised chain: fall back to self

    X, mask, R_idx = protein_dict["X"], protein_dict["mask"], protein_dict["R_idx"]
    S, chain_labels = protein_dict["S"], protein_dict["chain_labels"]
    xyz_37, xyz_37_m = protein_dict["xyz_37"], protein_dict["xyz_37_m"]

    keep_rows: list[int] = []            # source flat index for a real residue, or -1 for a gap
    stem_rows: list[int] = []            # stem flat index backing a gap row (else -1)
    new_letters: list[str] = []
    is_gap: list[bool] = []
    for i in range(L):
        if i in gap_at:
            stem = _stem_flat(i)
            for _ in range(gap_at[i]):
                keep_rows.append(-1)
                stem_rows.append(stem)
                new_letters.append(str(chain_letters[i]))
                is_gap.append(True)
        if i in excised:
            continue                     # drop the native CDR residue
        keep_rows.append(i)
        stem_rows.append(-1)
        new_letters.append(str(chain_letters[i]))
        is_gap.append(False)

    # Source index for gathering each output row: the residue itself, or its stem for a gap row.
    src = torch.tensor(
        [stem_rows[k] if keep_rows[k] < 0 else keep_rows[k] for k in range(len(keep_rows))],
        device=device, dtype=torch.long,
    )
    gap_t = torch.tensor(is_gap, device=device, dtype=torch.bool)

    out = dict(protein_dict)
    out["X"] = X.index_select(0, src)
    out["xyz_37"] = xyz_37.index_select(0, src)
    out["xyz_37_m"] = xyz_37_m.index_select(0, src)
    out["mask"] = mask.index_select(0, src)               # gap rows inherit a valid stem mask (present)
    out["chain_labels"] = chain_labels.index_select(0, src)
    out["chain_letters"] = new_letters

    S_new = S.index_select(0, src)
    S_new[gap_t] = _UNK_AA                                 # gap sequence = unknown; decoded back in
    out["S"] = S_new

    # Renumber the design chain(s) contiguously so positional edge features see gap nodes adjacent
    # to their stems; non-design chains keep their native numbering.
    R_new = R_idx.index_select(0, src).clone()
    for chain in design_chains:
        pos = [k for k, c in enumerate(new_letters) if str(c) == str(chain)]
        for rank, k in enumerate(pos):
            R_new[k] = rank
    out["R_idx"] = R_new

    out["chain_mask"] = gap_t.to(mask.dtype)              # 1 = gap/designable, 0 = fixed
    out["side_chain_mask"] = out["chain_mask"]

    # Rebuild the chain-membership helpers over the new ordering.
    chain_list = sorted(set(new_letters))
    out["mask_c"] = [
        torch.tensor([c == item for item in new_letters], device=device, dtype=torch.bool)
        for c in chain_list
    ]
    out["chain_list"] = chain_list
    return out
