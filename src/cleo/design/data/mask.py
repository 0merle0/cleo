"""Resolve semantic design regions into ProteinMPNN ``fixed_residues`` tokens.

The design region of an antibody example is defined by ``params.cdr_spans``
(authoritative post-length-sampling; SPEC 6.1), given as **half-open positional**
index ranges into the *design chain's* residue order (0-based, into the chain's
``vd_seq``). ``featurize_pdb`` designs every position whose residue token is **not**
in ``cfg.fixed_residues``, so masking to CDR-only means emitting the **complement**:
every residue token in the structure *except* the CDR positions on the design
chain(s). The antigen chain (``T``) is therefore always fully fixed.

Residue tokens are ``{chain}{R_idx}{icode}`` — identical to the enumeration
``featurize_pdb`` builds from ``parse_PDB`` (verified: no-icode residues render as
``L1``, ``L2``, ... under both ``parse_PDB`` and gemmi ``{chain}{seqid.num}``) — so
the tokens produced here line up 1:1 with the mask the featurizer applies.

Chain routing for a paired Fab: ``design_chain = ["A", "B"]`` and CDR keys are
routed by prefix — ``H*`` -> first design chain, ``L*`` -> second.
"""
from __future__ import annotations

import random


class MaskError(ValueError):
    """Raised when design regions cannot be resolved against a structure."""


def sample_cdr_lengths(params: dict, rng: random.Random | None = None) -> dict[str, int]:
    """Draw a design length for each CDR (SPEC 4.5 / 6.8 step 3.5).

    Length is an explicit design variable, decoupled from the framework template: the CDRs
    are gapped out and this many residues are decoded back in. Two sources, in priority order:

    - ``params.cdr_lengths`` — a fixed ``{cdr: n}`` override (inference / deterministic tests).
    - ``params.cdr_length_ranges`` — ``{cdr: [lo, hi]}`` inclusive; one length sampled per CDR
      each call (Stage-1 broad prior gets length diversity, per user 2026-07-05).

    A fixed override wins per-CDR; any CDR without either source falls back to its ``cdr_spans``
    native width (so a spans-only example reproduces its template length). Returns ``{cdr: n}``
    over the union of CDR keys mentioned by any of the three sources.
    """
    rng = rng or random
    fixed = {str(k): int(v) for k, v in (params.get("cdr_lengths") or {}).items()}
    ranges = params.get("cdr_length_ranges") or {}
    spans = params.get("cdr_spans") or {}

    keys = set(fixed) | set(ranges) | set(spans)
    out: dict[str, int] = {}
    for k in keys:
        if k in fixed:
            n = fixed[k]
        elif k in ranges:
            lo, hi = int(ranges[k][0]), int(ranges[k][1])
            if hi < lo:
                raise MaskError(f"cdr_length_ranges[{k!r}] = [{lo}, {hi}] is reversed (need lo <= hi)")
            n = rng.randint(lo, hi)
        else:
            s, e = int(spans[k][0]), int(spans[k][1])
            n = e - s
        if n <= 0:
            raise MaskError(f"CDR {k!r} resolved to non-positive length {n}")
        out[k] = n
    return out


def as_chain_list(design_chain) -> list[str]:
    """Normalize a ``design_chain`` field to a list of chain letters.

    Accepts a list (``["H", "L"]``), a single-chain string (``"H"``), or a whitespace-joined
    multi-chain string (``"H L"`` -> ``["H", "L"]``). The whitespace form is safe because chain
    IDs are single tokens, so an embedded space can only delimit chains — this keeps the VHH
    (1-chain) and Fv (2-chain) paths robust regardless of how ``design_chain`` was serialized."""
    if isinstance(design_chain, str):
        return design_chain.split()
    return [str(c) for c in design_chain]


def _cdr_key(key: str) -> str:
    k = str(key).upper()
    return k[4:] if k.startswith("CDR_") else k


def _route_spans(cdr_spans: dict, design_chains: list[str]) -> dict[str, list[tuple[int, int]]]:
    """Map CDR span keys to design chains by prefix (H* -> chain 0, L* -> chain 1)."""
    routed: dict[str, list[tuple[int, int]]] = {c: [] for c in design_chains}
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
        routed[chain].append((s, e))
    return routed


def _chain_positions(chain_letters, chain: str) -> list[int]:
    return [i for i, c in enumerate(chain_letters) if str(c) == chain]


def design_indices(chain_letters, design_chain, cdr_spans: dict) -> set[int]:
    """Return the set of flat residue indices that are DESIGNABLE (the CDR positions)."""
    design_chains = as_chain_list(design_chain)
    routed = _route_spans(cdr_spans, design_chains)
    idxs: set[int] = set()
    for chain, spans in routed.items():
        pos = _chain_positions(chain_letters, chain)
        if not pos:
            raise MaskError(f"design_chain {chain!r} has no residues in the structure")
        for (s, e) in spans:
            if s < 0 or e > len(pos):
                raise MaskError(
                    f"CDR span [{s},{e}) is out of bounds for chain {chain!r} (length {len(pos)})"
                )
            idxs.update(pos[s:e])
    return idxs


def token(chain_letters, R_idx, icodes, i: int) -> str:
    return f"{chain_letters[i]}{int(R_idx[i])}{icodes[i]}"


def resolve_fixed_residues(chain_letters, R_idx, icodes, design_chain, cdr_spans: dict) -> str:
    """Build the ``fixed_residues`` string = every residue token EXCEPT the CDR positions.

    Args:
        chain_letters, R_idx, icodes: parallel per-residue arrays in structure order,
            exactly as produced by ``parse_PDB`` (``chain_letters``, ``R_idx``, ``icodes``).
        design_chain: chain letter (VHH ``"A"``) or list (Fab ``["A", "B"]``).
        cdr_spans: ``{"H1": [s, e], ...}`` half-open positional ranges into the design chain.

    Returns:
        Space-separated ``{chain}{R_idx}{icode}`` tokens consumed by ``featurize_pdb``.
    """
    design = design_indices(chain_letters, design_chain, cdr_spans)
    n = len(chain_letters)
    return " ".join(token(chain_letters, R_idx, icodes, i) for i in range(n) if i not in design)
