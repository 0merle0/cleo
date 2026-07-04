"""Epitope-conditioned dual-encoder ProteinMPNN policy (SPEC 4.1)."""

from cleo.design.nanobody.epitope import (
    CDRNodeInit,
    ConditioningConfig,
    EpitopeConditioner,
    PerResidueCrossAttention,
    cdr_segments_from_chain_mask,
    pool_epitope,
)

__all__ = [
    "ConditioningConfig",
    "EpitopeConditioner",
    "CDRNodeInit",
    "PerResidueCrossAttention",
    "cdr_segments_from_chain_mask",
    "pool_epitope",
]
