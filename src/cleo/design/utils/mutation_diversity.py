"""
Batch-level mutation diversity rewards for the design pipeline.

Computes per-sequence rewards based on how many unique (position, amino_acid)
mutations each sequence contributes to a batch, incentivizing the policy to
explore diverse regions of mutation space rather than converging on the same
mutations.

Two scoring strategies are provided:

- **Marginal (exclusive)**: A sequence gets credit only for mutations that
  no other sequence in the batch carries. Sparse signal, strong pressure.
- **Fractional (1/k)**: Each mutation gives 1/k credit where k is the number
  of batch members sharing that (position, AA) pair. Smoother signal, more
  robust to batch size.
"""

from collections import Counter

import numpy as np
import pandas as pd


def pairwise_hamming(sequences):
    """Per-sequence mean fractional Hamming distance to every other batch member.

    Returns an array in [0,1]: 0 if a sequence is identical to all its peers,
    1 if it differs from all of them at every position.

    Computed per position rather than over pairs. At one position, if k of the
    B sequences carry your residue, then k-1 of your B-1 peers match you there,
    so mean pairwise identity is the average of (k-1)/(B-1) over positions and
    the distance is one minus that. Identical to enumerating all pairs, at O(B*L)
    instead of O(B^2*L).

    Distinct from the 1/k rarity score also computed here. 1/k is convex and so
    rewards being a *singleton* very steeply (k=1 -> 1.0, k=2 -> 0.5), whereas
    pairwise distance is linear in k and measures spread. For a batch of 16 with
    three identical members, the two disagree by 2.46x vs 1.14x on how much
    better a unique sequence is.

    All positions count, including any pinned by ``fixed_residues``. Those are
    constant across the batch and contribute zero distance, so they scale every
    sequence identically and do not affect ranking.
    """
    arr = np.array([list(s) for s in sequences])
    B, L = arr.shape
    if B < 2:
        return np.zeros(B)
    identity = np.zeros(B)
    for j in range(L):
        col = arr[:, j]
        _, inv, counts = np.unique(col, return_inverse=True, return_counts=True)
        k = counts[inv]
        identity += (k - 1) / (B - 1)
    return 1.0 - identity / L


def _get_mutation_sets(sequences, ref_seq=None):
    """Build per-sequence sets of (position, amino_acid) pairs.

    With ``ref_seq``, only positions differing from the reference count, which
    measures diversity *among mutations away from a known parent*.

    With ``ref_seq=None`` every position counts, giving reference-free pairwise
    diversity across the batch: how rare each of a sequence's choices is among
    its peers, with no privileged parent. Prefer this when there is no
    meaningful parent -- e.g. de novo backbones, whose design PDB is poly-ALA,
    where a reference would silently make every alanine choice invisible and
    turn the metric into non-alanine diversity only.
    """
    if ref_seq is None:
        return [{(i, aa) for i, aa in enumerate(seq)} for seq in sequences]
    return [{(i, aa) for i, aa in enumerate(seq) if aa != ref_seq[i]}
            for seq in sequences]


def _mutation_counts(mutation_sets):
    """Count how many sequences carry each (position, AA) mutation."""
    counts = Counter()
    for muts in mutation_sets:
        for m in muts:
            counts[m] += 1
    return counts


def mutation_diversity_from_df(df_input, cfg, step_name="mutation_diversity"):
    """Compute per-sequence mutation diversity metrics for a batch.

    For each sequence, reports both marginal (exclusive) and fractional (1/k)
    mutation contribution scores relative to the rest of the batch.

    Config fields:
        ref_seq (str, optional): Parent/reference amino acid sequence. Mutations
            are positions where a sequence differs from it. Omit (or set None)
            for reference-free pairwise diversity over the batch, where every
            position counts -- the right choice for de novo backbones with no
            meaningful parent. Reference-free, ``_fractional_normalized`` is the
            mean 1/k over all positions: 1.0 for a sequence unique everywhere,
            1/batch_size for one identical to all its peers.

    Output columns (prefixed by step_name):
        _total_muts: Total number of mutations vs the reference.
        _marginal_count: Number of mutations exclusive to this sequence
            (not shared with any other batch member).
        _marginal_fraction: Fraction of this sequence's mutations that are
            exclusive.
        _fractional_score: Sum of 1/k credits across all mutations, where
            k is the number of batch members sharing each mutation.
        _fractional_normalized: Fractional score divided by total mutations
            (average 1/k across this sequence's mutations).
    """
    ref_seq = cfg.get("ref_seq", None) if hasattr(cfg, "get") else getattr(cfg, "ref_seq", None)
    sequences = df_input["sequence"].tolist()

    mutation_sets = _get_mutation_sets(sequences, ref_seq)
    global_counts = _mutation_counts(mutation_sets)

    pw = pairwise_hamming(sequences)

    metrics_list = []
    for idx in range(len(sequences)):
        my_muts = mutation_sets[idx]
        total_muts = len(my_muts)

        marginal_count = sum(1 for m in my_muts if global_counts[m] == 1)

        fractional_score = sum(1.0 / global_counts[m] for m in my_muts)

        metrics_list.append({
            "name": df_input.iloc[idx]["name"],
            f"{step_name}_total_muts": total_muts,
            f"{step_name}_marginal_count": marginal_count,
            f"{step_name}_marginal_fraction": marginal_count / max(total_muts, 1),
            f"{step_name}_fractional_score": fractional_score,
            f"{step_name}_fractional_normalized": fractional_score / max(total_muts, 1),
            f"{step_name}_pairwise_hamming": float(pw[idx]),
        })

    output_df = pd.DataFrame(metrics_list)
    return pd.merge(df_input, output_df, on="name", how="inner")
