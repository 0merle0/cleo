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

import pandas as pd


def _get_mutation_sets(sequences, ref_seq):
    """Build per-sequence sets of (position, amino_acid) mutations vs reference."""
    mutation_sets = []
    for seq in sequences:
        muts = {(i, aa) for i, aa in enumerate(seq) if aa != ref_seq[i]}
        mutation_sets.append(muts)
    return mutation_sets


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
        ref_seq (str): Parent/reference amino acid sequence. Mutations are
            defined as positions where a sequence differs from this reference.

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
    ref_seq = cfg.ref_seq
    sequences = df_input["sequence"].tolist()

    mutation_sets = _get_mutation_sets(sequences, ref_seq)
    global_counts = _mutation_counts(mutation_sets)

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
        })

    output_df = pd.DataFrame(metrics_list)
    return pd.merge(df_input, output_df, on="name", how="inner")
