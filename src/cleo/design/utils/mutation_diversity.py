"""
Batch consensus-divergence reward.

For each sequence in a batch, counts the number of positions where its amino
acid differs from the batch consensus (the per-position modal AA across the
batch). Reference-free: rewards being unusual relative to wherever the batch
is currently piling up, rather than relative to a fixed parent sequence.

A position with multiple tied modal AAs treats any of them as consensus, so
sequences aren't penalized for being part of an even split.
"""

from collections import Counter

import pandas as pd


def _consensus_sets(sequences):
    """For each position, return the set of modal AAs (handles ties)."""
    seq_len = len(sequences[0])
    consensus = []
    for i in range(seq_len):
        counts = Counter(seq[i] for seq in sequences)
        top = max(counts.values())
        consensus.append({aa for aa, c in counts.items() if c == top})
    return consensus


def mutation_diversity_from_df(df_input, cfg, step_name="mutation_diversity"):
    """Per-sequence divergence from the batch consensus.

    Output columns (prefixed by step_name):
        _consensus_divergence: Number of positions where this sequence's AA
            is not among the batch's modal AA(s) at that position.
        _consensus_divergence_fraction: Above, divided by sequence length.
    """
    sequences = df_input["sequence"].tolist()
    if not sequences:
        return df_input

    if len({len(s) for s in sequences}) != 1:
        raise ValueError("All sequences in batch must have the same length.")

    consensus = _consensus_sets(sequences)
    seq_len = len(consensus)

    metrics_list = []
    for idx, seq in enumerate(sequences):
        divergence = sum(1 for i, aa in enumerate(seq) if aa not in consensus[i])
        metrics_list.append({
            "name": df_input.iloc[idx]["name"],
            f"{step_name}_consensus_divergence": divergence,
            f"{step_name}_consensus_divergence_fraction": divergence / seq_len,
        })

    output_df = pd.DataFrame(metrics_list)
    return pd.merge(df_input, output_df, on="name", how="inner")
