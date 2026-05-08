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


def _position_counts(sequences):
    """For each position, return a Counter of AA -> occurrence in the batch."""
    seq_len = len(sequences[0])
    return [Counter(seq[i] for seq in sequences) for i in range(seq_len)]


def mutation_diversity_from_df(df_input, cfg, step_name="mutation_diversity"):
    """Per-sequence divergence from the batch consensus.

    For each position, the consensus set is the batch's modal AA(s) (ties
    treated as multi-modal so even splits aren't penalized).

    Output columns (prefixed by step_name):
        _consensus_divergence: Number of positions where this sequence's AA
            is not among the batch's modal AA(s) at that position.
        _consensus_divergence_fraction: Above, divided by sequence length.
        _fractional_divergence: At each divergent position, credit 1/k where
            k is the number of batch members sharing this sequence's
            non-consensus AA. Penalizes secondary-mode collapse.
        _fractional_divergence_normalized: Fractional divergence divided by
            sequence length.
    """
    sequences = df_input["sequence"].tolist()
    if not sequences:
        return df_input

    if len({len(s) for s in sequences}) != 1:
        raise ValueError("All sequences in batch must have the same length.")

    pos_counts = _position_counts(sequences)
    seq_len = len(pos_counts)
    consensus = [
        {aa for aa, c in counts.items() if c == max(counts.values())}
        for counts in pos_counts
    ]

    metrics_list = []
    for idx, seq in enumerate(sequences):
        divergence = 0
        fractional = 0.0
        for i, aa in enumerate(seq):
            if aa not in consensus[i]:
                divergence += 1
                fractional += 1.0 / pos_counts[i][aa]

        metrics_list.append({
            "name": df_input.iloc[idx]["name"],
            f"{step_name}_consensus_divergence": divergence,
            f"{step_name}_consensus_divergence_fraction": divergence / seq_len,
            f"{step_name}_fractional_divergence": fractional,
            f"{step_name}_fractional_divergence_normalized": fractional / seq_len,
        })

    output_df = pd.DataFrame(metrics_list)
    return pd.merge(df_input, output_df, on="name", how="inner")
