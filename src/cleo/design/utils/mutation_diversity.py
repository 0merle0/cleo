"""
Batch-level mutation diversity rewards for the design pipeline, plus helpers for
batch-wide logging (pairwise distance, unique mutation-pair tallies).

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
        _total_muts_vs_consensus: Hamming distance to the per-position
            mode-consensus of the batch (ties broken by lexicographic AA).
            NaN if fewer than two same-length sequences are available.
        _pairwise_hamming_mean: Mean Hamming distance to all other same-length
            sequences in the batch. NaN if fewer than two are available.
        _pairwise_hamming_min: Nearest-neighbour Hamming distance in the batch
            (0 ⇒ duplicate sequence). NaN if fewer than two are available.
        _marginal_count_vs_consensus: Number of (position, AA) deviations from
            the batch consensus that no other batch member carries. Direct
            per-sequence surrogate for the batch-level count of unique mutation
            sites vs consensus.
        _marginal_fraction_vs_consensus: marginal_count_vs_consensus divided by
            total_muts_vs_consensus (NaN if total = 0).
    """
    ref_seq = cfg.ref_seq
    sequences = df_input["sequence"].tolist()

    mutation_sets = _get_mutation_sets(sequences, ref_seq)
    global_counts = _mutation_counts(mutation_sets)

    d_cons, pw_mean, pw_min, marg_cons, marg_cons_frac = _batch_consensus_and_pairwise(
        sequences
    )

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
            f"{step_name}_total_muts_vs_consensus": d_cons[idx],
            f"{step_name}_pairwise_hamming_mean": pw_mean[idx],
            f"{step_name}_pairwise_hamming_min": pw_min[idx],
            f"{step_name}_marginal_count_vs_consensus": marg_cons[idx],
            f"{step_name}_marginal_fraction_vs_consensus": marg_cons_frac[idx],
        })

    output_df = pd.DataFrame(metrics_list)
    return pd.merge(df_input, output_df, on="name", how="inner")


def _batch_consensus_and_pairwise(
    sequences: list[str],
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """
    Per-sequence (Hamming-to-consensus, mean pairwise Hamming, min pairwise Hamming,
    marginal_count vs consensus, marginal_fraction vs consensus), computed over the
    same-length subset (dominant length within the batch). Entries for sequences
    outside the dominant length, or when fewer than two same-length sequences exist,
    are NaN.

    marginal_count_vs_consensus: number of (position, AA) pairs where this sequence
    differs from consensus AND no other same-length batch member carries that exact
    (position, AA) pair.
    marginal_fraction_vs_consensus: marginal_count / total_muts_vs_consensus (NaN
    if total is 0).
    """
    n = len(sequences)
    nan = float("nan")
    d_cons = [nan] * n
    pw_mean = [nan] * n
    pw_min = [nan] * n
    marg_cons = [nan] * n
    marg_cons_frac = [nan] * n
    if n < 2:
        return d_cons, pw_mean, pw_min, marg_cons, marg_cons_frac

    lens = [len(s) for s in sequences]
    L = Counter(lens).most_common(1)[0][0]
    keep = [i for i, ln in enumerate(lens) if ln == L]
    if len(keep) < 2:
        return d_cons, pw_mean, pw_min, marg_cons, marg_cons_frac

    X = np.vstack(
        [np.frombuffer(sequences[i].encode("ascii"), dtype=np.uint8) for i in keep]
    )
    consensus = np.empty(L, dtype=np.uint8)
    for j in range(L):
        vals, counts = np.unique(X[:, j], return_counts=True)
        ties = vals[counts == counts.max()]
        consensus[j] = np.sort(ties)[0]

    dc = (X != consensus).sum(axis=1)
    D = (X[:, None, :] != X[None, :, :]).sum(axis=2).astype(np.int64)
    row_sum = D.sum(axis=1)
    means = row_sum / (D.shape[0] - 1)
    D_for_min = D.copy()
    np.fill_diagonal(D_for_min, np.iinfo(np.int64).max)
    mins = D_for_min.min(axis=1)

    diff_mask = X != consensus  # (n_keep, L)
    cons_mut_counts: Counter = Counter()
    for r in range(X.shape[0]):
        cols_r = np.where(diff_mask[r])[0]
        for j in cols_r:
            cons_mut_counts[(int(j), int(X[r, j]))] += 1
    marg_arr = np.zeros(X.shape[0], dtype=np.int64)
    for r in range(X.shape[0]):
        cols_r = np.where(diff_mask[r])[0]
        c = 0
        for j in cols_r:
            if cons_mut_counts[(int(j), int(X[r, j]))] == 1:
                c += 1
        marg_arr[r] = c

    for k, i in enumerate(keep):
        d_cons[i] = float(dc[k])
        pw_mean[i] = float(means[k])
        pw_min[i] = float(mins[k])
        marg_cons[i] = float(marg_arr[k])
        total_c = int(dc[k])
        marg_cons_frac[i] = float(marg_arr[k]) / total_c if total_c > 0 else nan
    return d_cons, pw_mean, pw_min, marg_cons, marg_cons_frac


def mode_consensus_sequence(sequences: list[str]) -> str:
    """Per-position majority consensus (for ties, use lexicographically first AA)."""
    if not sequences:
        return ""
    L = len(sequences[0])
    for s in sequences:
        if len(s) != L:
            raise ValueError("All sequences in a batch must be the same length for consensus")
    out: list[str] = []
    for i in range(L):
        col = [s[i] for s in sequences]
        c = Counter(col)
        mfreq = max(c.values())
        best = sorted([aa for aa, n in c.items() if n == mfreq])
        out.append(best[0])
    return "".join(out)


def count_unique_mutation_pairs(sequences: list[str], reference: str) -> int:
    """
    |⋃_seq { (i, seq[i]) : seq[i] ≠ reference[i] }| — distinct (position, amino acid) pairs
    that appear in the batch as a deviation from *reference* (e.g. template VHH, consensus, etc.).
    """
    if not sequences or len(reference) == 0:
        return 0
    L = len(sequences[0])
    if len(reference) != L or any(len(s) != L for s in sequences):
        raise ValueError("Sequence length mismatch vs reference in count_unique_mutation_pairs")
    u: set[tuple[int, str]] = set()
    for seq in sequences:
        for i, aa in enumerate(seq):
            if aa != reference[i]:
                u.add((i, aa))
    return len(u)


def pairwise_diversity_from_sequences(sequences: list[str]) -> dict[str, float]:
    """
    Mean Hamming and mean normalised Hamming (Hamming / length) over all unordered
    sequence pairs in the batch.
    """
    n = len(sequences)
    if n < 2:
        return {
            "batch_pairwise_hamming_mean": 0.0,
            "batch_pairwise_frac_diff_mean": 0.0,
        }
    L = len(sequences[0])
    hams: list[int] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            h = sum(1 for a, b in zip(sequences[i], sequences[j]) if a != b)
            hams.append(h)
    mean_h = float(np.mean(hams))
    return {
        "batch_pairwise_hamming_mean": mean_h,
        "batch_pairwise_frac_diff_mean": float(mean_h / L) if L else 0.0,
    }


def batch_diversity_log_metrics(
    sequences: list[str], ref_seq: str | None
) -> dict[str, float]:
    """
    One dict of **floats** to merge into training step logs: pairwise library diversity,
    count of unique (position, AA) vs the project reference, and the same count vs
    a per-position **consensus (mode) sequence** in the batch as a “average” design.
    """
    if not sequences:
        return {}

    d = pairwise_diversity_from_sequences(sequences)
    cons = mode_consensus_sequence(sequences)
    d["batch_unique_mutation_sites_wrt_consensus"] = float(
        count_unique_mutation_pairs(sequences, cons)
    )
    if ref_seq is not None and len(ref_seq) == len(sequences[0]):
        d["batch_unique_mutation_sites_wrt_ref"] = float(
            count_unique_mutation_pairs(sequences, ref_seq)
        )
    else:
        d["batch_unique_mutation_sites_wrt_ref"] = float("nan")
    return d
