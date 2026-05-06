"""
Sequence distance utilities for the reward pipeline.

Computes Hamming distances between sampled sequences and a set of
reference sequences, providing min/avg/max distance metrics.
"""

import pandas as pd

def get_dist_from_seqs(seq, ref_list):
    """
    Given a sequence string and a list of reference sequences, compute hamming distances
    and aggregate them.
    """

    distances = []
    for ref_seq in ref_list:
        if len(seq) != len(ref_seq):
            raise ValueError(f"Sequence length mismatch: {len(seq)} vs {len(ref_seq)}")
        dist = sum(c1 != c2 for c1, c2 in zip(seq, ref_seq))
        distances.append(dist)

    ret = {
        "min": min(distances),
        "avg": sum(distances) / len(distances),
        "max": max(distances)
    }

    return ret


def compute_dist_to_ref_seqs_from_df(df_input, cfg, step_name="dist_to_ref_seqs"):
    """
    Computes distance to reference sequences for each sequence in the dataframe.
    """
    if df_input is None or len(df_input) == 0:
        raise ValueError(
            "dist_to_ref_seqs: empty input dataframe. A prior step (e.g. af3) may have dropped "
            "all rows, or the pipeline was called with no sequences."
        )

    metrics_list = []

    for idx, row in df_input.iterrows():
        metrics = {}
        seq = row["sequence"]
        metrics = get_dist_from_seqs(seq, cfg.ref_seqs)

        _tmp = {f"{step_name}_{k}": v for k,v in metrics.items()}
        _tmp["name"] = row["name"]
        metrics_list.append(_tmp)

    output_df =  pd.DataFrame(metrics_list)
    df_merged = pd.merge(df_input, output_df, on="name", how="inner")

    return df_merged