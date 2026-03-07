"""Resample combinatorial sequences from a fragment dictionary.

Given a JSON fragment dictionary (one list of unique fragments per region),
combinatorially samples full-length sequences with inverse-count weighting
for uniform fragment coverage. Useful for constructing diverse libraries
from a fixed set of fragment variants.

Usage:
    python -m cleo.design.resample_fragments --config-name resample_fragments
"""
import os
import json
import random
from pathlib import Path

import numpy as np
import hydra
from tqdm import tqdm


def load_fragment_dict(path):
    """
    Load fragment dictionary from JSON. Expected format:
    { "1": [[name, seq], ...], "2": [...], ... }

    Keys are fragment numbers (as strings). Fragment names follow the
    convention {frag_num}.{unique_id} — the first dot-delimited token
    is always the integer fragment number.

    Returns an ordered list of (fragment_number_str, [(name, seq), ...]) tuples,
    sorted numerically by fragment number.
    """
    with open(path) as f:
        raw = json.load(f)

    fragments = []
    for key in sorted(raw.keys(), key=lambda k: int(k)):
        entries = [(name, seq) for name, seq in raw[key]]
        fragments.append((key, entries))
    return fragments


def resample_sequences(fragments, num_sequences, connector):
    """
    Combinatorially sample full-length sequences from fragment regions
    using inverse-count weighting for uniform fragment coverage.

    Returns lists of (name, sequence) tuples.
    """
    num_regions = len(fragments)
    counts = [np.ones(len(entries)) for _, entries in fragments]

    names = []
    sequences = []

    for _ in tqdm(range(num_sequences), desc="Resampling"):
        sampled_names = []
        sampled_seqs = []

        for region_idx in range(num_regions):
            _, entries = fragments[region_idx]
            region_counts = counts[region_idx]

            weights = 1.0 / region_counts
            weights = weights / weights.sum()

            chosen_idx = random.choices(range(len(entries)), weights=weights, k=1)[0]
            counts[region_idx][chosen_idx] += 1

            frag_name, frag_seq = entries[chosen_idx]
            sampled_names.append(frag_name)
            sampled_seqs.append(frag_seq)

        full_name = connector.join(sampled_names)
        full_seq = "".join(sampled_seqs)
        names.append(full_name)
        sequences.append(full_seq)

    return names, sequences


def write_fasta(names, sequences, path):
    """Write name/sequence pairs to a FASTA file.

    Args:
        names: List of sequence identifiers.
        sequences: List of amino acid strings.
        path: Output FASTA file path.
    """
    with open(path, "w") as f:
        for name, seq in zip(names, sequences):
            f.write(f">{name}\n{seq}\n")


_CONFIG_DIR = str(Path(__file__).resolve().parent / "../../../config/design")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="resample_fragments")
def resample_fragments(cfg):
    """Hydra entrypoint: load a fragment dictionary and resample combinatorial sequences."""
    assert cfg.fragment_dict_path is not None, "fragment_dict_path is required"
    assert os.path.exists(cfg.fragment_dict_path), f"Fragment dict not found: {cfg.fragment_dict_path}"

    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Loading fragment dictionary from {cfg.fragment_dict_path}")
    fragments = load_fragment_dict(cfg.fragment_dict_path)

    print(f"Fragment regions: {len(fragments)}")
    for region_name, entries in fragments:
        print(f"  {region_name}: {len(entries)} unique fragments")

    print(f"\nResampling {cfg.num_sequences} combinatorial sequences...")
    names, sequences = resample_sequences(fragments, cfg.num_sequences, cfg.connector)

    fasta_path = os.path.join(cfg.output_dir, f"{cfg.output_name}.fasta")
    write_fasta(names, sequences, fasta_path)
    print(f"\nWrote {len(sequences)} sequences to {fasta_path}")

    print("Done.")


if __name__ == "__main__":
    resample_fragments()
