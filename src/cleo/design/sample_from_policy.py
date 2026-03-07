"""Sample sequences from a trained ProteinMPNN policy checkpoint.

Loads one or more policy checkpoints, samples batches of amino acid
sequences, writes full-length FASTA output, and optionally splits
sequences into fragment regions for downstream library construction.

Usage:
    python -m cleo.design.sample_from_policy --config-name sample
"""
import os
import json
import secrets
from pathlib import Path

import torch
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm


def load_policy_from_checkpoint(ckpt_path, overrides=None):
    """
    Load a PolicyMPNN from a training checkpoint, applying optional overrides
    to the stored config (e.g. batch_size, temperature, pdb).
    """
    from cleo.design.utils.policy import PolicyMPNN

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_cfg = OmegaConf.create(ckpt["config"])

    if overrides:
        for key, val in overrides.items():
            if val is not None:
                OmegaConf.update(train_cfg, key, val)

    train_cfg.checkpoint_path = ckpt_path

    policy = PolicyMPNN(train_cfg)
    return policy, ckpt.get("step"), ckpt.get("reward")


def split_into_fragments(sequences, fragment_bounds):
    """
    Split full-length sequences into fragments defined by [start, end] bounds (inclusive).

    Returns a dict keyed by fragment number (as string):
        {"1": [(name, seq), ...], "2": [...], ...}

    Fragment names follow the convention: {frag_num}.{index}.{hex_id}
    where frag_num is a 1-indexed integer. Downstream tools (e.g.
    dna_fragment_design.py) parse the first dot-delimited token as the
    integer fragment number, so this prefix must always be a bare integer.
    """
    fragment_dict = {str(i + 1): [] for i in range(len(fragment_bounds))}

    for seq_idx, seq in enumerate(sequences):
        for frag_idx, (start, end) in enumerate(fragment_bounds):
            frag_seq = seq[start:end + 1]
            fragment_dict[str(frag_idx + 1)].append(frag_seq)

    deduped = {}
    for frag_num, frag_seqs in fragment_dict.items():
        seen = set()
        unique = []
        for seq in frag_seqs:
            if seq not in seen:
                seen.add(seq)
                name = f"{frag_num}.{len(unique):04d}.{secrets.token_hex(4)}"
                unique.append((name, seq))
        deduped[frag_num] = unique

    return deduped


def write_fasta(sequences, path, names=None):
    """Write sequences to a FASTA file."""
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            name = names[i] if names else f"seq_{i:06d}"
            f.write(f">{name}\n{seq}\n")


def write_fragment_dict_json(fragment_dict, path):
    """Write the fragment dictionary as JSON.

    Args:
        fragment_dict: Dict mapping fragment number (str) to list of
            ``(name, sequence)`` tuples.
        path: Output JSON file path.
    """
    serializable = {k: list(v) for k, v in fragment_dict.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def write_fragment_fasta(fragment_dict, output_dir, prefix):
    """Write one FASTA file per fragment region.

    Args:
        fragment_dict: Dict mapping fragment number (str) to list of
            ``(name, sequence)`` tuples.
        output_dir: Directory to write FASTA files into.
        prefix: Filename prefix (e.g. ``"{prefix}_{frag_num}.fasta"``).
    """
    for frag_name, entries in fragment_dict.items():
        path = os.path.join(output_dir, f"{prefix}_{frag_name}.fasta")
        with open(path, "w") as f:
            for name, seq in entries:
                f.write(f">{name}\n{seq}\n")


_CONFIG_DIR = str(Path(__file__).resolve().parent / "../../../config/design")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="sample")
def sample_from_policy(cfg):
    """Hydra entrypoint: sample sequences from policy checkpoints and write outputs."""
    os.makedirs(cfg.output_dir, exist_ok=True)

    overrides = {}
    if cfg.get("batch_size") is not None:
        overrides["batch_size"] = cfg.batch_size
    if cfg.get("temperature") is not None:
        overrides["temperature"] = cfg.temperature
    if cfg.get("pdb") is not None:
        overrides["pdb"] = cfg.pdb

    all_sequences = []
    seq_names = []

    for ckpt_path in cfg.checkpoints:
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

        print(f"\nLoading checkpoint: {ckpt_path}")
        policy, step, reward = load_policy_from_checkpoint(ckpt_path, overrides)
        print(f"  Step: {step}, Reward: {reward:.4f}")

        sequences = policy.sample_from_policy(cfg.num_batches)
        print(f"  Sampled {len(sequences)} sequences")

        run_name = os.path.basename(os.path.dirname(ckpt_path))
        for i, seq in enumerate(sequences):
            name = f"{run_name}.step{step:04d}.{i:04d}.{secrets.token_hex(4)}"
            seq_names.append(name)
        all_sequences.extend(sequences)

    # always write full-sequence FASTA
    fasta_path = os.path.join(cfg.output_dir, f"{cfg.output_name}.fasta")
    write_fasta(all_sequences, fasta_path, names=seq_names)
    print(f"\nWrote {len(all_sequences)} full sequences to {fasta_path}")

    # fragment splitting
    fragment_bounds = cfg.get("fragment_bounds")
    if fragment_bounds is not None:
        fragment_bounds = [list(fb) for fb in fragment_bounds]
        fragment_dict = split_into_fragments(all_sequences, fragment_bounds)

        json_path = os.path.join(cfg.output_dir, f"{cfg.output_name}_fragments.json")
        write_fragment_dict_json(fragment_dict, json_path)

        write_fragment_fasta(fragment_dict, cfg.output_dir, cfg.output_name)

        total_frags = sum(len(v) for v in fragment_dict.values())
        print(f"Wrote {total_frags} unique fragments across {len(fragment_bounds)} regions to {cfg.output_dir}")
        for frag_name, entries in fragment_dict.items():
            print(f"  {frag_name}: {len(entries)} unique sequences")

    print("\nDone.")


if __name__ == "__main__":
    sample_from_policy()
