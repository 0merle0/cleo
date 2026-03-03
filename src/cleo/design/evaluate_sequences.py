"""
Evaluate sequences through a configurable metric pipeline.

Reads a FASTA file and runs each sequence through the same step-based
pipeline used during training (e.g. structure prediction, distance
calculations). Outputs a CSV with all computed metrics — useful for
scoring resampled fragment combinations before ordering.

Usage:
    python -m cleo.design.evaluate_sequences --config-name evaluate
"""

import os
from pathlib import Path

import pandas as pd
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.utils import get_method


def read_fasta(path):
    """Read a FASTA file and return lists of names and sequences."""
    names = []
    sequences = []
    current_name = None
    current_seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name is not None:
                    names.append(current_name)
                    sequences.append("".join(current_seq))
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

    if current_name is not None:
        names.append(current_name)
        sequences.append("".join(current_seq))

    return names, sequences


_CONFIG_DIR = str(Path(__file__).resolve().parent / "../../../config/design")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="evaluate")
def evaluate_sequences(cfg):

    assert cfg.input_fasta is not None, "input_fasta is required"
    assert os.path.exists(cfg.input_fasta), f"Input FASTA not found: {cfg.input_fasta}"

    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Reading sequences from {cfg.input_fasta}")
    names, sequences = read_fasta(cfg.input_fasta)
    print(f"  Found {len(sequences)} sequences")

    df = pd.DataFrame({
        "name": names,
        "sequence": sequences,
    })

    rundir = os.path.join(cfg.output_dir, "run")
    os.makedirs(rundir, exist_ok=True)

    print("\n********* running metric steps *********")
    for step in cfg.steps:
        fn = get_method(step.target_fn)
        step_name = step.name
        print(f"Running step: {step_name} using function: {step.target_fn}")
        with open_dict(step.cfg):
            step.cfg.rundir = rundir
            step.cfg.step = step_name
        df = fn(df, step.cfg, step_name=step_name)

    csv_path = os.path.join(cfg.output_dir, f"{cfg.output_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote metrics for {len(df)} sequences to {csv_path}")

    print(f"\nSummary:")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

    print("Done.")


if __name__ == "__main__":
    evaluate_sequences()
