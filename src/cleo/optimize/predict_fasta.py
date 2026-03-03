"""Hydra entrypoint for running ensemble predictions on a FASTA file.

Loads a trained ensemble checkpoint, featurizes sequences with one-hot
encoding, runs batched inference, and saves per-sequence mean and variance
predictions to CSV.

Usage:
    python -m cleo.optimize.predict_fasta --config-name pred_fasta
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra
from cleo.optimize.utils.ensemble import Ensemble
from cleo.optimize.utils.pdb_tools import aa12num

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def featurize_sequences(seqs):
    """One-hot encode amino acid sequences into a flat feature tensor."""
    ret = torch.tensor([[aa12num[x] for x in seq] for seq in seqs], dtype=torch.long)
    ret = torch.nn.functional.one_hot(ret, num_classes=20).float()
    return ret.reshape(ret.shape[0], ret.shape[1] * ret.shape[2])


_CONFIG_DIR = str(Path(__file__).resolve().parent / "../../../config/optimize")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="pred_fasta")
def main(cfg):

    with open(cfg.fasta_path, "r") as f:
        lines = f.read().strip().split("\n")

    all_seqs = []
    for i in range(1, len(lines), 2):
        name = lines[i-1].lstrip(">")
        seq = lines[i].strip()
        all_seqs.append((name, seq))

    ckpt = torch.load(os.path.join(cfg.model_base_path, cfg.ckpt_name), map_location=torch.device('cpu'))
    model_config = OmegaConf.load(os.path.join(cfg.model_base_path, 'config.yaml'))

    model = Ensemble(model_config)
    model.load_state_dict(ckpt['state_dict'])
    model = model.eval()
    model = model.to(DEVICE)

    pred_data = {
        "name": [],
        "sequence": [],
        "pred_mean": [],
        "pred_var": [],
    }

    num_batches = int(np.ceil(len(all_seqs) / cfg.batch_size))

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            batch = all_seqs[batch_idx * cfg.batch_size : (batch_idx + 1) * cfg.batch_size]
            seqs = [s[1] for s in batch]
            names = [s[0] for s in batch]

            input_feat = featurize_sequences(seqs).to(DEVICE)
            out = model(input_feat)

            pred_data["name"].extend(names)
            pred_data["sequence"].extend(seqs)
            pred_data["pred_mean"].extend(out['mu'].tolist())
            pred_data["pred_var"].extend(out['sigma'].tolist())

    pred_df = pd.DataFrame(pred_data)

    outfolder = os.path.join(cfg.outdir, cfg.run_name)
    os.makedirs(outfolder, exist_ok=True)

    out_path = os.path.join(outfolder, "predictions.csv")
    pred_df.to_csv(out_path, index=False)

    config_path = os.path.join(outfolder, "config.yaml")
    OmegaConf.save(cfg, config_path)

    print(f"Predictions saved to {out_path}")

if __name__ == "__main__":
    main()
