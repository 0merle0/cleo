"""Hydra entrypoint for acquisition-function optimization over fragment space.

Loads a trained ensemble surrogate model, constructs a BatchUCB + diversity
acquisition function, and runs a REINFORCE-based optimization loop over the
combinatorial fragment space to propose candidate sequences for experimental
testing.

Usage:
    python -m cleo.optimize.batch_optimize --config-name momi_acqf_opt
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
import hydra
from cleo.optimize.utils.optimization import BatchUCBwithEntropy, opt_loop
from cleo.optimize.utils.ensemble import Ensemble
from cleo.optimize.utils.train_data import SequenceFunctionDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_results_df(candidate_seqs, model, dataset_cfg):
    """Run ensemble predictions on candidate sequences and return a DataFrame
    with columns: name, sequence, mu, sigma."""
    seqs = [s for n, s in candidate_seqs]
    names = [n for n, s in candidate_seqs]

    data = pd.DataFrame({
        "name": names,
        "sequence": seqs,
    })

    label_col = dataset_cfg.label_col
    data[label_col] = 0.

    dataset = SequenceFunctionDataset(dataset_cfg, data)

    model.eval()

    mu = []
    sigma = []

    for x, y in dataset:
        with torch.no_grad():
            output = model(x[None].to(DEVICE))
            mu.append(output["mu"].item())
            sigma.append(output["sigma"].item())

    data["mu"] = mu
    data["sigma"] = sigma

    data = data.drop(columns=[label_col])

    return data


_CONFIG_DIR = str(Path(__file__).resolve().parent / "../../../config/optimize")


@hydra.main(version_base=None, config_path=_CONFIG_DIR)
def main(cfg):

    out_path = os.path.join(cfg.outdir, cfg.run_name)
    os.makedirs(out_path, exist_ok=True)

    surrogate_ckpt_path = os.path.join(cfg.surrogate_ckpt, "last.ckpt")
    surrogate_config_path = os.path.join(cfg.surrogate_ckpt, "config.yaml")
    surrogate_config = OmegaConf.load(surrogate_config_path)

    ckpt = torch.load(surrogate_ckpt_path, map_location=DEVICE)
    model = Ensemble(surrogate_config)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE)
    print("Loaded surrogate model from", cfg.surrogate_ckpt)

    acqf = BatchUCBwithEntropy(
        model,
        model_batch_size=cfg.acqf.model_batch_size,
        gamma=cfg.acqf.gamma,
        eps=cfg.acqf.eps
    )

    with open(cfg.opt_loop.fragment_dictionary, "r") as f:
        fragment_dictionary = json.load(f)
    fragment_dictionary = {int(k): v for k, v in fragment_dictionary.items()}

    candidate_seqs, policy = opt_loop(
        acqf,
        fragment_dictionary,
        cfg.opt_loop.N,
        cfg.opt_loop.q,
        cfg.opt_loop.num_iter,
        cfg.opt_loop.lr,
        out_path,
        cfg.connector,
        DEVICE,
    )

    cfg_path = os.path.join(out_path, "config.yaml")
    OmegaConf.save(cfg, cfg_path)

    if cfg.write_fasta:
        fasta_path = os.path.join(out_path, "candidate_seqs.fasta")
        fasta_lines = [f">{n}\n{s}\n" for n, s in candidate_seqs]
        with open(fasta_path, "w") as f:
            f.writelines(fasta_lines)

    results_df = create_results_df(candidate_seqs, model, surrogate_config.data.dataset_cfg)

    csv_path = os.path.join(out_path, "candidates.csv")
    results_df.to_csv(csv_path, index=False)

    policy_path = os.path.join(out_path, "policy.pt")
    torch.save(policy, policy_path)

    print(f"Optimization completed. Results saved to {out_path}")

if __name__ == "__main__":
    main()
