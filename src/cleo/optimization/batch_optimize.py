import sys, os, json
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
import hydra
from optimization_util import BatchUCBwithEntropy, opt_loop
from ensemble import Ensemble
from data_util import FragmentDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create df with results from candidate seqs
def create_results_df(candidate_seqs, model, dataset_cfg):


    seqs = [s for n,s in candidate_seqs]
    names = [n for n,s in candidate_seqs]

    data = pd.DataFrame({
        "name": names,
        "sequence": seqs,
    })

    # placeholder column for dataframe
    label_col = dataset_cfg.label_col
    data[label_col] = 0.

    # make dataset
    dataset = FragmentDataset(dataset_cfg, data)

    model.eval()

    # run through the dataset
    mu = []
    sigma = []

    for x, y in dataset:
        
        with torch.no_grad():
            output = model(x[None].to(DEVICE))
            mu.append(output["mu"].item())
            sigma.append(output["sigma"].item())

    data["mu"] = mu
    data["sigma"] = sigma


    # drop label column
    data = data.drop(columns=[label_col])

    return data


@hydra.main(version_base=None, config_path="./config")
def main(cfg):

    # save results
    out_path = os.path.join(cfg.outdir, cfg.run_name)
    os.makedirs(out_path, exist_ok=True)

    # load surrogate model
    surrogate_ckpt_path = os.path.join(cfg.surrogate_ckpt, "last.ckpt")
    surrogate_config_path = os.path.join(cfg.surrogate_ckpt, "config.yaml")
    surrogate_config = OmegaConf.load(surrogate_config_path)

    ckpt = torch.load(surrogate_ckpt_path, map_location=DEVICE)
    model = Ensemble(surrogate_config)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(DEVICE)
    print("Loaded surrogate model from", cfg.surrogate_ckpt)


    # create acqf 
    acqf = BatchUCBwithEntropy(
        model, 
        model_batch_size=cfg.acqf.model_batch_size, 
        gamma=cfg.acqf.gamma, 
        eps=cfg.acqf.eps
    )
    
    # load fragment dictionary
    with open(cfg.opt_loop.fragment_dictionary, "r") as f:
        fragment_dictionary = json.load(f)
    fragment_dictionary = {int(k):v for k,v in fragment_dictionary.items()}

    # get candidate seqs
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

    # save config
    cfg_path = os.path.join(out_path, "config.yaml")
    OmegaConf.save(cfg, cfg_path)

    if cfg.write_fasta:
        # write candidate sequences to fasta
        fasta_path = os.path.join(out_path, "candidate_seqs.fasta")
        fasta_lines = [f">{n}\n{s}\n" for n,s in candidate_seqs]
        with open(fasta_path, "w") as f:
            f.writelines(fasta_lines)

    # create results dataframe
    results_df = create_results_df(candidate_seqs, model, surrogate_config.data.dataset_cfg)

    # save to csv
    csv_path = os.path.join(out_path, "candidates.csv")
    results_df.to_csv(csv_path, index=False)

    # save policy
    policy_path = os.path.join(out_path, "policy.pt")
    torch.save(policy, policy_path)

    print(f"Optimization completed. Results saved to {out_path}")

if __name__ == "__main__":
    main()
