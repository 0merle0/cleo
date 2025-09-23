import sys, os, json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra
from ensemble import Ensemble
import pdb_util
from fragment_util import get_all_sequences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def featurize_sequences(seqs):
    ret = torch.tensor([[pdb_util.aa12num[x] for x in seq] for seq in seqs], dtype=torch.long)
    ret = torch.nn.functional.one_hot(ret, num_classes=20).float()
    return ret.reshape(ret.shape[0], ret.shape[1] * ret.shape[2])

@hydra.main(version_base=None, config_path="./config")
def main(cfg):

    # load all sequences
    with open(cfg.fragment_dictionary, "r") as f:
        fragment_dictionary = json.load(f)

    fragment_dictionary = {int(k):v for k,v in fragment_dictionary.items()}

    all_seqs = get_all_sequences(fragment_dictionary)

    # load ckpt and config
    ckpt = torch.load(os.path.join(cfg.model_base_path, cfg.ckpt_name), map_location=torch.device('cpu'))
    model_config = OmegaConf.load(os.path.join(cfg.model_base_path,'config.yaml'))

    # load model
    model = Ensemble(model_config)
    model.load_state_dict(ckpt['state_dict'])
    model = model.eval()
    model = model.to(DEVICE)

    # run predictions
    pred_data = {
        "name": [],
        "sequence": [],
        "pred_mean": [],
        "pred_var": [],
    }

    num_batches = int(np.ceil(len(all_seqs) / cfg.batch_size))

    with torch.no_grad():
        # make greedy predictions for every sequence in the dataset
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

        
            # subset to top k predictions
            if cfg.top_k is not None:
                sorted_indices = np.argsort(pred_data["pred_mean"])[-cfg.top_k:]
                pred_data = {k: [v[i] for i in sorted_indices] for k, v in pred_data.items()}

    pred_df = pd.DataFrame(pred_data)
    

    outfolder = os.path.join(cfg.outdir, cfg.run_name)
    os.makedirs(outfolder, exist_ok=True)

    # save predictions to csv
    out_path = os.path.join(outfolder, "predictions.csv")
    pred_df.to_csv(out_path, index=False)

    # save config too
    config_path = os.path.join(outfolder, "config.yaml")
    OmegaConf.save(cfg, config_path)

    print(f"Predictions saved to {out_path}")

if __name__ == "__main__":
    main()