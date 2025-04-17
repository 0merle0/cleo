import os
import torch
from torch.utils.data import Dataset, DataLoader
import fragment_util
import pdb_util
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from icecream import ic


class FragmentDataset(Dataset):

    def __init__(self, cfg, df):
        """Simplified dataset."""

        super(FragmentDataset, self).__init__()
        self.cfg = cfg
        self.df = df
        self.fragment_dictionary = fragment_util.get_fragment_dictionary(
            cfg.fragment_csv
        )
        self.max_num_fragments = max(
            [len(self.fragment_dictionary[x]) for x in self.fragment_dictionary]
        )

    def featurize_inputs(self, index):
        """Featurize an index from dataframe."""

        # get info from dataframe
        row = self.df.iloc[index]
        name = row[self.cfg.name_col]
        seq = row[self.cfg.seq_col]
        label = row[self.cfg.label_col]
        label = torch.tensor(label)[None]  # add batch dimension

        if self.cfg.input_type == "embedding":
            # expecting directory structure of:
            #       embeddings_folder/frag1_frag2...fragNsub/frag1_frag2_frag3_...fragN.pt"
            name_split = name.split(self.cfg.split_fragments_on)
            subfolder = self.cfg.path_connecting_variable.join(
                name_split[: self.cfg.num_fragments_in_subfolder]
            )
            file_name = (
                self.cfg.path_connecting_variable.join(
                    name_split[: self.cfg.num_fragments]
                )
                + ".pt"
            )
            emb_path = os.path.join(self.cfg.path_to_embeddings, subfolder, file_name)
            assert os.path.exists(
                emb_path
            ), f"Embedding path does not exist: {emb_path}"
            input_feat = torch.load(emb_path, map_location="cpu")
            if self.cfg.embedding_key is not None:
                input_feat = input_feat[self.cfg.embedding_key]
            assert isinstance(
                input_feat, torch.Tensor
            ), f"Embedding must be a tensor, but got: {type(input_feat)}"

            if input_feat.shape[0] == 1:
                input_feat = input_feat[0]  # remove batch dimension if it exists

        elif self.cfg.input_type == "fragment":
            # using fragment featurization
            input_feat = fragment_util.featurize_fragments(
                [name],
                self.fragment_dictionary,
                to_split_on=self.cfg.split_fragments_on,
                num_fragments=len(self.fragment_dictionary),
            )
            input_feat = torch.nn.functional.one_hot(
                input_feat[0], num_classes=self.max_num_fragments
            )

        elif self.cfg.input_type == "sequence":
            # using one hot seqeunce encoding
            input_feat = torch.tensor([pdb_util.aa12num[x] for x in seq])
            input_feat = torch.nn.functional.one_hot(input_feat, num_classes=20)

        # NOTE: this is a quick patch to concatenate esm and mpnn embeddings
        elif self.cfg.input_type == "esm+mpnn":
            name_split = name.split(self.cfg.split_fragments_on)
            subfolder = self.cfg.path_connecting_variable.join(
                name_split[: self.cfg.num_fragments_in_subfolder]
            )
            file_name = (
                self.cfg.path_connecting_variable.join(
                    name_split[: self.cfg.num_fragments]
                )
                + ".pt"
            )

            esm_path = os.path.join(
                self.cfg.path_to_embeddings, "esm_embeddings", subfolder, file_name
            )
            mpnn_path = os.path.join(
                self.cfg.path_to_embeddings, "mpnn_embeddings", subfolder, file_name
            )
            assert os.path.exists(
                esm_path
            ), f"Embedding path does not exist: {esm_path}"
            assert os.path.exists(
                mpnn_path
            ), f"Embedding path does not exist: {mpnn_path}"

            esm_input_feat = torch.load(esm_path, map_location="cpu")[
                1:-1
            ]  # remove irrelevant tokens
            mpnn_input_feat = torch.load(mpnn_path, map_location="cpu")
            mpnn_input_feat = mpnn_input_feat["h_V"][0]

            input_feat = torch.cat([esm_input_feat, mpnn_input_feat], dim=-1)

        elif self.cfg.input_type == "vae_latent":
            latent_path = os.path.join(
                self.cfg.path_to_embeddings, self.cfg.subfolder, f"{self.cfg.subfolder}_{name}.pt"
            )
            assert os.path.exists(
                latent_path
            ), f"VAE latent path does not exist: {latent_path}"
            input_feat = torch.load(latent_path, map_location="cpu")
            
            # sample random index from the latent
            index_to_use = torch.randperm(input_feat.shape[0])[0]
            input_feat = input_feat[index_to_use]

        else:
            raise Exception(f"Input type not recognized: {self.cfg.input_type}")

        if self.cfg.input_shape == "BD":
            input_feat = input_feat.flatten()

        return input_feat.float(), label.float()  # set to float32

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.featurize_inputs(index)


class FragmentDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(FragmentDataModule, self).__init__()
        self.full_cfg = cfg
        self.cfg = cfg.data

        # only init val loader if in validation mode
        if self.cfg.validation_mode:
            self.val_dataloader = self._val_dataloader

    def get_train_val_split(self, df):
        """Split data into train and val."""

        df_val = None
        if self.cfg.validation_mode == "random":
            logging.info("Random validation")

            df_train, df_val = train_test_split(
                df,
                test_size=self.cfg.val_size,
                random_state=self.cfg.seed,
            )

        elif self.cfg.validation_mode == "top-k":
            logging.info("Top k validation")
            df_val = df.sort_values(
                by=self.cfg.dataset_cfg.label_col, ascending=False
            ).head(self.cfg.top_k_validation)
            df_train = df[~df.index.isin(df_val.index)]

        elif self.cfg.validation_mode == "label":
            logging.info("Label validation")
            df_train = df[~df[self.cfg.val_label]]
            df_val = df[df[self.cfg.val_label]]

        elif self.cfg.validation_mode is None:
            logging.info("Validation is turned off")
            df_train = df

        else:
            raise Exception(
                f"Validation mode not recognized: {self.cfg.validation_mode}"
            )

        # build datasets
        train_dataset = FragmentDataset(self.cfg.dataset_cfg, df_train)
        val_dataset = None
        if df_val is not None:
            val_dataset = FragmentDataset(self.cfg.dataset_cfg, df_val)
        return train_dataset, val_dataset

    def prepare_data(self):
        """Load local data on cpu."""
        df = pd.read_csv(self.cfg.dataset)
        self.train_dataset, self.val_dataset = self.get_train_val_split(df)

    def setup(self, stage):
        """Called on each DDP process."""
        # currently not being used at the moment
        pass

    def train_dataloader(self):
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

    def _val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
