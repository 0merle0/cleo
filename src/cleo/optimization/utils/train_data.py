import os
import torch
from torch.utils.data import Dataset, DataLoader
import fragment_util
import pdb_util
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
import logging


class SequenceFunctionDataset(Dataset):

    def __init__(self, cfg, df):
        """
        Simplified dataset.
        """
        super(SequenceFunctionDataset, self).__init__()
        self.cfg = cfg
        self.df = df
        
    def featurize_inputs(self, index):
        """
        Featurize an index from dataframe.
        """
        # get info from dataframe
        row = self.df.iloc[index]
        seq = row[self.cfg.seq_col]
        label = row[self.cfg.label_col]
        label = torch.tensor(label)[None]  # add batch dimension

        # Currently only sequence supported, 
        # previously tried fragment representation or embedding
        # but none of them worked better thank simple one hot sequence encoding, so removed for simplicity
        if self.cfg.input_type == "sequence":
            # using one hot seqeunce encoding
            input_feat = torch.tensor([pdb_util.aa12num[x] for x in seq])
            input_feat = torch.nn.functional.one_hot(input_feat, num_classes=20)

        else:
            raise Exception(f"Input type not recognized: {self.cfg.input_type}")

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
        """
        Split data into train and val.
        """

        # Currently only options are having a 
        # label column that indicates train vs val, or no validation at all
        df_val = None
        if self.cfg.validation_mode == "label":
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
