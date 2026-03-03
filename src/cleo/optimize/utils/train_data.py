"""Dataset and DataModule for training sequence-to-function surrogate models.

SequenceFunctionDataset handles one-hot encoding of amino acid sequences and
pairing them with scalar activity labels. FragmentDataModule wraps this into
a PyTorch Lightning DataModule with train/val splitting.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from cleo.optimize.utils.pdb_tools import aa12num
import pytorch_lightning as pl
import pandas as pd
import logging


class SequenceFunctionDataset(Dataset):
    """Dataset that pairs one-hot encoded amino acid sequences with scalar labels."""

    def __init__(self, cfg, df):
        super(SequenceFunctionDataset, self).__init__()
        self.cfg = cfg
        self.df = df

    def featurize_inputs(self, index):
        """One-hot encode the sequence at the given DataFrame index."""
        row = self.df.iloc[index]
        seq = row[self.cfg.seq_col]
        label = row[self.cfg.label_col]
        label = torch.tensor(label)[None]

        if self.cfg.input_type == "sequence":
            input_feat = torch.tensor([aa12num[x] for x in seq])
            input_feat = torch.nn.functional.one_hot(input_feat, num_classes=20)
        else:
            raise Exception(f"Input type not recognized: {self.cfg.input_type}")

        input_feat = input_feat.flatten()

        return input_feat.float(), label.float()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.featurize_inputs(index)


class FragmentDataModule(pl.LightningDataModule):
    """Lightning DataModule that loads a CSV dataset and splits into train/val."""

    def __init__(self, cfg):
        super(FragmentDataModule, self).__init__()
        self.full_cfg = cfg
        self.cfg = cfg.data

        if self.cfg.validation_mode:
            self.val_dataloader = self._val_dataloader

    def get_train_val_split(self, df):
        """Split DataFrame into train and val based on a boolean label column."""
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

        train_dataset = SequenceFunctionDataset(self.cfg.dataset_cfg, df_train)
        val_dataset = None
        if df_val is not None:
            val_dataset = SequenceFunctionDataset(self.cfg.dataset_cfg, df_val)
        return train_dataset, val_dataset

    def prepare_data(self):
        """Load dataset CSV from disk."""
        df = pd.read_csv(self.cfg.dataset)
        self.train_dataset, self.val_dataset = self.get_train_val_split(df)

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

    def _val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
