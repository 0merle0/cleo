"""Hydra entrypoint for training ensemble surrogate models.

Trains an ensemble of MLP (or Potts/conv1d) models on sequence-function data
using Gaussian NLL loss. Supports optional validation with label-based splits
and saves periodic checkpoints with CSV logging.

Usage:
    python -m cleo.optimize.train_surrogate --config-name momi_base
"""
import os
import datetime
from pathlib import Path

from omegaconf import OmegaConf
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from cleo.optimize.utils.ensemble import Ensemble
from cleo.optimize.utils.train_data import SequenceFunctionDataset, FragmentDataModule


_CONFIG_DIR = str(Path(__file__).resolve().parent / "../../../config/optimize")


@hydra.main(version_base=None, config_path=_CONFIG_DIR)
def train_surrogate(cfg):
    """Train an ensemble surrogate model from a sequence-function dataset."""

    OmegaConf.set_struct(cfg, False)
    now = datetime.datetime.now()
    datetime_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    loggers = None
    callbacks = []
    callbacks.append(pl.callbacks.RichModelSummary(max_depth=2))
    callbacks.append(pl.callbacks.RichProgressBar())

    if not cfg.debug:
        ckpt_dir = f"./ckpt/{cfg.run_name}/{cfg.run_name}.{datetime_str}"
        cfg.model.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        OmegaConf.save(cfg, f"{ckpt_dir}/config.yaml")

        loggers = []
        loggers.append(CSVLogger(
                save_dir=f"{ckpt_dir}/csv_logs",
                name=cfg.run_name,
                flush_logs_every_n_steps=1000,
            )
        )

        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                save_last=True,
                dirpath=ckpt_dir,
                monitor=cfg.checkpointer.monitor if cfg.data.validation_mode else None,
                mode=cfg.checkpointer.mode,
            )
        )
    else:
        cfg.data.num_workers = 1

    datamodule = FragmentDataModule(cfg)
    model = Ensemble(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    train_surrogate()
