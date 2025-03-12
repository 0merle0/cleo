import os
import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from ensemble import Ensemble
from data_util import FragmentDataModule


@hydra.main(version_base=None, config_path="./config")
def train_surrogate(cfg):
    """Train surrogate model."""

    OmegaConf.set_struct(cfg, False)
    now = datetime.datetime.now()
    datetime_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    loggers = None
    callbacks = []
    callbacks.append(pl.callbacks.RichModelSummary(max_depth=2))
    callbacks.append(pl.callbacks.RichProgressBar())

    if not cfg.debug:
        # if not in debug mode, save config, set up logger and checkpointer
        ckpt_dir = f"./ckpt/{cfg.run_name}/{cfg.run_name}.{datetime_str}"
        cfg.model.ckpt_dir = ckpt_dir  # add ckpt_dir to cfg
        os.makedirs(ckpt_dir, exist_ok=True)
        OmegaConf.save(cfg, f"{ckpt_dir}/config.yaml")

        loggers = []
        loggers.append(WandbLogger(
                name=cfg.run_name,
                project="combo_ab_filter",
                save_dir="./logs/wandb_logs",
                log_model=False,
            )
        )
        loggers.append(CSVLogger(
                save_dir=f"{ckpt_dir}/csv_logs",
                name=cfg.run_name,
                flush_logs_every_n_steps=1000, # NOTE: add this to the configs
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
        # set num workers to 1 for debugging
        cfg.data.num_workers = 1

    # setup datamodule
    datamodule = FragmentDataModule(cfg)

    # setup model
    model = Ensemble(cfg)

    # setup pytorch lightning trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    )

    # train model
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    train_surrogate()
