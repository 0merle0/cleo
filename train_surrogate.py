import os
import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from ensemble import Ensemble
from data_util import FragmentDataModule

@hydra.main(version_base=None, config_path="./config", config_name="train_surrogate")
def train_surrogate(cfg):
    """Train surrogate model."""
    now = datetime.datetime.now()
    datetime_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    # setup datamodule
    datamodule = FragmentDataModule(cfg.data)

    # setup model
    model = Ensemble(cfg.model)

    logger = None
    callbacks = []
    callbacks.append(pl.callbacks.RichModelSummary(max_depth=2))
    callbacks.append(pl.callbacks.RichProgressBar()) 
    if not cfg.debug:
        # if not in debug mode, save config, set up logger and checkpointer
        ckpt_dir = f'./ckpt/{cfg.run_name}:{datetime_str}'
        os.makedirs(ckpt_dir, exist_ok=True)
        OmegaConf.save(cfg, f'{ckpt_dir}/config.yaml')

        logger = WandbLogger(
                                name=cfg.run_name,
                                project="itopt",
                                save_dir="./logs/wandb_logs",
                                log_model=False
                            )
        callbacks.append(pl.callbacks.ModelCheckpoint(
                                        dirpath=ckpt_dir, 
                                        monitor=cfg.ckpt_mointer, 
                                        mode=cfg.ckpt_mode
                                       )
                                    )

    # setup pytorch lightning trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=logger,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=cfg.val_check_interval,
        callbacks=callbacks,
    )

    # train model
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )

if __name__ == "__main__":
    train_surrogate()