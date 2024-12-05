import os
import datetime
from omegaconf import DictConfig, OmegaConf
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from ensemble import Ensemble
from data_util import FragmentDataModule

@hydra.main(version_base=None, config_path="./config")
def train_surrogate(cfg):
    """Train surrogate model."""
    now = datetime.datetime.now()
    datetime_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    
    logger = None
    callbacks = []
    callbacks.append(pl.callbacks.RichModelSummary(max_depth=2))
    callbacks.append(pl.callbacks.RichProgressBar())

    if not cfg.debug:

        # if not in debug mode, save config, set up logger and checkpointer
        logger = WandbLogger(
                                name=cfg.run_name,
                                project="itopt",
                                save_dir="./logs/wandb_logs",
                                log_model=False
                            )
        if not cfg.test_only.enabled:
            # Only set up checkpointing if we're training
            ckpt_dir = f'./ckpt/{cfg.run_name}/{cfg.run_name}:{datetime_str}'
            os.makedirs(ckpt_dir, exist_ok=True)
            OmegaConf.save(cfg, f'{ckpt_dir}/config.yaml')
            
            # set num workers to 1 for debugging
            # cfg.data.num_workers = 1
            callbacks.append(pl.callbacks.ModelCheckpoint(
                                            save_last=True,
                                            dirpath=ckpt_dir,
                                            monitor=cfg.checkpointer.monitor, 
                                            mode=cfg.checkpointer.mode
                                        )
                                        )

    # setup datamodule
    datamodule = FragmentDataModule(cfg.data)

    # setup model
    model = Ensemble(cfg.model)

    # setup pytorch lightning trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    )

    if cfg.test_only.enabled: # test-only mode, no training
        if cfg.test_only.checkpoint_path is None:
            raise ValueError("Checkpoint path must be provided for test-only mode")
        
        print(f"Running in test-only mode with checkpoint: {cfg.test_only.checkpoint_path}")
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.test_only.checkpoint_path
        )
    else:
        # train model
        trainer.fit(
            model=model,
            datamodule=datamodule,
        )

        # test model
        ckpt_path = None if cfg.debug else f"{ckpt_dir}/last.ckpt"
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path
        )

if __name__ == "__main__":
    train_surrogate()