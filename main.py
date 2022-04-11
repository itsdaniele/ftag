import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
)

from ftag.model.classifier import ClassifierHugging, ClassifierCustom

logger = logging.getLogger(__name__)

from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="ftag")


@hydra.main(config_path="configs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(1234)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    data_module = hydra.utils.instantiate(cfg.data)

    # Instantiate all modules specified in the configs
    model = ClassifierCustom()

    # Let hydra manage direcotry outputs
    # tensorboard = pl.loggers.TensorBoardLogger(save_dir=".", default_hp_metric=False)
    callbacks = [
        pl.callbacks.ModelCheckpoint(every_n_train_steps=10),
        # pl.callbacks.EarlyStopping(monitor="val/acc", patience=50),
    ]

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=wandb_logger, _convert_="partial"
    )

    trainer.fit(model, datamodule=data_module)
    #trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
