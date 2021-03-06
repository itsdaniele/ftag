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

import torch
from torch.utils.data import DataLoader


from ftag.model.classifier import ClassifierCustom

logger = logging.getLogger(__name__)

from pytorch_lightning.loggers import WandbLogger

import json
import os
from ftag.data.h5dataset import HDF5DatasetTest

from hydra.utils import get_original_cwd

wandb_logger = WandbLogger(project="ftag")


@hydra.main(config_path="configs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(1234)

    logger.info("\n" + OmegaConf.to_yaml(cfg))

    dm = hydra.utils.instantiate(cfg.data)
    dm.setup()
    val_loader = dm.train_dataloader()

    dataset = HDF5DatasetTest(
        file_path="/srv/beegfs/scratch/groups/dpnc/atlas/FTag/samples/gnn-samples/v9/hybrids/MC16d-inclusive_testing_ttbar_PFlow.h5",
        scale_dict_path="/srv/beegfs/scratch/groups/dpnc/atlas/FTag/samples/gnn-samples/v9/PFlow-scale_dict.json",
    )

    test_loader = DataLoader(dataset, batch_size=None, num_workers=32)

    # Instantiate all modules specified in the configs
    model = ClassifierCustom.load_from_checkpoint(
        checkpoint_path=os.path.join(
            get_original_cwd(),
            "outputs/2022-04-19/full_run/ftag/2ond1jhd/checkpoints/epoch=59-step=1431009.ckpt",
        )
    )

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=wandb_logger, _convert_="partial"
    )

    # ßtrainer.fit(model, datamodule=data_module)
    trainer.test(model, dataloaders=[val_loader])


if __name__ == "__main__":
    main()
