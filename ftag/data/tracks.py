import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from .h5dataset import HDF5Dataset

logger = logging.getLogger(__name__)


class TRACKSDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = "", batch_size: int = 32, num_workers: int = 8, **kwargs
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        logger.info("Initializing TRACKS DataModule")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            tracks_full = HDF5Dataset(self.data_dir, batch_size=self.batch_size)
            split_size = [int(0.8 * len(tracks_full)), int(0.2 * len(tracks_full))]
            self.tracks_train, self.tracks_val = random_split(tracks_full, split_size)
            self.dims = tuple(self.tracks_train[0][0].shape)
        elif stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        return DataLoader(
            self.tracks_train, batch_size=None, num_workers=self.num_workers
        )

    # Double workers for val and test loaders since there is no backward pass and GPU computation is faster
    def val_dataloader(self):
        return DataLoader(
            self.tracks_val, batch_size=None, num_workers=self.num_workers * 2,
        )

    def test_dataloader(self):
        return self.val_dataloader()
