import hydra
import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from transformers import BertForSequenceClassification, BertConfig
from .bert.bert import BERT

from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MaxMetric

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import h5py


class Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.predictions = []
        self.targets = []

    def step(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        _ = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(
            "train/acc",
            self.train_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        _ = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "val/acc",
            self.val_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch, batch_idx: int):
        _, preds, targets = self.step(batch)
        logits = self.forward(batch)
        probs = nn.Softmax(dim=-1)(logits)

        # log val metrics
        _ = self.test_acc(preds, targets)
        self.log(
            "test/acc",
            self.test_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"preds": preds, "targets": targets, "probs": probs}

    def on_test_end(self):
        preds = np.concatenate(self.predictions, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        h5f = h5py.File("probs.h5", "w")
        h5f.create_dataset("probs", data=preds)
        h5f.create_dataset("targets", data=targets)
        h5f.close()

    def test_epoch_end(self, outputs):
        for elem in outputs:
            self.predictions.append(elem["probs"].cpu().numpy())
            self.targets.append(elem["targets"].cpu().numpy())

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log(
            "val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True
        )

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(params=self.parameters(), lr=0.001)


class ClassifierHugging(Classifier):
    def __init__(self):
        super().__init__()

        configuration = BertConfig()
        configuration.hidden_size = 8
        configuration.num_hidden_layers = 3
        configuration.num_attention_heads = 4
        configuration.intermediate_size = 8
        configuration.position_embedding_type = "relative_key"

        configuration.num_labels = 3
        self.bert = BertForSequenceClassification(configuration)

        # 21 = track variables, 2=jet pt and eta
        # TODO do better
        self.embed = torch.nn.Linear(21 + 0, configuration.hidden_size)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.val_acc_best = MaxMetric()

        # self.save_hyperparameters()

    def step(self, batch):
        x, y = batch

        y = torch.argmax(batch[1], dim=-1)
        out = self.forward(batch)
        preds = torch.argmax(out.logits, dim=1)
        return out.loss, preds, y

    def forward(self, batch):

        embeds = self.embed(batch[0])

        labels = torch.argmax(batch[1], dim=-1)
        out = self.bert(inputs_embeds=embeds, labels=labels)

        return out


class ClassifierCustom(Classifier):
    def __init__(self, hidden_size=32, depth=6):
        super(ClassifierCustom, self).__init__()

        self.bert = BERT(hidden=hidden_size, n_layers=depth, attn_heads=8, dropout=0.1)
        self.to_logits = nn.Linear(hidden_size, 3)

        # 21 = track variables, 2=jet pt and eta
        # TODO do better
        self.embed = torch.nn.Linear(21 + 2, hidden_size)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.val_acc_best = MaxMetric()

        # self.save_hyperparameters()

    def step(self, batch):
        x, y = batch

        y = torch.argmax(batch[1], dim=-1)
        out = self.forward(batch)
        preds = torch.argmax(out, dim=1)
        loss = F.cross_entropy(out, y)
        return loss, preds, y

    def forward(self, batch):

        embeds = self.embed(batch[0])

        # labels = torch.argmax(batch[1], dim=-1)
        out = self.bert(embeds)
        return self.to_logits(out)
