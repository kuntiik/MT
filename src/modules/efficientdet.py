import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import MaxMetric
import torch
from torchmetrics.detection import MeanAveragePrecision

import src.models.efficientdet as efficientdet
from torch.optim import Adam, SGD

from src.core.convertions import preds2dicts


class EfficientDetModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            learning_rate,
            optimizer,
            scheduler_patience=10,
            scheduler_factor=0.2,
            weight_decay=1e-6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.max_map50 = MaxMetric()
        self.mAP = MeanAveragePrecision()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (xb, yb), records = batch
        preds = self(xb, yb)
        loss = efficientdet.loss_fn(preds, yb)
        for k, v in preds.items():
            self.log(f"train/{k}", v)
        return loss

    def validation_step(self, batch, batch_idx):
        (xb, yb), records = batch
        with torch.no_grad():
            raw_preds = self(xb, yb)
            preds = efficientdet.convert_raw_predictions(
                (xb, yb), raw_preds["detections"], records, detection_threshold=0.0
            )
            loss = efficientdet.loss_fn(raw_preds, yb)
            preds_torch, targets_torch = preds2dicts(preds)
            self.mAP(preds_torch, targets_torch)

        for k, v in raw_preds.items():
            if "loss" in k:
                self.log(f"val/{k}", v)

    def predict_step(self, batch, batch_idx):
        (xb, yb), records = batch
        with torch.no_grad():
            raw_preds = self(xb, yb)
            preds = efficientdet.convert_raw_predictions(
                (xb, yb),
                raw_preds["detections"],
                records,
                detection_threshold=0.001,
                # nms_iou_threshold=0.6,
            )
        return preds

    def predict_batch(self, batch, batch_idx):
        (xb, yb), records = batch
        with torch.no_grad():
            raw_preds = self(xb, yb)
            preds = efficientdet.convert_raw_predictions(
                (xb, yb),
                raw_preds["detections"],
                records,
                detection_threshold=0.001,
                # nms_iou_threshold=0.6,
            )
        return preds

    def validation_epoch_end(self, outs):
        mAP_dict = self.mAP.compute()
        self.log_dict(mAP_dict)
        self.finalize_metrics()

    def on_epoch_end(self):
        self.mAP.reset()

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=self.hparams.scheduler_factor,
                patience=self.hparams.scheduler_patience,
            ),
            "monitor": "val/loss",
            "interval": "epoch",
            "name": "lr",
        }
        return [optimizer], [scheduler]
