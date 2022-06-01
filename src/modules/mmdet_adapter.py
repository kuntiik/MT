__all__ = ["MMDetModelAdapter"]

from abc import abstractmethod

import pytorch_lightning as pl
import torch.nn as nn
from typing import List
import torch
from torch.optim import Adam, AdamW, SGD

from src.metrics.coco_metric import COCOMetric, COCOMetricType
from src.metrics.common import Metric
from src.models.mmdet.common.prediction import convert_raw_predictions
from torchmetrics import MaxMetric


class MMDetModelAdapter(pl.LightningModule):

    def __init__(self, model: nn.Module, learning_rate=1e-4, weight_decay=1e-6, optimizer='adamW'):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.max_map50 = MaxMetric()
        self.map = COCOMetric(metric_type=COCOMetricType.bbox)
        self.metrics_keys_to_log_to_prog_bar = [("map_50", "val/map_50")]

    def convert_raw_predictions(self, batch, raw_preds, records):
        return convert_raw_predictions(batch=batch, raw_preds=raw_preds, records=records, detection_threshold=0.0
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        data, samples = batch

        outputs = self.model.train_step(data=data, optimizer=None)

        for k, v in outputs["log_vars"].items():
            self.log(f"train/{k}", v)

        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        data, records = batch

        outputs = self.model.train_step(data=data, optimizer=None)
        raw_preds = self.model.forward_test(
            imgs=[data["img"]], img_metas=[data["img_metas"]]
        )

        preds = self.convert_raw_predictions(
            batch=data, raw_preds=raw_preds, records=records
        )
        self.map.accumulate(preds)

        for k, v in outputs["log_vars"].items():
            self.log(f"val/{k}", v)

    def test_step(self, batch, batch_idx):
        data, records = batch
        outputs = self.model.train_step(data=data, optimizer=None)
        raw_preds = self.model.forward_test(
            imgs=[data["img"]], img_metas=[data["img_metas"]]
        )
        preds = self.convert_raw_predictions(
            batch=data, raw_preds=raw_preds, records=records
        )
        self.map.accumulate(preds)
        for k, v in outputs["log_vars"].items():
            self.log(f"test/{k}", v)

    def test_epoch_end(self, outs):
        self.finalize_metrics(stage='test')

    def validation_epoch_end(self, outs):
        self.finalize_metrics(stage='val')

    def predict_step(self, batch, batch_idx):
        xb, records = batch
        raw_preds = self(return_loss=False, rescale=False, **xb)
        preds = self.convert_raw_predictions(batch=xb, raw_preds=raw_preds, records=records)
        self.map.accumulate(preds)
        return preds

    def on_predict_epoch_end(self, results) -> None:
        self.finalize_metrics_pred(stage='pred')

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adamW":
            optimizer = AdamW(
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
        # scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer=optimizer,
        #         factor=self.hparams.scheduler_factor,
        #         patience=self.hparams.scheduler_patience,
        #     ),
        #     "monitor": "val/loss",
        #     "interval": "epoch",
        #     "name": "lr",
        # }
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 70, 5e-8),
            "interval": "epoch",
            "name": "lr"
        }
        return [optimizer], [scheduler]

    def finalize_metrics(self, stage='val') -> None:
        metric_logs = self.map.finalize()
        for k, v in metric_logs.items():
            for entry in self.metrics_keys_to_log_to_prog_bar:
                if entry[0] == k:
                    self.log(f"{stage}/map_50", v, prog_bar=True)
                    self.max_map50(v)
                    self.log("max_map_50", self.max_map50.compute())
                else:
                    self.log(f"{stage}/{k}", v)

    def finalize_metrics_pred(self, stage='val') -> None:
        metric_logs = self.map.finalize()
        for k, v in metric_logs.items():
            for entry in self.metrics_keys_to_log_to_prog_bar:
                if entry[0] == k:
                    print(v)
