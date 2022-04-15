__all__ = ["MMDetModelAdapter"]

from abc import abstractmethod

import pytorch_lightning as pl
import torch.nn as nn
from typing import List
import torch
from torch.optim import Adam

from src.metrics.coco_metric import COCOMetric, COCOMetricType
from src.metrics.common import Metric
from src.models.mmdet.common.prediction import convert_raw_predictions
from torchmetrics import MaxMetric


class MMDetModelAdapter(pl.LightningModule):

    def __init__(self, model: nn.Module, learning_rate=1e-4, weight_decay=1e-6):
        super().__init__()
        self.save_hyperparameters()
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

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
            ),
            "monitor": "val/loss",
            "interval": "epoch",
            "name": "lr",
        }
        return [optimizer], [scheduler]
        # return optimizer

    def validation_epoch_end(self, outs):
        self.finalize_metrics()


    def finalize_metrics(self) -> None:
        metric_logs = self.map.finalize()
        for k, v in metric_logs.items():
            for entry in self.metrics_keys_to_log_to_prog_bar:
                if entry[0] == k:
                    self.log(entry[1], v, prog_bar=True)
                    self.max_map50(v)
                    # self.log(f"{self.map.name}/{k}", v)
                    self.log("max_map_50", self.max_map50.compute())
                else:
                    self.log(f"val/{k}", v)
