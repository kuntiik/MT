__all__ = ["YoloV5Module"]

import pytorch_lightning.callbacks
import torch
from torch.optim import Adam
from torchmetrics import MaxMetric
from yolov5.utils.loss import ComputeLoss
import pytorch_lightning as pl
import torch.nn as nn
import src.models.yolov5 as yolov5
# from mytorchmetrics.detection import MeanAveragePrecision

from src.core.convertions import preds2dicts
from src.metrics.coco_metric import COCOMetric, COCOMetricType

pytorch_lightning.callbacks.ModelCheckpoint

class YoloV5Module(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate, weight_decay):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        # self.map = MeanAveragePrecision()
        self.compute_loss = ComputeLoss(model)

        self.max_map50 = MaxMetric()
        self.map = COCOMetric(metric_type=COCOMetricType.bbox)
        self.metrics_keys_to_log_to_prog_bar = [("map_50", "val/map_50")]

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (xb, yb), _ = batch
        preds = self(xb)
        loss = self.compute_loss(preds, yb)[0]
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (xb, yb), records = batch

        inference_out, training_out = self(xb)
        preds = yolov5.convert_raw_predictions(
            batch=xb,
            raw_preds=inference_out,
            records=records,
            detection_threshold=0.001,
            nms_iou_threshold=0.6,
        )
        loss = self.compute_loss(training_out, yb)[0]
        preds_torch, targets_torch = preds2dicts(preds, self.device)
        # self.map.update(preds_torch, targets_torch)
        self.map.accumulate(preds)
        self.log('val/loss', loss)

    def validation_epoch_end(self, outs):
        self.finalize_metrics()

    def predict_step(self, batch, batch_idx):
        xb, records = batch
        raw_preds = self(xb)[0]
        return yolov5.convert_raw_predictions(
            batch=xb,
            raw_preds=raw_preds,
            detection_threshold=0,
            num_iou_threshold = 1,
            keep_images=False
        )



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
