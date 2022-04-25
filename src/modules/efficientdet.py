import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import MaxMetric
import torch
# from torchmetrics.detection import MeanAveragePrecision
# from mytorchmetrics.detection import MeanAveragePrecision
# from torchmetrics.detection.map import MAP

import src.models.efficientdet as efficientdet
from torch.optim import Adam, SGD
from effdet import DetBenchPredict, unwrap_bench

from src.core.convertions import preds2dicts
from src.metrics.coco_metric import COCOMetric, COCOMetricType


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
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.max_map50 = MaxMetric()
        # self.map = MeanAveragePrecision()
        self.map = COCOMetric(metric_type=COCOMetricType.bbox)
        self.metrics_keys_to_log_to_prog_bar = [("map_50", "val/map_50")]
        # self.mAP = torchmetrics.detection.map()

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
            self.map.accumulate(preds)
        for k, v in raw_preds.items():
            if "loss" in k:
                self.log(f"val/{k}", v)

    def test_step(self, batch, batch_idx):
        (xb, yb), records = batch
        with torch.no_grad():
            raw_preds = self(xb, yb)
            preds = efficientdet.convert_raw_predictions(
                (xb, yb), raw_preds["detections"], records, detection_threshold=0.0
            )
            loss = efficientdet.loss_fn(raw_preds, yb)
            self.map.accumulate(preds)
        for k, v in raw_preds.items():
            if "loss" in k:
                self.log(f"test/{k}", v)

    def test_epoch_end(self, outputs) -> None:
        self.finalize_metrics(stage='test')

    def on_predict_start(self) -> None:
        self.bench = DetBenchPredict(unwrap_bench(self.model)).eval().to(self.device)

    def predict_step(self, batch, batch_idx):
        # (xb, yb), records = batch
        (imgs, img_info), records = batch
        raw_preds = self.bench(x=imgs, img_info=img_info)
        preds = efficientdet.convert_raw_predictions(
            batch=batch,
            raw_preds=raw_preds,
            records=records,
            detection_threshold=0.001,
            # nms_iou_threshold=0.6,
        )
        return preds

    def validation_epoch_end(self, outs):
        # self.log_dict(self.map)
        # result = self.map.compute()
        # self.log_dict("val/", result)
        # self.map.reset()
        # self.map.finalize()
        self.finalize_metrics(stage='val')
    #
    # def on_epoch_end(self):
    #     self.mAP.reset()

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
