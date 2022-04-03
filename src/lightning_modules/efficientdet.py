import torchvision
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import MaxMetric
import torch
import models.efficientdet as efficientdet
from torch.optim import Adam, SGD

def ice_preds_to_dict(preds_records):
    preds = []
    target = []
    for record in preds_records:
        preds.append(
            dict(
                boxes=torch.Tensor([[*box.xyxy] for box in record.pred.detection.bboxes]),
                scores=torch.Tensor(record.pred.detection.scores),
                labels=torch.Tensor(record.pred.detection.label_ids),
            )
        )

        target.append(
            dict(
                boxes=torch.Tensor([[*box.xyxy] for box in record.ground_truth.detection.bboxes]),
                labels=torch.Tensor(record.ground_truth.detection.label_ids),
            )
        )
    return preds, target


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
        # self.metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
        self.metrics_keys_to_log_to_prog_bar = [("AP (IoU=0.50) area=all", "val/Pascal_VOC")]
        self.max_map50 = MaxMetric()
        # self.mAP = MeanAveragePrecision()

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
            preds_torch, targets_torch = ice_preds_to_dict(preds)

        # self.mAP(preds_torch, targets_torch)
        for k, v in raw_preds.items():
            if "loss" in k:
                self.log(f"val/{k}", v)
        # self.accumulate_metrics(preds)

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

    # def validation_epoch_end(self, outs):
        # mAP_dict = self.mAP.compute()
        # self.log_dict(mAP_dict)
        # self.finalize_metrics()

    # def on_epoch_end(self):
    #     self.mAP.reset()

    def accumulate_metrics(self, preds):
        for metric in self.metrics:
            metric.accumulate(preds=preds)

    def finalize_metrics(self) -> None:
        for metric in self.metrics:
            metric_logs = metric.finalize()
            for k, v in metric_logs.items():
                for entry in self.metrics_keys_to_log_to_prog_bar:
                    if entry[0] == k:
                        self.log(entry[1], v, prog_bar=True)
                        self.max_map50(v)
                        self.log(f"{metric.name}/{k}", v)
                        self.log("val/best_mAP_50", self.max_map50.compute())
                    else:
                        self.log(f"val/{metric.name}/{k}", v)

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
