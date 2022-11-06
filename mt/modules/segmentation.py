from typing import Tuple, Any

from torchmetrics import MeanMetric
from pytorch_lightning import LightningModule
import torch
from mt.utils.segmentation.losses import  *
import torch.nn as nn

# from segmentation.modules import
from mt.models.segmentation.Unet import UNet

class SegmentationModule(LightningModule):
    def __init__(
            self,
            learning_rate : float = 0.01,
            num_classes: int = 19,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            loss_type = 'cse_dice',
            threshold = 0.5
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss()
        self.iou_metrics = MeanMetric()
        self.non_empty_iou_metrics = MeanMetric()
        self.dice_metric = MeanMetric()
        self.loss_type = loss_type

        self.net = UNet(
            num_classes=num_classes,
            num_layers=num_layers,
            features_start=features_start,
            bilinear=bilinear,
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        out = self(img)
        if self.loss_type == 'cse_dice':
            loss_val = self.loss(out, mask) + soft_dice_loss(out, mask)
        elif self.loss_type == 'dice':
            loss_val = soft_dice_loss(out, mask)
        else:
            loss_val = self.loss(out, mask)
        self.log('train/loss', loss_val)
        return loss_val

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        out = self(img)
        loss_val = self.loss(out, mask) + soft_dice_loss(out, mask)
        self.log('val/loss', loss_val)
        self.iou_metrics.update(iou(out, mask))
        self.dice_metric.update(dice_values(out, mask))

    def validation_epoch_end(self, outs):
        self.log("val/iou", self.iou_metrics, prog_bar=True)
        self.log("val/dice_metric", self.dice_metric)

    # def predict_step(self, batch, batch_idx):
    #     out = self(batch)
    #     out_probs = F.softmax(out, dim=1)[0]
    #     # out = F.softmax()
    #     # tf = transforms.Compose()
    #     out = out_probs > 0.5
    #     return out
    # def validation_epoch_end(self, outputs):
    #     loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     self.log('val/epoch_loss', loss_val, prog_bar=True)
    #     log_dict = {"val_loss": loss_val}
    #     return {"log": log_dict, "val_loss": log_dict["val_loss"], "progress_bar": log_dict}
    def predict_step(self, batch, batch_idx):
        img, mask = batch
        out = F.softmax(self(img), dim=1)[:,1,...]
        return out

    # def on_predict_epoch_end(self, results) -> Tuple[Any, Any]:
    #     preds, masks = [], []
    #     print(type(results))
    #     print(type(results[0]))
    #     print(type(results[0][0]))
    #     for batch in results:
    #         for pred, mask in batch:
    #             preds.append(pred)
    #             masks.append(mask)
    #     preds = F.softmax(torch.cat(preds), dim=1)[:, 1, ...]
    #     masks = torch.cat(masks)
    #     results = (preds, masks)
    #     return preds, masks


    def test_step(self, batch, batch_idx):
        img, mask = batch
        out = self(img)
        iou_value = iou(out, mask, self.hparams.threshold)
        mask_sum = torch.sum(mask, dim=(1,2)) > 0
        self.non_empty_iou_metrics.update(iou_value[mask_sum])
        self.iou_metrics.update(iou_value)
        self.dice_metric.update(dice_values(out, mask, self.hparams.threshold))

    def test_epoch_end(self, outs):
        self.log('test/iou_non_empty', self.non_empty_iou_metrics)
        self.log('test/iou_metrics', self.iou_metrics)
        self.log('test/dice_metrics', self.dice_metric)


    def configure_optimizers(self):
        # opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.hparams.learning_rate)
        # sch = {"scheduler" : torch.optim.lr_scheduler.ReduceLROnPlateau(opt), "monitor" : 'val/loss'}
        sch = {"scheduler" : torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40, 1e-7), "interval" : 'epoch', "name": 'lr'}
        return [opt], [sch]
