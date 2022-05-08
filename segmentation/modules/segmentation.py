import torch
from pytorch_lightning import LightningModule, seed_everything
from torch.nn import functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from segmentation.utils.losses import *
from torchmetrics import MeanMetric


# from segmentation.modules import
from segmentation.models.Unet import UNet

class SegmentationModule(LightningModule):
    def __init__(
            self,
            lr: float = 0.01,
            num_classes: int = 19,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss()
        self.iou_metrics = MeanMetric()
        self.dice_metric = MeanMetric()

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
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = self.loss(out, mask) + soft_dice_loss(out, mask)
        self.log('train/loss', loss_val)
        return loss_val

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
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

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        sch = {"scheduler" : torch.optim.lr_scheduler.ReduceLROnPlateau(opt), "monitor" : 'val/loss'}
        return [opt], [sch]