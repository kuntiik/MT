from pytorch_lightning import LightningModule
from pytorch_toolbelt.losses import JaccardLoss, BinaryFocalLoss, BinarySoftF1Loss
from torch.optim import AdamW
import torch
from torchmetrics import MeanMetric
import cv2
import numpy as np

EPSILON = 1e-15


def binary_mean_iou(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    output = (logits > 0).int()
    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)
    intersection = (targets * output).sum()
    union = targets.sum() + output.sum() - intersection
    result = (intersection + EPSILON) / (union + EPSILON)
    return result

class SegmentationModuleTorch(LightningModule):
    def __init__(self, model, learning_rate, weight_decay):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.losses = [('jaccard', 0.1, JaccardLoss(mode='binary', from_logits=True)),
                       ('focal', 0.9, BinaryFocalLoss())]
        self.model = model
        self.val_iou = MeanMetric()
        self.train_loss = MeanMetric()

    def forward(self, batch) :
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
            ),
            "monitor": "val/iou",
            "interval": "epoch",
            "name": "lr",
            "patience": 5
        }
        return [optimizer], [scheduler]
        # return [optimizer]

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        yb = yb.unsqueeze(1)
        logits = self.forward(xb)
        total_loss = 0
        for loss_name, loss_weight, loss in self.losses:
            loss_from_gt = loss(logits, yb)
            total_loss += loss_weight * loss_from_gt
            self.log(f"train/{loss_name}", loss_from_gt)
        self.train_loss.update(total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        yb = yb.unsqueeze(1)
        logits = self.forward(xb)
        for loss_name, _, loss in self.losses:
            self.log(f"val/{loss_name}", loss(logits, yb))
        self.val_iou.update(binary_mean_iou(logits, yb))

    def validation_epoch_end(self, outs):
        self.log('val/iou', self.val_iou, prog_bar=True)
        self.log('train/loss', self.train_loss)

    def predict_step(self, batch, batch_idx):
        xb = batch['img']
        yb = batch['mask'].unsqueeze(1)


        out_images = []
        logits = (self.forward(xb) > 0).detach().cpu().numpy().astype(np.uint8)
        for image, img_name in zip(logits, batch['img_name']):
            #TODO fix this : remove the fixed size and add padding removal !!!!
            out_images.append((img_name ,cv2.resize(image[0], dsize=(1068, 847))))
        return out_images









