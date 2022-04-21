import torch
from pytorch_lightning import LightningModule, seed_everything
from torch.nn import functional as F

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

        self.net = UNet(
            num_classes=num_classes,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.binary_cross_entropy(out, mask)
        self.log('train/loss', loss_val)
        return loss_val
        # log_dict = {"train_loss": loss_val}
        # return {"loss": loss_val, "log": log_dict, "progress_bar": log_dict}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        # loss_val = F.cross_entropy(out, mask, ignore_index=250)
        loss_val = F.binary_cross_entropy(out, mask)
        self.log('val/loss', loss_val)
        # return {"val_loss": loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('val/epoch_loss', loss_val, prog_bar=True)
    #     log_dict = {"val_loss": loss_val}
    #     return {"log": log_dict, "val_loss": log_dict["val_loss"], "progress_bar": log_dict}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        return [opt], [sch]
