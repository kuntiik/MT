from hydra.utils import instantiate
import pytorch_lightning as pl

import src.transforms.transforms_composer
from src.datamodules.dental_caries import DentalCaries
from src.transforms import TransformsComposer


def test_training_cpu(cfg):
    trainer: pl.Trainer = instantiate(cfg.trainer, fast_dev_run=True)
    transforms: TransformsComposer = instantiate(cfg.transforms)
    train_tfms, val_tfms = transforms.train_val_transforms()
    dm: DentalCaries = instantiate(cfg.datamodule, train_transforms=train_tfms, val_transforms=val_tfms)
    model = instantiate(cfg.module)
    trainer.fit(model=model, datamodule=dm)