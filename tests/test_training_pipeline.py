import hydra
import pytest
from hydra.utils import instantiate
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch

import mt.transforms.transforms_composer
from mt.datamodules.dental import DentalCaries
from mt.transforms import TransformsComposer


def test_training_cpu(cfg):
    trainer: pl.Trainer = instantiate(cfg.trainer, fast_dev_run=True)
    transforms: TransformsComposer = instantiate(cfg.transforms, _recursive_=False)
    train_tfms, val_tfms = transforms.train_val_transforms()
    dm: DentalCaries = instantiate(cfg.datamodule, train_transforms=train_tfms, val_transforms=val_tfms, num_workers=0)
    model = instantiate(cfg.module.model)
    trainer.fit(model=model, datamodule=dm)

def test_training_cpu_rectangle(cfg_rectangle):
    cfg = cfg_rectangle
    trainer: pl.Trainer = instantiate(cfg.trainer, fast_dev_run=True)
    transforms: TransformsComposer = instantiate(cfg.transforms, _recursive_=False)
    train_tfms, val_tfms = transforms.train_val_transforms()
    dm: DentalCaries = instantiate(cfg.datamodule, train_transforms=train_tfms, val_transforms=val_tfms, num_workers=0)
    model = instantiate(cfg.module.model)
    trainer.fit(model=model, datamodule=dm)

def test_training_cpu_multiple_epochs(cfg):
    trainer: pl.Trainer = instantiate(cfg.trainer, limit_train_batches=1, limit_val_batches=5, max_epochs=3, gpus=1)
    transforms: TransformsComposer = instantiate(cfg.transforms, _recursive_=False)
    train_tfms, val_tfms = transforms.train_val_transforms()
    dm: DentalCaries = instantiate(cfg.datamodule, train_transforms=train_tfms, val_transforms=val_tfms, num_workers=0)
    model = instantiate(cfg.module.model)
    trainer.fit(model=model, datamodule=dm)


@pytest.mark.parametrize('experiment', ['yolov5'])
def test_training_pipeline_cpu(experiment):
    with hydra.initialize(config_path='../configs'):
        experiment_string = f"experiment={experiment}"
        cfg = hydra.compose(config_name='train.yaml', overrides=['datamodule.num_workers=0', f"experiment={experiment}"])

    trainer: pl.Trainer = instantiate(cfg.trainer, fast_dev_run=True, gpus=1)
    transforms: TransformsComposer = instantiate(cfg.transforms, _recursive_=False)
    train_tfms, val_tfms = transforms.train_val_transforms()
    dm: DentalCaries = instantiate(cfg.datamodule, train_transforms=train_tfms, val_transforms=val_tfms)
    model = instantiate(cfg.module.model)
    trainer.fit(model=model, datamodule=dm)
