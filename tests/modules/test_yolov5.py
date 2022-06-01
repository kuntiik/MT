import pytest
import hydra
import pytorch_lightning as pl

# @pytest.mark.parametrize('module', ['yolov5'])
from src.datamodules.dental import DentalCaries
from src.transforms import TransformsComposer


def test_training_pipeline_cpu():
    with hydra.initialize(config_path='../../configs'):
        cfg = hydra.compose(config_name='train.yaml', overrides=['datamodule.num_workers=0', "module=yolov5"])

    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, fast_dev_run=True, gpus=1, limit_predict_batches=2)
    transforms: TransformsComposer = hydra.utils.instantiate(cfg.transforms, _recursive_=False)
    train_tfms, val_tfms = transforms.train_val_transforms()
    dm: DentalCaries = hydra.utils.instantiate(cfg.datamodule, train_transforms=train_tfms, val_transforms=val_tfms)
    dm.setup()
    model = hydra.utils.instantiate(cfg.module.model)
    test_prediction = trainer.predict(model, dm.predict_dataloader('test'))
    val_prediction = trainer.predict(model, dm.predict_dataloader('val'))
    train_prediction = trainer.predict(model, dm.predict_dataloader('train'))
