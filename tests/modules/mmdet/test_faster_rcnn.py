import pytest
import src.models.mmdet.faster_rcnn
from src.data import parsers
import pytorch_lightning as pl
import logging
from src.transforms import Adapter
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import hydra
from src.datamodules.dental import DentalCaries


@pytest.fixture()
def cfg_faster_rcnn():
    with hydra.initialize(config_path='../../../configs'):
        cfg = hydra.compose(config_name='train.yaml', overrides=['module=faster_rcnn'])
    return cfg

def test_faster_rcnn_instantiation(fridge_ds):
    train, val = fridge_ds
    backbone = src.models.mmdet.faster_rcnn.backbones.resnet50_fpn_1x()
    logging.getLogger('mmcv').disabled = True
    extra_args = {}
    extra_args['img_size'] = 384
    model = src.models.mmdet.faster_rcnn.model(backbone=backbone(pretrained=False), num_classes=5, )
    # t = Adapter([ab])
    train_dl = src.models.mmdet.faster_rcnn.train_dl(train, batch_size=2)
    val_dl = src.models.mmdet.faster_rcnn.valid_dl(val, batch_size=2)
    module = src.modules.MMDetModelAdapter(model)
    trainer = pl.Trainer(max_epochs=3)
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)

def test_instantiate_faster_rcnn(cfg_faster_rcnn):
    cfg = cfg_faster_rcnn
    # del cfg.datamodule._target_
    # DentalCaries(cfg.datamodule)

    dm = hydra.utils.instantiate(cfg.datamodule)
    module = hydra.utils.instantiate(cfg.module.model)
    trainer = hydra.utils.instantiate(cfg.trainer, fast_dev_run=True)
    trainer.fit(module, dm)