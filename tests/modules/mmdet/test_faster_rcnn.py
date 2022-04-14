import src.models.mmdet.faster_rcnn
from src.data import parsers
import pytorch_lightning as pl
import logging
from src.transforms import Adapter
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


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
