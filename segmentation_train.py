from src.datamodules.dental_caries.dental_restorations import COCOSegmentation, CariesRestorationDatamodule
from segmentation.modules.segmentation import SegmentationModule
import albumentations as A
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pathlib import Path
import pytorch_lightning


def train():
    annotations_path = Path('/datagrid/personal/kuntluka/dental_restorations/annotations.json')
    imgs_path = Path('/datagrid/personal/kuntluka/dental_restorations/images')
    logger = WandbLogger(project='Segmentation', job_type='train', log_model=False)

    callbacks = [pytorch_lightning.callbacks.ModelCheckpoint(monitor='val/iou', save_top_k=1,
                                                             mode='max',
                                                             filename="epoch_{epoch:03d}_{val/iou:.3f}",
                                                             auto_insert_metric_name=False,
                                                             save_weights_only=True
                                                             ),
                 pytorch_lightning.callbacks.LearningRateMonitor(logging_interval='epoch')]
    trainer = Trainer(gpus=1, min_epochs=30, max_epochs=50, logger=logger, callbacks=callbacks)
    # transforms = A.Compose[A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
    dm = CariesRestorationDatamodule(imgs_path, annotations_path, batch_size=1, num_workers=8)
    model = SegmentationModule(num_classes=2, lr=1e-3, num_layers=6)

    trainer.fit(model, dm)


if __name__ == '__main__':
    train()
