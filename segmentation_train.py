from src.datamodules.dental_caries.dental_restorations import COCOSegmentation, CariesRestorationDatamodule
from segmentation.modules.segmentation import SegmentationModule
import albumentations as A
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pathlib import Path

def train():
    annotations_path = Path('/datagrid/personal/kuntluka/dental_rtg1/annotations.json')
    imgs_path = Path('/datagrid/personal/kuntluka/dental_rtg1/images')
    logger = WandbLogger(project='Segmentation', job_type='train', log_model=False)
    trainer = Trainer(gpus=1, max_epochs=10)
    # transforms = A.Compose[A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
    dm = CariesRestorationDatamodule(imgs_path, annotations_path, batch_size=1, num_workers=8)
    model = SegmentationModule(num_classes=2)

    trainer.fit(model, dm)



if __name__ == '__main__':
    train()