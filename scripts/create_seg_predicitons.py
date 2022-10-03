import hydra
from pathlib import Path
import cv2
import torch
import numpy as np


with hydra.initialize(config_path='../configs'):
    cfg = hydra.compose(config_name='train',
                        overrides=['experiment=unet', 'trainer.gpus=1', 'datamodule.num_workers=0', 'module.model.model.encoder_name=resnet50'])
model = hydra.utils.instantiate(cfg.module.model)
pretrained_path = '/home.stud/kuntluka/MT/epoch_089_0.782.ckpt'
pretrained_model = torch.load(pretrained_path)
model.load_state_dict(pretrained_model["state_dict"])

dm = hydra.utils.instantiate(cfg.datamodule)
dm.setup(stage='predict')
trainer = hydra.utils.instantiate(cfg.trainer)
val_data = trainer.predict(model, dm.predict_dataloader('val'))
train_data = trainer.predict(model, dm.predict_dataloader('train'))

target_folder = Path('/home.stud/kuntluka/MT/data/seg_predictions')
target_folder.mkdir()
# for vaL_file in val_data:
for vd in val_data:
    file_name, img = vd[0]
    norm_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(str(target_folder / file_name), norm_image)

for td in train_data:
    file_name, img = td[0]
    norm_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(str(target_folder / file_name), norm_image)
