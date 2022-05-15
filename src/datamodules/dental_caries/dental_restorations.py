from typing import Tuple, Optional

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, random_split, DataLoader
from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

from pytorch_lightning import LightningDataModule
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class COCOSegmentation(Dataset):
    def __init__(self, img_root, ann_path, transforms=None, apply_transforms=True):
        super().__init__()
        # self.transforms = transforms
        self.img_root = img_root if type(img_root) == str else Path(img_root)
        self.coco = COCO(ann_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cat_id = self.coco.getCatIds(catNms=['restoration'])
        self.transforms = A.Compose(
            [*self._default_transforms((1068, 847))]) if transforms is None else A.Compose(
            [*transforms, *self._default_transforms((1068, 847))])
        if not apply_transforms:
            self.transforms = None
        # self.val_transforms = A.Compose([*self._default_transforms(1068, 847)])

    def _load_img(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]['file_name']
        # return np.asarray(Image.open(self.img_root / path).convert("RGB")).transpose(2,0,1)
        return np.asarray(Image.open(self.img_root / path).convert("RGB"))

    def _load_mask(self, id: int) -> Image.Image:
        targets = self.coco.loadAnns(self.coco.getAnnIds(imgIds=id, catIds=self.cat_id))
        img = self.coco.loadImgs(id)[0]
        mask = np.zeros((img['height'], img['width']))
        for i in range(len(targets)):
            mask = np.maximum(self.coco.annToMask(targets[i]), mask)
        return mask

    @property
    def dental_caries_statistics(self):
        return dict(mean=0.3669, std=0.2768)

    def _default_transforms(self, image_size):
        normalize_stat = self.dental_caries_statistics
        return [
            A.Normalize(mean=self.dental_caries_statistics["mean"], std=self.dental_caries_statistics["std"]),
            ToTensorV2()
        ]

    def __getitem__(self, index):
        id = self.ids[index]
        img = self._load_img(id)
        mask = self._load_mask(id)
        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img, mask = transformed['image'].float(), transformed['mask'].long()
        return img, mask

    def __len__(self) -> int:
        return len(self.ids)


class CariesRestorationDatamodule(LightningDataModule):
    def __init__(self, img_root, ann_path, batch_size, transforms=True, seed: int = 42, num_workers: int = 8,
                 data_split: Tuple[int, int, int] = [0.8, 0.2, 0.0]):
        super().__init__()
        self.save_hyperparameters()

        self.train_transforms = [
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.Affine(translate_percent=(-0.1, 0.1), p=0.5),
            A.GaussianBlur(blur_limit=(7, 31), p=0.3),
            A.RandomGamma(gamma_limit=(60, 140), p=0.3),
        ]

    def setup(self, stage : Optional[str] = None):
        tf, vf, tstf = self.hparams.data_split
        dataset = COCOSegmentation(self.hparams.img_root, self.hparams.ann_path, self.train_transforms, apply_transforms=self.hparams.transforms)
        tn, vn, tstn = int(tf * len(dataset)), int(vf * len(dataset)), int(tstf * len(dataset))
        tn -= ((tn + vn + tstn) - len(dataset))
        split = [tn, vn, tstn]
        self.train_ds, _, _ = random_split(dataset, split, generator=torch.Generator().manual_seed(self.hparams.seed))
        dataset = COCOSegmentation(self.hparams.img_root, self.hparams.ann_path, None,  apply_transforms=self.hparams.transforms)
        _, self.val_ds, self.test_ds = random_split(dataset, split,
                                                    generator=torch.Generator().manual_seed(self.hparams.seed))

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)
