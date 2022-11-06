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

from mt.transforms.albumentations_utils import resize_and_pad


class COCOSegmentationPrediction(Dataset):
    def __init__(self, img_root, ann_path, img_size):
        super().__init__()
        # img names is hot fix to get the name of the image from dataloader - TODO fix this by adopting icevision approach
        self.img_root = Path(img_root) if type(img_root) == str else img_root
        self.coco = COCO(ann_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cat_id = self.coco.getCatIds(catNms=['restoration'])
        img_size = img_size if img_size is not None else (1088, 864)
        self.transforms = A.Compose(self._default_transforms(img_size))

    def _load_img(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]['file_name']
        return np.asarray(Image.open(self.img_root / path).convert("RGB"))

    def _load_img_name(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]['file_name']
        return self.img_root / path

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
            *resize_and_pad(image_size),
            A.Normalize(mean=self.dental_caries_statistics["mean"], std=self.dental_caries_statistics["std"]),
            ToTensorV2()
        ]

    def __getitem__(self, index):
        id = self.ids[index]
        img = self._load_img(id)
        mask = self._load_mask(id)
        img_name =  str(Path(self._load_img_name(self.ids[index])).name)
        img_size = img.shape[:2]
        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img, mask = transformed['image'].float(), transformed['mask'].long()
        return {'img' : img, 'mask' : mask, 'img_name' : img_name, 'img_size' : img_size}

    def __len__(self) -> int:
        return len(self.ids)

class COCOSegmentation(Dataset):
    def __init__(self, img_root, ann_path, transforms=None, apply_transforms=True, img_names=False):
        super().__init__()
        # img names is hot fix to get the name of the image from dataloader - TODO fix this by adopting icevision approach
        self.img_names = img_names
        self.img_root = Path(img_root) if type(img_root) == str else img_root
        self.coco = COCO(ann_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cat_id = self.coco.getCatIds(catNms=['restoration'])
        self.transforms = transforms
        if not apply_transforms:
            self.transforms = None

    def _load_img(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]['file_name']
        return np.asarray(Image.open(self.img_root / path).convert("RGB"))

    def _load_img_name(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]['file_name']
        return self.img_root / path

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
        if self.img_names:
            return str(self._load_img_name(self.ids[index]))
        id = self.ids[index]
        img = self._load_img(id)
        mask = self._load_mask(id)
        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img, mask = transformed['image'].float(), transformed['mask'].long()
        return img, mask

    def __len__(self) -> int:
        return len(self.ids)



class DentalRestorations(LightningDataModule):
    def __init__(self, img_root, ann_path, batch_size, train_transforms=None, val_transforms=None, transforms=True,
                 seed: int = 42, num_workers: int = 8,
                 data_split: Tuple[int, int, int] = [0.8, 0.2, 0.0], train_shuffle=True, img_size=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = kwargs

    def setup(self, stage: Optional[str] = None):
        if stage != 'predict':
            dataset = COCOSegmentation(self.hparams.img_root, self.hparams.ann_path, self.hparams.train_transforms,
                                       apply_transforms=self.hparams.transforms, **self.kwargs)
            dataset_val_test = COCOSegmentation(self.hparams.img_root, self.hparams.ann_path, self.hparams.val_transforms,
                                       apply_transforms=self.hparams.transforms, **self.kwargs)
        else:
            dataset = COCOSegmentationPrediction(self.hparams.img_root, self.hparams.ann_path, self.hparams.img_size)
            dataset_val_test = COCOSegmentationPrediction(self.hparams.img_root, self.hparams.ann_path, self.hparams.img_size)
        tf, vf, tstf = self.hparams.data_split
        tn, vn, tstn = int(tf * len(dataset)), int(vf * len(dataset)), int(tstf * len(dataset))
        tn -= ((tn + vn + tstn) - len(dataset))
        split = [tn, vn, tstn]
        self.train_ds, _, _ = random_split(dataset, split, generator=torch.Generator().manual_seed(self.hparams.seed))
        _, self.val_ds, self.test_ds = random_split(dataset_val_test, split,
                                                    generator=torch.Generator().manual_seed(self.hparams.seed))

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.hparams.batch_size, shuffle=self.hparams.train_shuffle, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def predict_dataloader(self, stage="test"):
        if stage == "test":
            ds = self.test_ds
        elif stage == "val":
            ds = self.val_ds
        else:
            ds = self.train_ds
        return DataLoader(ds, batch_size=1, num_workers=self.hparams.num_workers, shuffle=False)
