import albumentations as A
import hydra
from albumentations.pytorch import ToTensorV2

from mt.transforms.albumentations_utils import resize_and_pad, get_transform
from mt.transforms.albumentations_adapter import Adapter


class TransformsComposer:
    def __init__(self, cfg, tfms_mode='ice'):
        transforms = []
        for tfms_key, tfms_conf in cfg.items():
            if tfms_key == 'image_size':
                image_size = tfms_conf
                continue
            if tfms_conf.apply:
                transforms.append(hydra.utils.instantiate(tfms_conf.transform))
        if tfms_mode == 'ice':
            self.train_transforms = Adapter([*transforms, *self._default_transforms(image_size)])
            self.val_transforms = Adapter([*self._default_transforms(image_size)])
        else:
            self.train_transforms = A.Compose([*transforms, *self._default_transforms(image_size), ToTensorV2()])
            self.val_transforms = A.Compose([*self._default_transforms(image_size), ToTensorV2()])

    def train_val_transforms(self):
        return self.train_transforms, self.val_transforms

    @property
    def dental_caries_statistics(self):
        return dict(mean=0.3669, std=0.2768)

    def _default_transforms(self, image_size):
        normalize_stat = self.dental_caries_statistics
        return [
            *resize_and_pad(image_size),
            A.Normalize(mean=self.dental_caries_statistics["mean"], std=self.dental_caries_statistics["std"]),
        ]

    def get_val_transform(self, transform):
        return get_transform(self.val_transforms.tfms_list, transform)

    def get_train_transform(self, transform):
        return get_transform(self.train_transforms.tfms_list, transform)

