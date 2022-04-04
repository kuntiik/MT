import albumentations as A
import hydra

from src.transforms.albumentations_utils import resize_and_pad
from albumentations_adapter import Adapter


class TransformsComposer:
    def __init__(self, cfg):
        transforms = []
        for _, tfms_conf in cfg.items():
            transforms.append(hydra.utils.instantiate(tfms_conf))
        self.train_transforms = Adapter([*self._default_transforms(cfg.image_size), *transforms])
        self.val_transforms = Adapter([*self._default_transforms(cfg.image_size)])

    def train_val_transforms(self):
        return self.train_transforms, self.val_transforms

    @property
    def dental_caries_statistics(self):
        return dict(mean=0.3669, std=0.2768)

    def _default_transforms(self, image_size):
        normalize_stat = self.dental_caries_statistics
        return [
            resize_and_pad(image_size),
            A.Normalize(mean=self.dental_caries_statistics["mean"], std=self.dental_caries_statistics["std"]),
        ]

