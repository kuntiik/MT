import hydra
import numpy as np
from albumentations import PadIfNeeded, Normalize, LongestMaxSize, HorizontalFlip

from src.core import ClassMap, BaseRecord, ImageRecordComponent
from src.transforms import TransformsComposer


def test_transforms_instantiation(cfg):
    composer = TransformsComposer(cfg.transforms.cfg)
    # composer: TransformsComposer = hydra.utils.instantiate(cfg.transforms)
    train, val = composer.train_val_transforms()
    assert isinstance(composer.get_train_transform('Normalize'), Normalize)
    assert isinstance(composer.get_train_transform('PadIfNeeded'), PadIfNeeded)
    assert isinstance(composer.get_train_transform('LongestMaxSize'), LongestMaxSize)
    assert isinstance(composer.get_train_transform('HorizontalFlip'), HorizontalFlip)

    assert isinstance(composer.get_val_transform('Normalize'), Normalize)
    assert isinstance(composer.get_val_transform('PadIfNeeded'), PadIfNeeded)
    assert isinstance(composer.get_val_transform('LongestMaxSize'), LongestMaxSize)


def test_resize():
    target_img_size = (100, 80)
    original_img_size = (50, 50)
    cfg = dict(image_size=target_img_size)

    img = np.ones((*original_img_size, 3), dtype=np.uint8)
    record = BaseRecord(
        (
            ImageRecordComponent(),
        )
    )
    record.set_record_id(1)
    record.set_img(img, original_img_size=True)

    composer = TransformsComposer(cfg)
    train_tfms, val_tfms = composer.train_val_transforms()
    out_img = train_tfms(record)
    assert out_img.img_size == target_img_size
    assert out_img.original_img_size == original_img_size


def test_instantiation(cfg):
    hydra.utils.instantiate(cfg.transforms, _recursive_=False)
