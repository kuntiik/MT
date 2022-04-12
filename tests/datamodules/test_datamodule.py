import pytest
import torch

from src.data.random_splitter import SingleSplitSplitter, RandomSplitter
from src.data.record_collection import RecordCollection
from src.datamodules.dental_caries.dental_caries import DentalCariesParser, DentalCaries
from pathlib import Path
from hydra.utils import instantiate
import hydra

from src.transforms import TransformsComposer



def test_parsing(mini_caries_records):
    assert len(mini_caries_records) == 3


def test_splitted_parsing(mini_caries_parser):
    train, val, test = mini_caries_parser.parse(RandomSplitter([1 / 3, 1 / 3, 1 / 3], seed=42))
    train2, val2, test2 = mini_caries_parser.parse(RandomSplitter([1 / 3, 1 / 3, 1 / 3], seed=42))
    assert len(train) + len(val) + len(test) == 3
    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1
    assert train[0].record_id == train2[0].record_id
    assert val[0].record_id == val2[0].record_id
    assert test[0].record_id == test2[0].record_id


def test_type(mini_caries_records):
    assert type(mini_caries_records) == RecordCollection


def test_bboxes(mini_caries_records):
    assert len(mini_caries_records.get_by_record_id('1.png').detection.bboxes) == 4
    assert len(mini_caries_records.get_by_record_id('10.png').detection.bboxes) == 2
    assert len(mini_caries_records.get_by_record_id('100.png').detection.bboxes) == 0


def test_orig_img_size(mini_caries_records):
    record = mini_caries_records[0]
    assert record.img_size == record.original_img_size


def test_dental_caries(cfg):
    dm = instantiate(cfg.datamodule)
    dm.setup()
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()
    assert (len(train_dl) + len(val_dl) + len(test_dl)) == 2599


def test_dental_caries_sizes(cfg):
    target_size = cfg.module.img_size
    if type(target_size) == int:
        target_size = (target_size, target_size)
    target_shape = torch.Size([1, 3, *target_size])

    transforms_composer: TransformsComposer = instantiate(cfg.transforms, _recursive_=False)
    train_transforms, val_transforms = transforms_composer.train_val_transforms()
    dm: DentalCaries = instantiate(cfg.datamodule, train_transforms=train_transforms, val_transforms=val_transforms)
    dm.setup()
    train_dl = iter(dm.train_dataloader())
    val_dl = iter(dm.val_dataloader())
    test_dl = iter(dm.test_dataloader())

    for _ in range(20):
        (x_train, y_train), _ = next(train_dl)
        (x_val, y_val), _ = next(val_dl)
        (x_test, y_test), _ = next(test_dl)
        assert x_train.shape == target_shape
        assert x_val.shape == target_shape
        assert x_test.shape == target_shape


def test_dental_caries_sizes_rectangle(cfg_rectangle):
    target_size = cfg_rectangle.module.img_size
    if type(target_size) == int:
        target_size = (target_size, target_size)
    width, height = target_size
    target_shape = torch.Size([1, 3, height, width])

    transforms_composer: TransformsComposer = instantiate(cfg_rectangle.transforms, _recursive_=False)
    train_transforms, val_transforms = transforms_composer.train_val_transforms()
    dm: DentalCaries = instantiate(cfg_rectangle.datamodule, train_transforms=train_transforms, val_transforms=val_transforms)
    dm.setup()
    train_dl = iter(dm.train_dataloader())
    val_dl = iter(dm.val_dataloader())
    test_dl = iter(dm.test_dataloader())

    for _ in range(20):
        (x_train, y_train), _ = next(train_dl)
        (x_val, y_val), _ = next(val_dl)
        (x_test, y_test), _ = next(test_dl)
        assert x_train.shape == target_shape
        assert x_val.shape == target_shape
        assert x_test.shape == target_shape
