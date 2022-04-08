import pytest

from src.data.random_splitter import SingleSplitSplitter, RandomSplitter
from src.data.record_collection import RecordCollection
from src.datamodules.dental_caries.dental_caries import DentalCariesParser, DentalCaries
from pathlib import Path
from hydra.utils import instantiate


@pytest.fixture()
def parser(samples_source):
    ann_file = "annotations.json"
    data_root = samples_source / 'dataset'
    yield DentalCariesParser(data_root, ann_file)


@pytest.fixture()
def records(parser):
    records = parser.parse(SingleSplitSplitter())[0]
    yield records


def test_parsing(records):
    assert len(records) == 3


def test_splitted_parsing(parser):
    train, val, test = parser.parse(RandomSplitter([1 / 3, 1 / 3, 1 / 3], seed=42))
    train2, val2, test2 = parser.parse(RandomSplitter([1 / 3, 1 / 3, 1 / 3], seed=42))
    assert len(train) + len(val) + len(test) == 3
    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1
    assert train[0].record_id == train2[0].record_id
    assert val[0].record_id == val2[0].record_id
    assert test[0].record_id == test2[0].record_id


def test_type(records):
    assert type(records) == RecordCollection


def test_bboxes(records):
    assert len(records.get_by_record_id('1.png').detection.bboxes) == 4
    assert len(records.get_by_record_id('10.png').detection.bboxes) == 2
    assert len(records.get_by_record_id('100.png').detection.bboxes) == 0


def test_orig_img_size(records):
    record = records[0]
    assert record.img_size == record.original_img_size


def test_dental_caries(cfg):
    dm = instantiate(cfg.datamodule)
    dm.setup()
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()
    assert (len(train_dl) + len(val_dl) + len(test_dl)) == 2599
