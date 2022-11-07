from typing import Tuple

import pytest
import hydra
from pathlib import Path
import albumentations as A

import mt.transforms.albumentations_adapter
from mt.core import ClassMap
from mt.data.dataset import Dataset
from mt.data.parsers.pasacal_voc import VOCBBoxParser
from mt.data.random_splitter import RandomSplitter, SingleSplitSplitter
from mt.datamodules.dental import DentalCaries
from mt.core.record_components import *
from mt.datamodules.dental.dental_caries import DentalCariesParser


@pytest.fixture()
def cfg():
    with hydra.initialize(config_path='../mt/configs'):
        cfg = hydra.compose(config_name='train.yaml', overrides=['datamodule.num_workers=0'])
    return cfg


@pytest.fixture()
def cfg_rectangle():
    with hydra.initialize(config_path='../mt/configs'):
        cfg = hydra.compose(config_name='train.yaml',
                            overrides=['datamodule.num_workers=0', 'module.img_size=[512, 256]'])
    return cfg


@pytest.fixture()
def dataset():
    data_root = "samples/dataset"
    ann_file = "annotations.json"
    dm = DentalCaries(data_root, 'efficientdet', ann_file, 1, 0, train_val_test_split=[1 / 3, 1 / 3, 1 / 3])
    dm.setup()
    yield dm
    # train_loader = dm.train_dataloader()
    # val_loader = dm.val_dataloader()


@pytest.fixture()
def mini_caries_parser(samples_source):
    ann_file = "annotations.json"
    data_root = samples_source / 'dataset'
    yield DentalCariesParser(data_root, ann_file)


@pytest.fixture()
def mini_caries_records(mini_caries_parser):
    records = mini_caries_parser.parse(SingleSplitSplitter())[0]
    yield records


@pytest.fixture(scope="session")
def samples_source():
    return Path(__file__).absolute().parent.parent / "samples"


@pytest.fixture(scope="session")
def config_source():
    return Path(__file__).absolute().parent.parent / "configs"


@pytest.fixture(scope="session")
def fridge_class_map():
    classes = sorted({"milk_bottle", "carton", "can", "water_bottle"})
    return ClassMap(classes)


@pytest.fixture(scope="module")
def fridge_ds(samples_source, fridge_class_map) -> Tuple[Dataset, Dataset]:
    IMG_SIZE = 384
    from albumentations.pytorch.transforms import ToTensorV2

    parser = VOCBBoxParser(
        annotations_dir=samples_source / "fridge/annotations",
        images_dir=samples_source / "fridge/images",
        class_map=fridge_class_map,
    )

    data_splitter = RandomSplitter([0.5, 0.5], seed=42)
    train_records, valid_records = parser.parse(data_splitter)

    tfms_ = mt.transforms.albumentations_adapter.Adapter([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize()])

    train_ds = Dataset(train_records, tfms_)
    valid_ds = Dataset(valid_records, tfms_)

    return train_ds, valid_ds


component_field = {
    RecordIDRecordComponent: "record_id",
    ClassMapRecordComponent: "class_map",
    FilepathRecordComponent: "img",
    ImageRecordComponent: "img",
    SizeRecordComponent: "img_size",
    InstancesLabelsRecordComponent: "labels",
    BBoxesRecordComponent: "bboxes",
    AreasRecordComponent: "areas",
}


@pytest.fixture
def check_attributes_on_component():
    def _inner(record):
        for component in record.components:
            name = component_field[component.__class__]
            task_subfield = getattr(record, component.task.name)
            assert getattr(task_subfield, name) is getattr(component, name)

    return _inner


@pytest.fixture(scope="session")
def voc_class_map():
    classes = sorted(
        {
            "person",
            "bird",
            "cat",
            "cow",
            "dog",
            "horse",
            "sheep",
            "aeroplane",
            "bicycle",
            "boat",
            "bus",
            "car",
            "motorbike",
            "train",
            "bottle",
            "chair",
            "diningtable",
            "pottedplant",
            "sofa",
            "tvmonitor",
        }
    )

    return ClassMap(classes=classes)
