import cv2
import numpy as np
import pytest

from src.core import BaseRecord, BBoxesRecordComponent, SizeRecordComponent, BBox, InstancesLabelsRecordComponent, \
    ClassMap, ImageRecordComponent
from src.transforms import Adapter
import albumentations as A
from PIL import Image

from src.transforms.albumentations_utils import resize_and_pad
from src.utils.bbox_inverse_transform import inverse_transform_bbox, inverse_transform_record


@pytest.fixture()
def sample_record(samples_source):
    record = BaseRecord(
        (
            BBoxesRecordComponent(),
            SizeRecordComponent(),
            InstancesLabelsRecordComponent(),
            ImageRecordComponent(),
    )
    )
    record.record_id = 0

    record.detection.set_class_map(ClassMap(['decay']))
    bboxes = [BBox.from_xyxy(100, 200, 300, 400)]
    record.detection.set_bboxes(bboxes)
    record.detection.set_img_size((1068, 847), original=True)
    record.detection.set_labels_by_id([1])
    record.set_img(Image.open(samples_source/'dataset/images/1.png'))
    return record


def test_inverse_transform_bbox(sample_record):
    orig_bbox = sample_record.detection.bboxes[0]

    transforms = [A.LongestMaxSize(1024), A.PadIfNeeded(896,1024, border_mode=cv2.BORDER_CONSTANT,value=[0,0,0])]
    transforms_adapter = Adapter(transforms)
    # record_t = transforms(sample_record)
    record_t = transforms_adapter.apply(sample_record)
    assert record_t.img_size == (1024, 896)
    record_inv_t = inverse_transform_bbox(record_t.detection.bboxes[0], transforms, record_t.original_img_size,
                                          record_t.img_size)
    assert record_inv_t.approx_eq(orig_bbox)


def test_inverse_transform_bbox(sample_record):
    orig_bbox = sample_record.detection.bboxes[0]

    transforms = [A.Resize(896, 1024)]
    transforms_adapter = Adapter(transforms)
    # record_t = transforms(sample_record)
    record_t = transforms_adapter.apply(sample_record)
    assert record_t.img_size == (1024, 896)
    record_inv_t = inverse_transform_bbox(record_t.detection.bboxes[0], transforms, record_t.original_img_size,
                                          record_t.img_size)
    assert record_inv_t.approx_eq(orig_bbox)

def test_inverse_record(sample_record):
    t = inverse_transform_record(sample_record)