from pathlib import Path

import numpy as np
import pytest
import torch

from mt.core import BaseRecord, ScoresRecordComponent, ImageRecordComponent, InstancesLabelsRecordComponent, \
    BBoxesRecordComponent, ClassMap, BBox, Prediction
from mt.data.random_splitter import SingleSplitSplitter
from mt.datamodules.dental.dental_caries import DentalCariesParser

@pytest.fixture()
def pred():
    pred = BaseRecord(
        (
            ScoresRecordComponent(),
            ImageRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )
    det = np.array([[100, 100, 300, 300, 0.2, 1], [50, 50.5, 150, 400, 0.8, 1]])

    pred.detection.set_class_map(ClassMap(['decay']))
    pred.detection.set_labels_by_id(det[:, 5].astype(int))
    pred.detection.set_bboxes([BBox.from_xyxy(*xyxy) for xyxy in det[:, :4]])
    pred.detection.set_scores(det[:, 4])
    yield pred


def test_prediction(samples_source, pred):
    data_root = samples_source / 'dataset'
    ann_file = "annotations.json"
    parser = DentalCariesParser(data_root, ann_file)
    record = parser.parse(SingleSplitSplitter())[0][0]

    det = np.array([[100, 100, 300, 300, 0.2, 1], [50, 50.5, 150, 400, 0.8, 1]])

    final_pred = Prediction(pred=pred, ground_truth=record)
    pred_dict, gt_dict = final_pred.prediction_as_dict()
    # self.assertEqual(pred_dict['boxes'], torch.from_numpy(self.det[:, :4]))
    assert torch.all(torch.eq(pred_dict['boxes'], torch.from_numpy(det[:, :4])))
    assert torch.all(torch.eq(pred_dict['scores'], torch.from_numpy(det[:, 4]).type(torch.float32)))
    assert 'boxes' in gt_dict.keys()
    assert 'labels' in gt_dict.keys()


def test_empty_prediction(mini_caries_records, pred):
    record = mini_caries_records.get_by_record_id('100.png')
    assert len(record.detection.bboxes) == 0
    final_pred = Prediction(pred=pred, ground_truth=record)
    pred_dict, gt_dict = final_pred.prediction_as_dict()
    assert 'boxes' in pred_dict.keys()
    assert 'labels' in pred_dict.keys()
    assert 'scores' in pred_dict.keys()
    assert 'boxes' in gt_dict.keys()
    assert 'labels' in gt_dict.keys()
