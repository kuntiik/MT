from pathlib import Path

import numpy as np
import torch

from src.core import BaseRecord, ScoresRecordComponent, ImageRecordComponent, InstancesLabelsRecordComponent, \
    BBoxesRecordComponent, ClassMap, BBox, Prediction
from src.data.random_splitter import SingleSplitSplitter
from src.datamodules.dental_caries.dental_caries import DentalCariesParser


def test_prediction(samples_source):
    data_root = samples_source / 'dataset'
    ann_file = "annotations.json"
    parser = DentalCariesParser(data_root, ann_file)
    record = parser.parse(SingleSplitSplitter())[0][0]

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

    final_pred = Prediction(pred=pred, ground_truth=record)
    pred_dict, gt_dict = final_pred.prediction_as_dict()
    # self.assertEqual(pred_dict['boxes'], torch.from_numpy(self.det[:, :4]))
    assert torch.all(torch.eq(pred_dict['boxes'], torch.from_numpy(det[:, :4])))
    assert torch.all(torch.eq(pred_dict['scores'], torch.from_numpy(det[:, 4]).type(torch.float32)))
