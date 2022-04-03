from typing import Sequence, List

import torch
import torchvision

from src.core import BaseRecord, Prediction, ScoresRecordComponent, ImageRecordComponent, \
    InstancesLabelsRecordComponent, BBoxesRecordComponent, BBox
from src.utils.torch_utils import tensor_to_image


def convert_raw_predictions(
        batch,
        raw_preds: torch.Tensor,
        records: Sequence[BaseRecord],
        detection_threshold: float,
        keep_images: bool = False,
) -> List[Prediction]:
    tensor_images, *_ = batch
    dets = raw_preds.detach().cpu().numpy()
    preds = []
    for det, record, tensor_image in zip(dets, records, tensor_images):
        if detection_threshold > 0:
            scores = det[:, 4]
            keep = scores > detection_threshold
            det = det[keep]

        pred = BaseRecord(
            (
                ScoresRecordComponent(),
                ImageRecordComponent(),
                InstancesLabelsRecordComponent(),
                BBoxesRecordComponent(),
            )
        )

        pred.detection.set_class_map(record.detection.class_map)
        pred.detection.set_labels_by_id(det[:, 5].astype(int))
        pred.detection.set_bboxes([BBox.from_xyxy(*xyxy) for xyxy in det[:, :4]])
        pred.detection.set_scores(det[:, 4])

        if keep_images:
            record.set_img(tensor_to_image(tensor_image))

        preds.append(Prediction(pred=pred, ground_truth=record))

    return preds
