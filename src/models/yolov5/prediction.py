import torch
from typing import Sequence, List
from yolov5.utils.general import non_max_suppression

from src.core import BaseRecord, Prediction, ScoresRecordComponent, ImageRecordComponent, \
    InstancesLabelsRecordComponent, BBoxesRecordComponent, BBox
from src.utils import tensor_to_image


def convert_raw_predictions(
        batch,
        raw_preds: torch.Tensor,
        records: Sequence[BaseRecord],
        detection_threshold: float,
        nms_iou_threshold: float,
        keep_images: bool = False,
) -> List[Prediction]:
    dets = non_max_suppression(
        raw_preds, conf_thres=detection_threshold, iou_thres=nms_iou_threshold
    )
    dets = [d.detach().cpu().numpy() for d in dets]
    preds = []
    for det, record, tensor_image in zip(dets, records, batch):

        pred = BaseRecord(
            (
                ScoresRecordComponent(),
                ImageRecordComponent(),
                InstancesLabelsRecordComponent(),
                BBoxesRecordComponent(),
            )
        )

        pred.detection.set_class_map(record.detection.class_map)
        pred.detection.set_labels_by_id(det[:, 5].astype(int) + 1)
        pred.detection.set_bboxes([BBox.from_xyxy(*xyxy) for xyxy in det[:, :4]])
        pred.detection.set_scores(det[:, 4])

        if keep_images:
            record.set_img(tensor_to_image(tensor_image))

        preds.append(Prediction(pred=pred, ground_truth=record))

    return preds
