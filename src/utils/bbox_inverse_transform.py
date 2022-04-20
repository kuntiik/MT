import copy
from typing import List, Any, Callable, Tuple
import numpy as np

from src.core import BBox, record, BaseRecord, FilepathRecordComponent, BBoxesRecordComponent, ScoresRecordComponent, \
    InstancesLabelsRecordComponent, Prediction
from src.transforms.albumentations_utils import resize_and_pad


def get_transform(tfms_list: List[Any], t: str) -> Any:
    for el in tfms_list:
        if t in str(type(el)):
            return el
    return None


def func_max_size(
        height: int, width: int, max_size: int, func: Callable[[int, int], int]
) -> Tuple[int, int]:
    scale = max_size / float(func(width, height))
    if scale != 1.0:
        height, width = tuple(dim * scale for dim in (height, width))
    return height, width


def get_size_without_padding(
        tfms: List[Any], before_height, before_width, after_height, after_width
):
    if get_transform(tfms, "Pad") is not None:
        t = get_transform(tfms, "SmallestMaxSize")
        if t is not None:
            presize = t.max_size
            after_height, after_width = func_max_size(before_height, before_width, presize, min)
        t = get_transform(tfms, "LongestMaxSize")
        if t is not None:
            size = t.max_size
            after_height, after_width = func_max_size(before_height, before_width, size, max)
    return after_height, after_width


#TODO this function is not ready for adding padding to change width
def inverse_transform_bbox(
        bbox: BBox, tfms: List[Any], original_size, size
):
    before_width, before_height = original_size
    after_width, after_height = size

    no_pad_height, no_pad_width = get_size_without_padding(
        tfms, before_height, before_width, after_height, after_width
    )
    bbox = copy.deepcopy(bbox)
    pad = np.abs(after_height - no_pad_height) / 2
    h_scale, w_scale = no_pad_height / before_height, no_pad_width / before_width
    # if after_height < after_width:
    x1, x2, y1, y2 = bbox.xmin, bbox.xmax, bbox.ymin - pad, bbox.ymax - pad
    # else:
        # x1, x2, y1, y2 = bbox.xmin - pad, bbox.xmax - pad, bbox.ymin, bbox.ymax

    x1, x2, y1, y2 = (max(x1, 0), min(x2, after_width), max(y1, 0), min(y2, after_height))
    x1, x2, y1, y2 = (x1 / w_scale, x2 / w_scale, y1 / h_scale, y2 / h_scale)
    return BBox.from_xyxy(x1, y1, x2, y2)


def inverse_transform_record(
        record: record, tfms=None
):
    prediction = BaseRecord(
        (
            ScoresRecordComponent(),
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )

    orig_size = record.common.original_img_size
    size = record.common.img_size

    if tfms is None:
        tfms = resize_and_pad(size)

    prediction.detection.set_class_map(record.detection.class_map)

    prediction.record_id = record.record_id
    pred = record.pred.detection
    inverted_bboxes = [inverse_transform_bbox(bbox, tfms, orig_size, size) for bbox in record.pred.detection.bboxes]
    prediction.detection.set_bboxes(inverted_bboxes)
    prediction.detection.set_labels(record.pred.detection.labels)
    prediction.detection.set_scores(record.pred.detection.scores)

    ground_truth = None
    # if record.ground_truth is not None:
    #     ground_truth = BaseRecord(
    #         (
    #             FilepathRecordComponent(),
    #             InstancesLabelsRecordComponent(),
    #             BBoxesRecordComponent(),
    #         )
    #     )
    #     inverted_bboxes = [inverse_transform_bbox(bbox) for bbox in record.ground_truth.detection.bboxes]
    #     ground_truth.detection.set_bboxes(inverted_bboxes)
    #     ground_truth.detection.set_labels(record.ground_truth.detection.labels)

    return Prediction(pred=prediction, ground_truth=ground_truth)

def predictions_to_fiftyone(predictions, stage=None):
    data = {}
    for pred in predictions:
        if type(pred) == list and len(pred) == 1:
            pred = pred[0]
        img_id = pred.record_id
        bboxes_list, scores, labels = [], [], []
        record_inv = inverse_transform_record(pred)
        for bbox, score, label in zip(record_inv.pred.detection.bboxes, record_inv.pred.detection.scores, record_inv.pred.detection.labels):
            bboxes_list.append(np.array([*bbox.xyxy]).astype(np.double).tolist())
            scores.append(np.double(score))
            labels.append(1)
        data[img_id] = {
            "bboxes" : bboxes_list,
            "labels" : labels,
            "scores" : scores,
            "stage" : stage
        }
    return data



