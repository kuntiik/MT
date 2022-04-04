from __future__ import annotations

__all__ = ["Prediction"]

from typing import Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from src.core import BaseRecord


class Prediction:
    def __init__(self, pred: BaseRecord, ground_truth: Optional[BaseRecord] = None):
        self.ground_truth = ground_truth
        self.pred = pred

        # TODO: record_id, img_size and stuff has to be set even if ground_truth is not stored
        if ground_truth is not None:
            pred.set_record_id(ground_truth.record_id)
            pred.set_img_size(ground_truth.original_img_size, original=True)
            pred.set_img_size(ground_truth.img_size)
            # HACK
            if ground_truth.img is not None:
                pred.set_img(ground_truth.img)

    def __getattr__(self, name):
        if name == "pred":
            raise AttributeError

        return getattr(self.pred, name)

    def prediction_as_dict(self):
        predictions = dict(
            boxes=torch.Tensor([[*bbox.xyxy] for bbox in self.pred.detection.bboxes]),
            scores=torch.Tensor(self.pred.detection.scores),
            labels=torch.IntTensor(self.pred.detection.label_ids)
        )
        ground_truths = dict(
            boxes=torch.Tensor([[*bbox.xyxy] for bbox in self.ground_truth.detection.bboxes]),
            labels=torch.IntTensor(self.ground_truth.detection.label_ids)
        )
        return predictions, ground_truths
