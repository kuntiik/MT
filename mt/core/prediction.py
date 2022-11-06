from __future__ import annotations

__all__ = ["Prediction"]

from typing import Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from mt.core import BaseRecord


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

    def prediction_as_dict(self, device='cpu'):
        predictions = dict(
            boxes=torch.tensor([[*bbox.xyxy] for bbox in self.pred.detection.bboxes], device=device),
            scores=torch.tensor(self.pred.detection.scores, device=device),
            labels=torch.tensor(self.pred.detection.label_ids, device=device, dtype=torch.int)
        )
        ground_truths = dict(
            boxes=torch.tensor([[*bbox.xyxy] for bbox in self.ground_truth.detection.bboxes], device=device),
            labels=torch.tensor(self.ground_truth.detection.label_ids, device=device, dtype=torch.int)
        )
        return predictions, ground_truths
