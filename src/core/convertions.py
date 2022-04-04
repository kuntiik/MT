from typing import List

from src.core import Prediction


def preds2dicts(preds: List[Prediction]):
    pred_out, gt_out = [], []
    for pred in preds:
        pred_dict, gt_dict = pred.prediction_as_dict()
        pred_out.append(pred_dict)
        gt_out.append(gt_out)
    return pred_out, gt_out
