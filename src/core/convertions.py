from typing import List

from src.core import Prediction


def preds2dicts(preds: List[Prediction], device='cpu'):
    pred_out, gt_out = [], []
    for pred in preds:
        pred_dict, gt_dict = pred.prediction_as_dict(device)
        pred_out.append(pred_dict)
        gt_out.append(gt_dict)
    return pred_out, gt_out
