import numpy as np
from scipy.stats import wilcoxon

from mt.evaluation.comparison.core import Comparison


def generate_gold_standard_data(annotations: str, model_preds, ids, model_id=0, annotator_id=1, iou_threshold=0.0, per_img=False):
    comparison = Comparison()
    comparison.parse_coco_data(annotations)
    errors, ious, names = [], [], []
    for i in ids:
        if i == annotator_id:
            continue
        if i == model_id:
            comparison.load_json_data(model_preds, False)
        else:
            comparison.load_coco_data(i, False)
        comparison.load_coco_data(annotator_id, True)
        if per_img:
            iou, e = comparison.pairwise_evaluate_per_img(iou_threshold=iou_threshold)
        else:
            iou, e = comparison.pairwise_evaluate(iou_threshold=iou_threshold)
        errors.append(e)
        ious.append(iou)
        names.append((annotator_id, i))
    return names, np.asarray(ious), np.asarray(errors)

def generate_gold_wilcoxon_signed_pvals(ious, errors):
    ids = [0,2,3,4,5,6,7,8]
    signs_ious = np.zeros((8,8))
    signs_errors = np.zeros((8,8))
    ious_mean = ious.mean(axis=1)
    errors_mean = errors.mean(axis=1)
    for i, id1 in enumerate(ids):
        for j, id2 in enumerate(ids):
            if i == j:
                continue
            signs_ious[i,j] = 1 if ious_mean[i] > ious_mean[j] else -1
            signs_errors[i,j] = -1 if errors_mean[i] > errors_mean[j] else 1

    errors_p = np.zeros((8, 8))
    ious_p = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i == j:
                errors_p[i, j] = 0
                ious_p[i, j] = 0
            else:
                errors_p[i, j] = wilcoxon(errors[i] - errors[j]).pvalue
                ious_p[i, j] = wilcoxon(ious[i] - ious[j]).pvalue
    ious_signed = np.multiply(1 - ious_p, signs_ious)
    errors_signed = np.multiply(1 - errors_p, signs_errors)
    return ious_signed, errors_signed