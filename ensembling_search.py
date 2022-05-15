from typing import List, Dict
import numpy as np
from src.evaluation.boxes_fusion.ensemble_boxes_wbf import weighted_boxes_fusion
from src.evaluation.boxes_fusion.ensemble_boxes_nms import nms, soft_nms
from src.evaluation.boxes_fusion.ensemble_boxes_nmw import non_maximum_weighted
from src.utils.conver_to_coco import to_coco
from src.evaluation.prediction_evaluation import PredictionEval
from pathlib import Path


class BoxesEnsemble:
    def __init__(self, metadata: Dict, prediction_dictionaries: List, annotations_path=Path,
                 confidence_thresholds: List = None):
        self.metadata = metadata
        self.prediction_dictionaries = prediction_dictionaries
        self.confidence_thresholds = np.asarray(confidence_thresholds)
        self.normalize_confidence()
        self.pred_eval = None
        self.annotations_path = annotations_path
        # if confidence_thresholds is None:
        #     self.confidence_thresholds = [1 ]

    def load_pred_eval(self, annotations_path, train_val_names):
        self.pred_eval = PredictionEval()
        self.pred_eval.load_ground_truth(annotations_path, train_val_names)

    def normalize_confidence(self):
        max_conf = np.max(self.confidence_thresholds)
        self.conf_norm_coeffs = max_conf / self.confidence_thresholds
        for index, coef in enumerate(self.conf_norm_coeffs):
            for i in self.prediction_dictionaries[index].keys():
                self.prediction_dictionaries[index][i]['scores'] = [s * coef for s in self.prediction_dictionaries[index][i]['scores']]
            # self.prediction_dictionaries[index]['conf_norm'] = coef

    def prepare_img(self, img_name):
        boxes_all = []
        scores_all = []
        labels_all = []
        for pred_dict in self.prediction_dictionaries:
            boxes = []
            for bbox in pred_dict[img_name]['bboxes']:
                x1, y1, x2, y2 = bbox
                boxes.append([x1 / 1068, y1 / 847, x2 / 1068, y2 / 847])
            if len(boxes) == 0:
                boxes = np.array([]).reshape(0,4)
            boxes_all.append(boxes)
            scores_all.append(pred_dict[img_name]['scores'])
            labels_all.append(pred_dict[img_name]['labels'])
        return boxes_all, scores_all, labels_all

    def ensemble(self, weights, iou_thr, files='val_files', skip_box_thr=0.01, stage='val', ensemble_method='wbf', sigma=0.1):
        dict_out = {}
        for img_name in self.metadata[files]:
            boxes, scores, labels = self.prepare_img(img_name)
            if ensemble_method == 'wbf':
                boxes_out, scores_out, labels_out = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr,
                                                                      skip_box_thr=skip_box_thr)
            elif ensemble_method == 'nwm':
                boxes_out, scores_out, labels_out = non_maximum_weighted(boxes, scores, labels, weights=weights, iou_thr=iou_thr,
                                                                          skip_box_thr=skip_box_thr)
            elif ensemble_method == 'snms':
                boxes_out, scores_out, labels_out = soft_nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr, sigma=sigma,
                                                                         thresh=skip_box_thr)
            elif ensemble_method == 'nms':
                boxes_out, scores_out, labels_out = nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr)

            boxes_rescaled = []
            for bbox in boxes_out:
                x1, y1, x2, y2 = bbox
                boxes_rescaled.append([x1 * 1068, y1 * 847, x2 * 1068, y2 * 847])
            dict_out[img_name] = {'bboxes': boxes_rescaled, 'scores': scores_out, 'labels': labels_out, 'stage': stage}
        return dict_out

    def evaluate_ensemble(self, weights, iou_thr, stage='val', ensemble_method='wbf', sigma=0.5):
        if stage == 'val':
            files = 'val_files'
        else:
            files = 'test_files'
        ensembled_boxes = self.ensemble(weights, iou_thr, files, stage=stage, ensemble_method=ensemble_method, sigma=sigma)
        preds_coco, _ = to_coco(ensembled_boxes, str(self.annotations_path))
        self.pred_eval.load_predictions(preds_coco)

        return self.pred_eval.map_query(stage=stage)
