import json
from pathlib import Path
from typing import List, Dict

import numpy as np

from src.evaluation.boxes_fusion.ensemble_boxes_nms import nms, soft_nms
from src.evaluation.boxes_fusion.ensemble_boxes_nmw import non_maximum_weighted
from src.evaluation.boxes_fusion.ensemble_boxes_wbf import weighted_boxes_fusion
from src.evaluation.prediction_evaluation import PredictionEval
from src.utils.conver_to_coco import to_coco


class BoxesEnsemble:
    def __init__(self, metadata: Dict, prediction_dictionaries: List, annotations_path=Path,
                 confidence_thresholds: List = None):
        self.metadata = metadata
        self.prediction_dictionaries = prediction_dictionaries
        self.confidence_thresholds = np.asarray(confidence_thresholds)
        if confidence_thresholds is not None:
            self.normalize_confidence()
        self.pred_eval = None
        self.annotations_path = annotations_path
        self.num_models = len(prediction_dictionaries)
        # if confidence_thresholds is None:
        #     self.confidence_thresholds = [1 ]

    @property
    def ensemble_method_dict(self):
        """Dict to represent function names by their shortcuts"""
        return {'wbf': weighted_boxes_fusion, 'nms': nms, 'nmw': non_maximum_weighted, 'snms': soft_nms}

    def load_pred_eval(self, annotations_path, train_val_names):
        self.pred_eval = PredictionEval()
        self.pred_eval.load_ground_truth(annotations_path, train_val_names)

    def normalize_confidence(self):
        max_conf = np.max(self.confidence_thresholds)
        self.conf_norm_coeffs = max_conf / self.confidence_thresholds
        for index, coef in enumerate(self.conf_norm_coeffs):
            for i in self.prediction_dictionaries[index].keys():
                self.prediction_dictionaries[index][i]['scores'] = [s * coef for s in
                                                                    self.prediction_dictionaries[index][i]['scores']]
            # self.prediction_dictionaries[index]['conf_norm'] = coef

    def prepare_img(self, img_name):
        """Changes the boxes coordinates to relative and outputs boxes, scores,
        labels in the format required for the ensemble"""
        boxes_all, scores_all, labels_all = [], [], []
        for pred_dict in self.prediction_dictionaries:
            boxes = []
            for bbox in pred_dict[img_name]['bboxes']:
                x1, y1, x2, y2 = bbox
                boxes.append([x1 / 1068, y1 / 847, x2 / 1068, y2 / 847])
            if len(boxes) == 0:
                boxes = np.array([]).reshape(0, 4)
            boxes_all.append(boxes)
            scores_all.append(pred_dict[img_name]['scores'])
            labels_all.append(pred_dict[img_name]['labels'])
        return boxes_all, scores_all, labels_all

    def ensemble(self, weights, iou_thr, skip_box_thr=0.001, stage='val', ensemble_method='wbf', sigma=0.1):
        """Ensembles boxes by specified ensemble_method, returns dictionary of ensembled predictions"""
        assert stage in ['test', 'val', 'train']
        files = stage + '_files'
        dict_out = {}
        for img_name in self.metadata[files]:
            boxes, scores, labels = self.prepare_img(img_name)
            boxes_out, scores_out, labels_out = self.ensemble_method_dict[ensemble_method](boxes, scores, labels,
                                                                                           weights=weights,
                                                                                           iou_thr=iou_thr,
                                                                                           skip_box_thr=skip_box_thr,
                                                                                           sigma=sigma)
            boxes_rescaled = []
            for bbox in boxes_out:
                x1, y1, x2, y2 = bbox
                boxes_rescaled.append([x1 * 1068, y1 * 847, x2 * 1068, y2 * 847])
            dict_out[img_name] = {'bboxes': boxes_rescaled, 'scores': scores_out, 'labels': labels_out, 'stage': stage}
        return dict_out

    def enseble_and_save(self, name: str, weights: List[float], iou_thr: float, ensemble_method: str = 'wbf',
                         skip_box_thr: float = 0.01, sigma: float = 0.7):
        """Does the ensemble and saves it as the name.json file."""
        train_data = self.ensemble(weights, iou_thr, skip_box_thr, stage='train', ensemble_method=ensemble_method)
        val_data = self.ensemble(weights, iou_thr, skip_box_thr, stage='val', ensemble_method=ensemble_method)
        test_data = self.ensemble(weights, iou_thr, skip_box_thr, stage='test', ensemble_method=ensemble_method)
        data = {**test_data, **val_data, **train_data}
        for keys, values in data.items():
            data[keys]['labels'] = data[keys]['labels'].astype(int).tolist()
            data[keys]['scores'] = data[keys]['scores'].tolist()
        with open(name, 'w') as f:
            json.dump(data, f)
        return data

    def evaluate_ensemble(self, weights, iou_thr, stage='val', ensemble_method='wbf', sigma=0.5, skip_box_thr=0.001):
        """Does the ensemble and returns AP@.iou_thr on the given stage. """
        assert stage in ['test', 'val', 'train']
        files = stage + '_files'
        ensembled_boxes = self.ensemble(weights, iou_thr,skip_box_thr=skip_box_thr, stage=stage, ensemble_method=ensemble_method,
                                        sigma=sigma)
        preds_coco, _ = to_coco(ensembled_boxes, str(self.annotations_path))
        self.pred_eval.load_predictions(preds_coco)

        # this is hot-patch for cross-validation return to this and rewrite it
        if stage == 'val':
            return self.pred_eval.map_query(stage=stage)
        else:
            return self.pred_eval.get_latex_table('ensembling_' + ensemble_method)
