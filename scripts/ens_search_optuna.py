
import sys
sys.path.append('..')
import optuna
from pathlib import Path
import json

from src.evaluation.prediction_evaluation import PredictionEval
from src.utils.conver_to_coco import to_coco

from src.evaluation.ensembling import BoxesEnsemble

class Objective:
    def __init__(self, boxes_ensemble, ensemble_method='wbf', weight_area=False):
        self.boxes_ensemble = boxes_ensemble
        self.ensemble_method = ensemble_method
        self.weight_area = weight_area
        self.area_dict = ["small", "medium", "large"]

    def __call__(self, trial: optuna.trial.Trial, stage='val'):
        weights = []
        if self.weight_area:
            for i in range(self.boxes_ensemble.num_models):
                sub_w = []
                for j in range(3):
                    sub_w.append(trial.suggest_float(f"model_weight{i}_area_{self.area_dict[j]}", 0, 1))
                weights.append(sub_w)
        else:
            for i in range(self.boxes_ensemble.num_models):
                weights.append(trial.suggest_float(f"model_weight{i}", 0, 1))

        threshold = trial.suggest_float(f'threshold', 0.05, 1)
        return boxesensemble.evaluate_ensemble(weights, threshold, ensemble_method=self.ensemble_method, stage=stage)


def normalize_confidences(preds, ann_path):
    with open(ann_path, 'r') as f:
        ann_file = json.load(f)

    confidences = []
    eval = PredictionEval()
    for pred in preds:
        preds_coco, names = to_coco(pred, ann_file)
        eval.load_data_coco_files(ann_path, preds_coco, names)
        # _, _, _, confidence = eval.precision_by_iou(stage='valid')
        confidence = eval.get_conf()
        confidences.append(confidence)

    preds_normalized = []
    max_conf = max(confidences)

    for pred, confidence in zip(preds, confidences):
        pred_n = {}
        for img_name, values in pred.items():
            pred_n[img_name] = values
            for c in pred_n[img_name]['scores']:
                c *= (max_conf/confidence)
        preds_normalized.append((pred_n))
        # pred_n = {}
        # pred_n['categories'] = pred['categories']
        # pred_n['images'] = pred['images']
        # pred_n['annotations'] = []
        # for ann in pred['annotations']:
        #     ann_new = ann
        #     ann_new['confidence'] *= (max_conf / confidence)
        #     pred_n['annotations'].append(ann_new)
    return preds_normalized, confidences


if __name__ == '__main__':

    ############################################
    # TODO change this to be modifiable from the command line
    pred_path = Path('../data/predictions/ens_search_all6')
    # pred_path = Path('../ens_test')
    ann_path = Path('/datagrid/personal/kuntluka/dental_rtg/caries6.json')

    preds = list(pred_path.iterdir())
    preds = [pred.name for pred in preds]

    # with open('confidence_thresholds.json', 'r') as f:
    #     confidence_thr = json.load(f)
    # confidences = [confidence_thr[p] for p in preds]

    preds_dicts = []
    for pred_name in preds:
        with open(pred_path / pred_name, 'r') as f:
            preds_dicts.append(json.load(f))

    # preds_dicts_normalized, confidences = normalize_confidences(preds_dicts, ann_path)

    with open('../data/metadata/metadata4000.json', 'r') as f:
        meta_data = json.load(f)

    preds_coco, names = to_coco(preds_dicts[0], ann_path)
    boxesensemble = BoxesEnsemble(meta_data, preds_dicts, ann_path, None)
    boxesensemble.load_pred_eval(ann_path, names)
    ###########################################
    study = optuna.load_study(
        # direction='maximize',
        storage="sqlite:///ens_search_caries_61.db",
        study_name='ens_search_6'
    )
    study.optimize(Objective(boxesensemble, weight_area=False, ensemble_method='wbf'), n_trials=5000)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial: ")
    trial = study.best_trial

    print(" Value: {}".format(trial.value))
    print(" Params: ")
    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))

