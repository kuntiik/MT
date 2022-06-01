import optuna
from pathlib import Path
import json
import ensembling_search


class Objective:
    def __init__(self, boxes_ensemble, ensemble_method='wbf', weight_area=False):
        self.boxes_ensemble = boxes_ensemble
        self.ensemble_method = ensemble_method
        self.weight_area = weight_area
        self.area_dict = ["small", "medium", "large"]

    def __call__(self, trial: optuna.trial.Trial):
        weights = []
        if self.weight_area:
            for i in range(self.boxes_ensemble.num_models):
                sub_w = []
                for j in range(3):
                    sub_w.append(trial.suggest_float(f"model_weight{i}_area_{self.area_dict[j]}", 0.01, 1))
                weights.append(sub_w)
        else:
            for i in range(self.boxes_ensemble.num_models):
                weights.append(trial.suggest_float(f"model_weight{i}", 0.01, 1))

        threshold = trial.suggest_float(f'threshold', 0.05, 1)
        return boxesensemble.evaluate_ensemble(weights, threshold, ensemble_method=self.ensemble_method)



if __name__ == '__main__':

    ############################################
    # TODO change this to be modifiable from the command line
    pred_path = Path('predictions_non_nms/yolov5/medium')
    ann_path = Path('/datagrid/personal/kuntluka/dental_rtg3/annotations.json')

    preds = list(pred_path.iterdir())
    preds = [pred.name for pred in preds]

    # with open('confidence_thresholds.json', 'r') as f:
    #     confidence_thr = json.load(f)
    # confidences = [confidence_thr[p] for p in preds]

    preds_dicts = []
    for pred_name in preds:
        with open(pred_path / pred_name, 'r') as f:
            preds_dicts.append(json.load(f))

    with open('metadata4000.json', 'r') as f:
        meta_data = json.load(f)

    boxesensemble = ensembling_search.BoxesEnsemble(meta_data, preds_dicts, ann_path, None)
    boxesensemble.load_pred_eval(ann_path, meta_data)
    ###########################################
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///ens_search_yolo_m.db",
        study_name='ens_search_wbf_yolo_non_nms'
    )
    study.optimize(Objective(boxesensemble, weight_area=False, ensemble_method='wbf'), n_trials=3000)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial: ")
    trial = study.best_trial

    print(" Value: {}".format(trial.value))
    print(" Params: ")
    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))

