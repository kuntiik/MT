import json
from pathlib import Path
import numpy as np

import numpy.testing
import pytest

from src.evaluation.ensembling import BoxesEnsemble
from src.evaluation.prediction_evaluation import PredictionEval
from src.utils.conver_to_coco import to_coco


@pytest.fixture
def ensemble_data():
    ann_path = Path('/datagrid/personal/kuntluka/dental_rtg/caries6.json')
    pred_path = Path('../data/predictions/ens_search_test/')
    preds = list(pred_path.iterdir())
    preds = [pred.name for pred in preds]

    preds_dicts = []
    for pred_name in preds:
        with open(pred_path / pred_name, 'r') as f:
            preds_dicts.append(json.load(f))
    with open('../data/metadata/metadata4000.json', 'r') as f:
        meta_data = json.load(f)
    return preds_dicts, ann_path, meta_data

def test_ensemble_evaluation(ensemble_data):
    preds_dicts, ann_path, meta_data = ensemble_data
    pred_eval = PredictionEval()
    preds_coco, names = to_coco(preds_dicts[0], ann_path)
    pred_eval.load_data_coco_files(ann_path, preds_coco, names)
    eval_map = pred_eval.map_query(0.5, 'val')

    boxesensemble = BoxesEnsemble(meta_data, preds_dicts, ann_path)
    boxesensemble.load_pred_eval(ann_path, names)
    ensemble_map = boxesensemble.evaluate_ensemble([1, 0], 1, 'val', 'wbf')
    assert eval_map == ensemble_map

def test_ensemble(ensemble_data):
    preds_dicts, ann_path, meta_data = ensemble_data
    preds_dicts[0] = {'1905.png' : preds_dicts[0]['1905.png']}
    preds_dicts[1] = {'1905.png' : preds_dicts[1]['1905.png']}
    meta_data['val_ids'] = [1905]
    meta_data['val_files'] = ['1905.png']

    boxesensemble = BoxesEnsemble(meta_data, preds_dicts, ann_path)
    boxesensemble.load_pred_eval(ann_path, meta_data)
    ensembled_data = boxesensemble.ensemble([1, 0], 1, skip_box_thr=0.00001, stage='val', ensemble_method='wbf', sigma=0.5)
    a,b =  np.asarray(ensembled_data['1905.png']['bboxes']), np.asarray(preds_dicts[0]['1905.png']['bboxes'])
    sa, sb = np.asarray(ensembled_data['1905.png']['scores']), np.asarray(preds_dicts[0]['1905.png']['scores'])

    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(sa, sb)

def test_ensemble_normalization(ensemble_data):
    preds_dicts, ann_path, meta_data = ensemble_data
    pred_eval = PredictionEval()
    preds_coco, names = to_coco(preds_dicts[0], ann_path)
    pred_eval.load_data_coco_files(ann_path, preds_coco, names)
    eval_map = pred_eval.map_query(0.5, 'val')

    # boxesensemble = BoxesEnsemble(meta_data, preds_dicts, ann_path, confidence_thresholds=[0.1, 100.3])
    boxesensemble = BoxesEnsemble(meta_data, preds_dicts, ann_path)
    boxesensemble.load_pred_eval(ann_path, names)
    ensemble_map1 = boxesensemble.evaluate_ensemble([1, 0], 0.6, 'val', 'wbf', skip_box_thr=0)
    ensemble_map2 = boxesensemble.evaluate_ensemble([0, 1], 0.6, 'val', 'wbf', skip_box_thr=0)

    boxesensemble = BoxesEnsemble(meta_data, preds_dicts, ann_path, confidence_thresholds=[1/1e5, 1e5])
    boxesensemble.load_pred_eval(ann_path, names)
    ensemble_map_norm1 = boxesensemble.evaluate_ensemble([1, 1], 0.6, 'val', 'wbf', skip_box_thr=0)
    boxesensemble = BoxesEnsemble(meta_data, preds_dicts, ann_path, confidence_thresholds=[1e5, 1/1e5])
    boxesensemble.load_pred_eval(ann_path, names)
    ensemble_map_norm2 = boxesensemble.evaluate_ensemble([1, 1], 0.6, 'val', 'wbf', skip_box_thr=0)
    assert abs(ensemble_map_norm1 - ensemble_map1) < 0.001
    assert abs(ensemble_map_norm2 - ensemble_map2) < 0.001
    # assert eval_map == ensemble_map
