import torch
from pathlib import Path
import json
from ensembling_search import *
from src.utils.conver_to_coco import to_coco
import ensembling_search
from tqdm import tqdm
import argparse

import os, sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def search(method):
    results = torch.zeros((7, 7, 7, 7, 7))
    pred_path = Path('ens_search')
    ann_path = Path('/datagrid/personal/kuntluka/dental_rtg3/annotations.json')

    preds = list(pred_path.iterdir())
    preds = [pred.name for pred in preds]

    with open('confidence_thresholds.json', 'r') as f:
        confidence_thr = json.load(f)
    confidences = [confidence_thr[p] for p in preds]

    preds_dicts = []
    for pred_name in preds:
        with open(pred_path / pred_name, 'r') as f:
            preds_dicts.append(json.load(f))

    with open('metadata4000.json', 'r') as f:
        meta_data = json.load(f)

    boxesensemble = ensembling_search.BoxesEnsemble(meta_data, preds_dicts, ann_path, confidences)
    boxesensemble.load_pred_eval(ann_path, meta_data)
    for i in tqdm(range(7)):
        for j in tqdm(range(7)):
            for k in range(7):
                for l in range(7):
                    for m in range(7):
                        results[i, j, k, l, m] = boxesensemble.evaluate_ensemble([
                            i * 0.15 + 0.1,
                            j * 0.15 + 0.1,
                            k * 0.15 + 0.1,
                            l * 0.15 + 0.1,
                        ], m * 0.05 + 0.40
                            , ensemble_method=method)
    return results

def search_soft_nms(method='snms'):
    results = torch.zeros((5,5,5,5,5,6))
    pred_path = Path('ens_search')
    ann_path = Path('/datagrid/personal/kuntluka/dental_rtg3/annotations.json')

    preds = list(pred_path.iterdir())
    preds = [pred.name for pred in preds]

    with open('confidence_thresholds.json', 'r') as f:
        confidence_thr = json.load(f)
    confidences = [confidence_thr[p] for p in preds]

    preds_dicts = []
    for pred_name in preds:
        with open(pred_path / pred_name, 'r') as f:
            preds_dicts.append(json.load(f))

    with open('metadata4000.json', 'r') as f:
        meta_data = json.load(f)

    boxesensemble = ensembling_search.BoxesEnsemble(meta_data, preds_dicts, ann_path, confidences)
    boxesensemble.load_pred_eval(ann_path, meta_data)
    for i in tqdm(range(5)):
        for j in tqdm(range(5)):
            for k in range(5):
                for l in range(5):
                    for m in range(5):
                        for n in range(6):
                            results[i, j, k, l, m] = boxesensemble.evaluate_ensemble([
                                i * 0.22 + 0.12,
                                j * 0.22 + 0.12,
                                k * 0.22 + 0.12,
                                l * 0.22 + 0.12,
                                ], m * 0.10 + 0.30
                                , ensemble_method=method, sigma=0.3 +0.1*n)
    return results


argparser = argparse.ArgumentParser()

argparser.add_argument('-m', '--method', type=str, default='wbf')
args = argparser.parse_args()

if __name__ == '__main__':
    with HiddenPrints():
        if args.method == 'snms':
            res = search_soft_nms()
        else:
            res = search(args.method)
    torch.save(res, args.method + '_ens_results.pt')
