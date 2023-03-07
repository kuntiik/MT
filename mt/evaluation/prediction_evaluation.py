from typing import Dict, List, Union

import numpy as np

from mt.evaluation.pycocotools.coco import COCO
from mt.evaluation.pycocotools.cocoeval import COCOeval, Params


class PredictionEval:
    def __init__(self, map_params=Params()):
        self.map_params = map_params
        self._extend_map_params()
        self.img_name2id = {}
        self.img_id2name = {}

    def _extend_map_params(self):
        params = Params()
        params.iouThrs = np.round(
            np.linspace(0.05, 0.95, int((0.95 - 0.05) / 0.05) + 2, endpoint=True), 3
        )
        self.map_params = params

    def load_data_coco_files(self, annotations_path, predictions_coco, train_val_names=None):
        """Loads both predictions and ground truth. The data should be in the COCO format"""
        self.load_ground_truth(annotations_path, train_val_names)
        self.load_predictions(predictions_coco)

    def load_ground_truth(self, annotations_path : str, train_val_names=None) -> None:
        """Loads ground truth annotations given by annotations path"""
        self.cocoGt = COCO(annotations_path)
        for image in self.cocoGt.imgs.values():
            self.img_name2id[image["file_name"]] = image["id"]
            self.img_id2name[image["id"]] = image["file_name"]

        # TODO this is not clean at all
        if 'train_ids' in train_val_names.keys():
            self.train_ids = train_val_names["train_ids"]
            self.val_ids = train_val_names["val_ids"]
            self.test_ids = train_val_names["test_ids"]

        elif train_val_names is not None:
            if train_val_names["type"] == "id":
                self.train_ids = train_val_names["train"]
                self.val_ids = train_val_names["val"]
                self.test_ids = train_val_names["test"]
            elif train_val_names["type"] == "file_name":
                self.train_ids = [self.img_name2id[name] for name in train_val_names["train_files"]]
                self.val_ids = [self.img_name2id[name] for name in train_val_names["val_files"]]
                self.tests_ids = [self.img_name2id[name] for name in train_val_names["test_files"]]

    def load_predictions(self, predictions_coco: Union[Dict, List]) -> None:
        """Loads predictions of a model in the COCO format."""
        if type(predictions_coco) == dict and 'annotations' in predictions_coco.keys():
            predictions_anns = predictions_coco['annotations']
        else:
            predictions_anns = predictions_coco
        self.cocoDt = self.cocoGt.loadRes(predictions_anns)
        self.cocoEval = COCOeval(self.cocoGt, self.cocoDt)

    @staticmethod
    def default_queries(recall: bool = False):
        """Returns default queries for AP if recall is set to False, otherwise for AR"""
        if recall:
            queries = [
                {"ap": 0, "maxDets": 100},
                {"ap": 0, "iouThr": 0.5, "areaRng": "all", "maxDets": 10},
                {"ap": 0, "iouThr": 0.5, "areaRng": "all", "maxDets": 100},
                {"ap": 0, "iouThr": 0.75, "areaRng": "all", "maxDets": 100},
                {"ap": 0, "iouThr": 0.5, "areaRng": "small", "maxDets": 100},
                {"ap": 0, "iouThr": 0.5, "areaRng": "medium", "maxDets": 100},
                {"ap": 0, "iouThr": 0.5, "areaRng": "large", "maxDets": 100},
            ]
        else:
            queries = [
                {"ap": 1},
                {"ap": 1, "iouThr": 0.3, "areaRng": "all", "maxDets": 100},
                {"ap": 1, "iouThr": 0.5, "areaRng": "all", "maxDets": 100},
                {"ap": 1, "iouThr": 0.75, "areaRng": "all", "maxDets": 100},
                {"ap": 1, "iouThr": 0.5, "areaRng": "small", "maxDets": 100},
                {"ap": 1, "iouThr": 0.5, "areaRng": "medium", "maxDets": 100},
                {"ap": 1, "iouThr": 0.5, "areaRng": "large", "maxDets": 100},
            ]
        return queries

    def get_latex_table(self, name: str = "", stage: str = 'test', recall: bool = False):
        """Generates row of latex table with results for given stage. Recall parameter sets if values will
         be for AP or AR. Name sets the first column of the latex row.
        """
        queries = self.default_queries(recall)
        text = ""
        results = self.evaluate_map(queries, stage=stage, verbose=False)
        results = [round(res, 3) for res in results]
        if recall:
            text = f"""{name} & {results[0]}& {results[1]} & {results[2]} & {results[3]} & {results[4]} & {results[5]} \\ \hline"""
        else:
            text = f"""{name} & {results[0]}& {results[1]} & {results[2]} & {results[3]} & {results[4]} & {results[5]} & {results[6]} \\ \hline"""
        return text

    # def get_data(self, name):
    #     text = self.get_latex_table(name=name, stages=['test'])
    #     text_recall = self.get_latex_table_recall(name=name, stages=['test'])
    #     precisions = self.precision_by_iou(stage="test")
    #     return text, precisions, text_recall

    def indices_by_stage(self, stage: str):
        """Get image indices corresponding to given stage"""
        if stage == "val":
            img_ids = self.val_ids
        elif stage == "train":
            img_ids = self.train_ids
        elif stage == "test":
            img_ids = self.test_ids
        else:  # stage == 'all':
            img_ids = list(self.img_id2name.keys())
        return img_ids

    def evaluate_by_stage(self, stage: str):
        """Evaluates the loaded data for the given stage"""
        self.cocoEval.params.iouThrs = np.round(
            np.linspace(0.05, 0.95, int((0.95 - 0.05) / 0.05) + 2, endpoint=True), 3
        )
        self.cocoEval.params.imgIds = self.indices_by_stage(stage)
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()


    def evaluate_map(self, queries: List[Dict], stage: str = "all", summary: bool = False, verbose: bool = True):
        """Evaluates AP for given queries on given stage."""
        self.evaluate_by_stage(stage)
        results = []
        for query in queries:
            if verbose:
                print(self.cocoEval._summarize(**query))
            results.append(self.cocoEval._summarize(**query))
        if summary:
            self.cocoEval.summarize()
        return results

    def results_maximizing_fscore(self, iouThr: float = 0.5, stage: str = "all", areRng: str = "all",
                                  decimals: int = 3) -> dict:
        """Returns precision, recall, F-score and confidence for the highest possible F-score on the given stage."""
        area_idx = self.cocoEval.params.area_str2idx(areRng)
        iou_idx = np.where(self.cocoEval.params.iouThrs == iouThr)[0][0]
        self.evaluate_by_stage(stage)
        precisions = self.cocoEval.eval["precision"][iou_idx, :, 0, area_idx, 2]
        recalls = np.linspace(0, 1, 101)
        confidences = self.cocoEval.eval["scores"][iou_idx, :, 0, area_idx, 2]
        fscore = [(p * r) / (p + r) * 2 for p, r in zip(precisions, recalls)]
        best_idx = np.argmax(fscore)
        return {
            'precision': round(precisions[best_idx], decimals),
            'recall': round(recalls[best_idx], decimals),
            'fscore': round(fscore[best_idx], decimals),
            'confidence': round(confidences[best_idx], decimals),
        }

    def results_latex_table(self, iouThr=0.5, stage="all", areRng="all") -> str:
        """Generates row of latex table for results maximizing F-score"""
        results = self.results_maximizing_fscore(iouThr, stage, areRng, 3)
        text = f"""{results['precision']} & {results['recall']} & {results['fscore']} & {results['confidence']}"""
        return text

    def precision_recall_score(self, iouThr=0.5, stage='all', areaRng='all'):
        area_idx = self.cocoEval.params.area_str2idx(areaRng)
        iou_idx = np.where(self.cocoEval.params.iouThrs == iouThr)[0][0]
        self.evaluate_by_stage(stage)
        precisions = self.cocoEval.eval["precision"][iou_idx, :, 0, area_idx, 2]
        recalls = np.linspace(0, 1, 101)
        confidences = self.cocoEval.eval["scores"][iou_idx, :, 0, area_idx, 2]
        return precisions, recalls, confidences

    def map_query(self, iouThr: float = 0.5, stage: str = 'all', areaRng: str = 'all'):
        """Returns AP for given IOU threshold on the given stage for predictions with specified area."""
        self.evaluate_by_stage(stage)
        query = {"ap": 1, "iouThr": iouThr, "areaRng": areaRng, "maxDets": 100}
        return self.cocoEval._summarize(**query, doprint=False)

    def evaluate_img_by_name(self, img_name: str):
        self.cocoEval.params.imgIds = [self.img_name2id[img_name]]
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        query = {"ap": 1, "iouThr": 0.5, "areaRng": 'all', "maxDets": 100}
        return self.cocoEval._summarize(**query, doprint=False)
