import numpy as np
from src.evaluation.pycocotools.coco import COCO
from src.evaluation.pycocotools.cocoeval import COCOeval, Params


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

        self.cocoGt = COCO(annotations_path)
        for image in self.cocoGt.imgs.values():
            self.img_name2id[image["file_name"]] = image["id"]
            self.img_id2name[image["id"]] = image["file_name"]

        if type(predictions_coco) == dict and 'annotations' in predictions_coco.keys():
            predictions_anns = predictions_coco['annotations']
        else:
            predictions_anns = predictions_coco
        self.cocoDt = self.cocoGt.loadRes(predictions_anns)
        self.cocoEval = COCOeval(self.cocoGt, self.cocoDt)
        if train_val_names is not None:
            if train_val_names["type"] == "id":
                self.train_ids = train_val_names["train"]
                self.val_ids = train_val_names["val"]
                self.test_ids = train_val_names["test"]
            elif train_val_names["type"] == "names":
                self.train_ids = [self.img_name2id[name] for name in train_val_names["train"]]
                self.val_ids = [self.img_name2id[name] for name in train_val_names["val"]]
                self.tests_ids = [self.img_name2id[name] for name in train_val_names["test"]]

    def default_queries(self):
        queries = [
            {"ap": 1},
            {"ap": 1, "iouThr": 0.1, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.3, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.7, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.9, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "small", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "medium", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "large", "maxDets": 100},
        ]
        return queries

    def get_latex_table(self,name="", stages=['test']):
        queries = [
            {"ap": 1},
            {"ap": 1, "iouThr": 0.3, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.75, "areaRng": "all", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "small", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "medium", "maxDets": 100},
            {"ap": 1, "iouThr": 0.5, "areaRng": "large", "maxDets": 100},
        ]

        text = ""
        if 'val' in stages:
            val_results = self.evaluate_map(queries, stage="val", verbose=False)
            val_results = [round(res, 3) for res in val_results]
        if 'train' in stages:
            train_results = self.evaluate_map(queries, stage="train", verbose=False)
            train_results = [round(res, 3) for res in train_results]
        if 'test' in stages:
            test_results = self.evaluate_map(queries, stage="test", verbose=False)
            test_results = [round(res, 3) for res in test_results]
            text = f"""{name} & {test_results[0]}& {test_results[1]} & {test_results[2]} & {test_results[3]} & {test_results[4]} & {test_results[5]} & {test_results[6]} \\\\ \\hline"""

        # text = f"""stage  & AP & AP@.3 & AP@.5 & AP@.75 & AP@.5_S & AP@.5_M & AP@.5_L \\ \hline
        # training & {train_results[0]}& {train_results[1]} & {train_results[2]} & {train_results[3]}
        # & {train_results[4]} & {train_results[5]} & {train_results[6]} \\hline
        # validation & {val_results[0]}& {val_results[1]} & {val_results[2]} & {val_results[3]}
        # & {val_results[4]} & {val_results[5]} & {val_results[6]} \\hline
        # """
        return text

    def get_data(self, name):
        text = self.get_latex_table(name=name, stages=['test'])
        precisions = self.precision_by_iou(stage="test")
        return text, precisions


    def indices_by_stage(self, stage):
        evaluate_imgs = []
        if stage == "val":
            evaluate_imgs = self.val_ids
        elif stage == "train":
            evaluate_imgs = self.train_ids
        elif stage == "test":
            evaluate_imgs = self.test_ids
        else:  # stage == 'all':
            evaluate_imgs = list(self.img_id2name.keys())
        return evaluate_imgs

    def evaluate_map(self, queries, stage="all", summary=False, verbose=True):
        evaluate_imgs = self.indices_by_stage(stage)

        tmp_map_params = self.map_params
        tmp_map_params.imgIds = evaluate_imgs
        self.cocoEval.params = tmp_map_params
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        results = []
        for query in queries:
            if verbose:
                print(self.cocoEval._summarize(**query))
            results.append(self.cocoEval._summarize(**query))
        if summary:
            self.cocoEval.summarize()
        return results

    def precision_by_iou(self, iouThr=0.5, stage="all", areRng="all"):
        evaluate_imgs = self.indices_by_stage(stage)

        area_idx = self.cocoEval.params.area_str2idx(areRng)
        iou_idx = np.where(self.cocoEval.params.iouThrs == iouThr)[0][0]

        self.cocoEval.params.imgIds = evaluate_imgs
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        precisions = self.cocoEval.eval["precision"][iou_idx, :, 0, area_idx, 2]
        recalls = np.linspace(0,1,101)
        confidences = self.cocoEval.eval["scores"][iou_idx, :, 0, area_idx, 2]
        f1_score = [(p*r)/(p+r)*2 for p,r in zip(precisions, recalls)]
        best_idx = np.argmax(f1_score)
        text = f"""{round(precisions[best_idx],3)} & {round(recalls[best_idx],3)} & {round(f1_score[best_idx],3)} & {round(confidences[best_idx],3)}"""
        return text

    def map_query(self, iouThr=0.5, stage='all', areaRng='all'):
        evaluate_imgs = self.indices_by_stage(stage)
        area_idx = self.cocoEval.params.area_str2idx(areaRng)
        iou_idx = np.where(self.cocoEval.params.iouThrs == iouThr)[0][0]
        self.cocoEval.params.imgIds = evaluate_imgs
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        query = {"ap": 1, "iouThr": iouThr, "areaRng": areaRng, "maxDets": 100}
        return self.cocoEval._summarize(**query, doprint=False)

    def evaluate_img_by_name(self, img_name):
        self.cocoEval.params.imgIds = [self.img_name2id[img_name]]
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        query = {"ap": 1, "iouThr": 0.5, "areaRng": 'all', "maxDets": 100}
        # query = {"ap": 1},
        return self.cocoEval._summarize(**query, doprint=False)
