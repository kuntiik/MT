from typing import Union, List
from pathlib import Path
import json
from torchvision.ops import box_iou
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from .common import Centroid


def exclude_row_col(matrix, row, col):
    matrix = torch.cat((matrix[:row], matrix[row + 1:]))
    matrix = torch.cat((matrix[:, :col], matrix[:, col + 1:]), dim=1)
    return matrix



class Comparison:
    def __init__(self):
        self.coco_data = None
        self.first = None
        self.second = None

    def parse_coco_data(self, data: Union[Path, str, dict]):
        if type(data) == str or type(data) == Path:
            with open(data, 'r') as f:
                data = json.load(f)
        self.coco_data = data

    def load_coco_data(self, category: Union[str, int, List[int]], first=False):
        assert self.coco_data is not None
        if type(category) == str:
            for cat in self.coco_data['categories']:
                if cat['name'] == category:
                    category = cat['id']
                    break
        if first:
            self.first = self._from_coco(self.coco_data, category)
        else:
            self.second = self._from_coco(self.coco_data, category)

    def load_json_data(self, data, first=False):
        if first:
            self.first = self._from_json(data)
        else:
            self.second = self._from_json(data)

    @staticmethod
    def _from_json(data):
        if type(data) == str or type(data) == Path:
            with open(data, 'r') as f:
                data = json.load(f)
        new_data = {}
        for key, value in data.items():
            new_data[key] = value['bboxes']
        return new_data

    @staticmethod
    def _from_coco(data, category_id):
        if type(category_id) == int: category_id = [category_id]
        new_data = {}
        for img in data['images']:
            new_data[img['file_name']] = []
            img_id = img['id']
            for ann in data['annotations']:
                if ann['image_id'] == img_id and ann['category_id'] in category_id:
                    x, y, w, h = ann['bbox']
                    new_data[img['file_name']].append([x, y, x + w, y + h])
        return new_data

    def _get_iou_matrix(self, img):
        b1 = torch.tensor(self.first[img])
        b2 = torch.tensor(self.second[img])
        if b1.size(0) == 0 or b2.size(0) == 0:
            # return max(b1.size(0), b2.size(0))
            return b1.size(0), b2.size(0)
        return box_iou(b1, b2)

    def assign_boxes_centroids(self, key):
        b1 = self.first[key]
        b2 = self.second[key]
        c1 = [Centroid(box) for box in b1]
        c2 = [Centroid(box) for box in b2]

        s = min(len(b1), len(b2))
        if s == 0:
            return 0, len(b2), len(b1), s

        pairwise_c1 = np.asarray([c.inside_batch(b2) for c in c1])
        pairwise_c2 = np.asarray([c.inside_batch(b1) for c in c2])

        cost_matrix = np.logical_and(pairwise_c1, pairwise_c2.T)
        row_indices, col_indices = linear_sum_assignment(cost_matrix, maximize=True)
        valid_assignment = [pair for pair in zip(row_indices, col_indices) if cost_matrix[pair]]
        # TODO use the assignments itself
        correct = len(valid_assignment)
        fn = len(b1) - correct
        fp = len(b2) - correct
        return correct, fp, fn, s
        # return valid_assignment

    def assign_boxes_area(self, key, threshold, return_fn_fp=False):
        ious = []
        iou_matrix = self._get_iou_matrix(key)
        if type(iou_matrix) == tuple:
            if return_fn_fp:
                fn, fp = iou_matrix
                return ious, fp, fn, 0
            return ious, max(iou_matrix), 0
        s = min(iou_matrix.size(0), iou_matrix.size(1))

        while iou_matrix.size(0) * iou_matrix.size(1) > 0:
            max_val = torch.max(iou_matrix)
            if max_val <= threshold:
                break
            ious.append(max_val)
            row, col = torch.nonzero(iou_matrix == torch.max(iou_matrix))[0, :]
            iou_matrix = exclude_row_col(iou_matrix, row, col)
        if return_fn_fp:
            fn = iou_matrix.size(0)
            fp = iou_matrix.size(1)
            return ious, fp, fn, s
        else:
            e = iou_matrix.size(0) + iou_matrix.size(1)
            return ious, e, s

    def pairwise_evaluate(self):
        ious, errors, sizes = [], [], []
        for key in self.first.keys():
            iou, e, s = self.assign_boxes_area(key, 0.0)
            iou = [0] if len(iou) == 0 else iou
            ious.append(sum(iou) / len(iou))
            errors.append(e)
            sizes.append(s)
        ious_scaled = [i * s for (i, s) in zip(ious, sizes)]
        iou_avg = sum(ious_scaled) / sum(sizes)
        return float(iou_avg), sum(errors)

    def precision_evaluate(self, criterion='area', iou_threshold=0.5):
        TP = 0
        FN = 0
        FP = 0
        for key in self.first.keys():
            if criterion == 'area':
                iou, fp, fn, _ = self.assign_boxes_area(key, iou_threshold, return_fn_fp=True)
                TP += len(iou)
            if criterion == 'centroids':
                c, fp, fn, _ = self.assign_boxes_centroids(key)
                TP += c
            FN += fn
            FP += fp
        return TP, FN, FP


# class PairwiseComparison(Comparison):

if __name__ == '__main__':
    comparison = Comparison()
    criterion = 'centroids'
    iou_threshold = 0.3
    comparison.parse_coco_data('/datagrid/personal/kuntluka/dental_rtg_test/silver_dataset.json')
    comparison.load_coco_data([1], True)
    comparison.parse_coco_data('/datagrid/personal/kuntluka/dental_rtg_test/test_annotations.json')
    comparison.load_coco_data(7, False)
    count = 0
    for key, value in comparison.first.items():
        count += len(value)
    print("count", count)
    count = 0
    for key, value in comparison.second.items():
        count += len(value)
    print("count", count)
    tp, fn, fp = comparison.precision_evaluate(criterion=criterion, iou_threshold=iou_threshold)
    print(tp, fn, fp)
