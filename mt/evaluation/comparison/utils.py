from typing import List
import numpy as np

def reorder(data):
    """Helper function, that reorders data from all subject to order used in CBM paper"""
    return [data[-2], data[-1], data[6], data[0], data[1], data[7], data[8], data[2], data[3], data[5]]

def get_areas_negative(boxes):
    areas = []
    for box in boxes:
        x1, y1, x2, y2 = box
        areas.append(-(x2 - x1) * (y2 - y1))
    return areas


def get_areas(boxes):
    areas = []
    for box in boxes:
        x1, y1, x2, y2 = box
        areas.append(-(x2 - x1) * (y2 - y1))
    return areas

def count_boxes(data : dict) -> int:
    """
    Counts number of bboxes in prediction dataset where data is dict with keys, that are image names and values are bboxes, scores, labels
    :param data:
    :return number of boxes in the dataset:
    """
    count = 0
    for key, value in data.items():
        count += len(value['bboxes'])
    return count


def filter_by_confidence(data: dict, confidence_threshold: float) -> dict:
    """
    Given prediction data in dict format with keys : image names and values : bboxes, scores, labels.
    Filter out predictions with lower confidence, than given threshold
    :param data:
    :param confidence_threshold:
    :return filtered data by the confidence threshold:
    """

    new_data = {}
    for key, value in data.items():
        bboxes = np.asarray(data[key]['bboxes'])
        scores = np.asarray(data[key]['scores'])
        labels = np.asarray(data[key]['labels'])
        # stage = np.asarray(data['stage'])
        stage = data[key]['stage']
        indices = scores >= confidence_threshold
        temp_data = {'bboxes': bboxes[indices], 'scores': scores[indices], 'labels': labels[indices], 'stage': stage}
        new_data[key] = temp_data
    return new_data


def merge_dataset_per_img(data, category_ids: List[int]) -> dict:
    """
    Generates merged dataset as a dictionary, where keys are indices of the image and values are Lists of Lists of bounding boxes
    :param data:
    :param category_ids:
    :return dict, key : img id, value : len(category_ids) Lists, each containing List[List[float, float, float, float]:
    """
    merged_dataset = {}
    for i in range(1, 101):
        merged_dataset[i] = []
        for _ in category_ids:
            merged_dataset[i].append([])

    for ann in data['annotations']:
        if ann['category_id'] in category_ids:
            merged_dataset[ann['image_id']][category_ids.index(ann['category_id'])].append(ann['bbox'])
    return merged_dataset
