import numpy as np
import torch
from torchvision.ops import box_iou
from typing import List
from .core import Centroid
from .utils import get_areas_negative, get_areas, merge_dataset_per_img


def get_index(cumsum, i):
    index = 0
    last_c = 0
    for c in cumsum:
        if c > i:
            break
        index += 1
        last_c = c
    return index, i - last_c


def empty_boxes_fix(boxes):
    for i, box in enumerate(boxes):
        if len(box) == 0:
            boxes[i] = np.empty((0, 4))

def generate_silver_dataset(data : dict, expert : int, expert_ids : List[int]) -> dict:
    """
    Generates silver standard dataset from a COCO dataset

    :param data: Dataset in coco format
    :param expert: The id for which the dataset would be generated
    :param expert_ids: List of all expert ids to include in the dataset generation
    :return: generated dataset in COCO format
    """
    ann_id = 0
    merged_dataset = merge_dataset_per_img(data, expert_ids)
    new_data = {'annotations': [], 'images': data['images'], 'categories': [{'id': 1, 'name': 'two_plus'}, {'id': 2, 'name': 'single'}]}
    for img in range(1, 101):

        bboxes = [merged_dataset[img][i] for i in range(len(expert_ids)) if i != expert]
        multi_match, single_match = silver_dataset_image(bboxes)
        for mm in multi_match:
            new_data['annotations'].append({'bbox': mm.tolist(), 'id': ann_id, 'category_id': 1, 'image_id': img})
            ann_id += 1

        for sm in single_match:
            new_data['annotations'].append({'bbox': sm.tolist(), 'id': ann_id, 'category_id': 2, 'image_id': img})
            ann_id += 1
    return new_data


def silver_dataset_image(boxes):
    empty_boxes_fix(boxes)
    all_boxes = np.concatenate(boxes, axis=0)
    n_boxes = all_boxes.shape[0]
    already_assigned = np.ones((1, n_boxes)).flatten()

    areas_boxes = get_areas_negative(all_boxes)
    sorted_indices = np.argsort(areas_boxes)

    sizes = [len(box) for box in boxes]
    sizes_cumulative = np.cumsum(sizes)
    indices_dict = {i: get_index(sizes_cumulative, i) for i in range(n_boxes)}
    single_match = []
    multi_match = []
    while np.count_nonzero(already_assigned) > 0:
        available_boxes = [ids for ids in sorted_indices if already_assigned[ids]]
        max_id = available_boxes[0]
        box = all_boxes[max_id]
        box_centroid = Centroid(box, coco_format=True)
        already_assigned[max_id] = 0

        available_boxes = [ids for ids in sorted_indices if already_assigned[ids]]
        group_id = indices_dict[max_id][0]
        # TODO is hardoced
        groups = [[] for _ in range(len(boxes))]
        groups_ids = [[] for _ in range(len(boxes))]
        for ab in available_boxes:
            groups[indices_dict[ab][0]].append(all_boxes[ab])
            groups_ids[indices_dict[ab][0]].append(ab)
        matched_boxes = []
        num_matched_boxes = 0
        for i in range(len(boxes)):
            if i != group_id:
                group_boxes = np.asarray(groups[i])
                box_ids = np.asarray(groups_ids[i])
                matches = box_centroid.match_criterion(group_boxes)
                num_matches = np.count_nonzero(matches)
                # print(num_matches)
                if num_matches == 1:
                    matched_boxes.append(group_boxes[matches == 1])
                    assigned_id = box_ids[matches == 1]
                    already_assigned[assigned_id] = 0
                    num_matched_boxes += 1
                elif num_matches > 1:
                    a = torch.Tensor(box).unsqueeze(0)
                    b = torch.Tensor(group_boxes[matches == 1])
                    a[0, 2] = a[0, 0] + a[0, 2]
                    a[0, 3] = a[0, 1] + a[0, 3]
                    b[:, 2] = b[:, 0] + b[:, 2]
                    b[:, 3] = b[:, 1] + b[:, 3]
                    ious = box_iou(a, b).numpy()
                    max_index = np.argmax(ious)
                    matched_boxes.append([group_boxes[matches == 1][max_index]])
                    assigned_id = box_ids[matches == 1][max_index]
                    already_assigned[assigned_id] = 0
                    num_matched_boxes += 1
        if num_matched_boxes > 0:
            matched_boxes.append(box)
            multi_match.append(np.average(matched_boxes, axis=0)[0])
        elif num_matched_boxes == 0:
            single_match.append(box)
    return multi_match, single_match
