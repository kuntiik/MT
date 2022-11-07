from typing import Union
import json
from copy import deepcopy

def load_json_if_need(json_file):
    if type(json_file) == dict:
        return json_file
    with open(json_file, 'r') as f:
        return json.load(f)

def category_to_int(data, category) -> int:
    if type(category)  == int:
        return category
    for cat in data['categories']:
        if cat['name'] == category:
            return  cat['id']


def count_annotations(data, category : Union[int, str]) -> int:
    data = load_json_if_need(data)
    category = category_to_int(data, category)

    count = 0
    for ann in data['annotations']:
        if ann['category_id'] == category:
            count += 1
    return count

def get_num_annotations_dict(data) -> dict:
    data = load_json_if_need(data)
    num_ann = {}
    for category in data['categories']:
        num_ann[category['name']] = count_annotations(data, category['id'])
    return num_ann

def merge_coco_datasets_same_images(datasets, name_modifications=None):
    name_modifications = ["" for _ in datasets] if name_modifications is None else name_modifications
    merged = load_json_if_need(datasets[0])
    cat_id = 0
    for cat in merged['categories']:
        if cat['id'] > cat_id:
            cat_id = cat['id']
    cat_id += 1

    ann_id = 0
    for ann in merged['annotations']:
        if ann['id'] > ann_id:
            ann_id = ann['id']
    ann_id += 1

    for dataset, mod in zip(datasets[1:], name_modifications[1:]):
        dataset = deepcopy(load_json_if_need(dataset))
        for cat in dataset['categories']:
            merged['categories'].append({'id' : cat_id, 'name' : mod + cat['name'] })
            for ann in dataset['annotations']:
                if ann['category_id'] == cat['id']:
                    ann['id'] = ann_id
                    ann_id += 1
                    ann['category_id'] = cat_id
                    merged['annotations'].append(ann)
            cat_id +=1
    return merged


