import json
from pathlib import Path
from typing import Union
from mt.utils.format_handling import load_dict


def to_coco(pred_dict: Union[Path, str, dict], annotations: Union[Path, str, dict]):
    """Converts predictions represented by pred_dict to COCO format. Annotations are required to get
     image names."""
    ann_file = load_dict(annotations)
    preds = load_dict(pred_dict)

    pred_id = 0
    train, val, test, all = [], [], [], []
    for image in ann_file['images']:
        id = image['id']
        if image['file_name'] not in preds:
            continue
        pred = preds[image['file_name']]
        for (bbox, score) in zip(pred['bboxes'], pred['scores']):
            x1, y1, x2, y2 = bbox
            width, height = x2 - x1, y2 - y1
            final_box = [x1, y1, width, height]
            area = width * height
            pred_dict = {'area': area, 'id': pred_id, 'bbox': final_box, 'image_id': id, 'category_id': 1,
                         'score': score}
            all.append(pred_dict)
            pred_id += 1
        if pred['stage'] == 'val':
            val.append(id)
        elif pred['stage'] == 'train':
            train.append(id)
        elif pred['stage'] == 'test':
            test.append(id)

    data = {'categories': [{'supercategory': "", 'name': 'decay', 'id': 1}], 'images': ann_file['images'],
            'annotations': all}
    stage_ids = {'type': 'id', 'train': train, 'val': val, 'test': test}
    return data, stage_ids
