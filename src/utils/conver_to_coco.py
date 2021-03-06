import json
from pathlib import Path

def to_coco(pred_dict, annotations):
    if type(annotations) == str or type(annotations) == Path:
        with open(annotations, 'r') as f:
            ann_file = json.load(f)
    else:
        ann_file = annotations

    if type(pred_dict) == str or type(pred_dict) == Path:
        with open(pred_dict, 'r') as f:
            preds = json.load(f)
    else:
        preds = pred_dict

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
            pred_dict = {'area' : area, 'id' : pred_id, 'bbox' : final_box, 'image_id' : id, 'category_id' : 1, 'score' : score}
            all.append(pred_dict)
            pred_id += 1
        if pred['stage'] == 'val':
            val.append(id)
        elif pred['stage'] == 'train':
            train.append(id)
        elif pred['stage'] == 'test':
            test.append(id)

    data = {'categories' : [{'supercategory' : "", 'name' : 'decay', 'id' : 1}]}
    data['images'] = ann_file['images']
    data['annotations'] = all
    stage_ids = {'type' : 'id', 'train' : train, 'val' : val, 'test' : test}
    return data, stage_ids
