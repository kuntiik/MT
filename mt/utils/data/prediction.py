import numpy as np

def filter_by_confidence(data, confidence_threshold):
    new_data = {}
    for key, value in data.items():
        bboxes = np.asarray(data[key]['bboxes'])
        scores = np.asarray(data[key]['scores'])
        labels = np.asarray(data[key]['labels'])
        stage = data[key]['stage']
        indices = scores >= confidence_threshold
        temp_data = {'bboxes': bboxes[indices], 'scores': scores[indices], 'labels': labels[indices],
                     'stage': stage}
        new_data[key] = temp_data
    return new_data

def add_preds(data, preds, key):
    data[key] = []
    for _, value in preds.items():
        for box in value['bboxes']:
            x1, y1, x2, y2 = box
            new_box = [x1, x2, x2-x1, y2-y1]
            data[key].append(new_box)
    return data