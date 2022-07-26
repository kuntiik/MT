import pytest
from src.evaluation.boxes_fusion import weighted_boxes_fusion, nms
import numpy as np


# Boxes are in x1, y1, x2, y2 relative format
@pytest.fixture()
def boxes_data():
    boxes_list = [[
        [0.0, 0.0, 0.5, 0.3],  # cluster 1
        [0.0, 0.0, 0.4, 0.3],  # cluster 1
        [0.5, 0.0, 1, 0.3],  # cluster 2
        [0.5, 0.0, 0.9, 0.3],  # cluster 2
        [0.3, 0.2, 0.6, 0.5],  # cluster 3
        [0.2, 0.2, 0.6, 0.5]  # cluster 3
    ], [
        [0.05, 0.0, 0.5, 0.3],  # cluster 1
        [0.6, 0.0, 1, 0.3],  # cluster 2
        [0.2, 0.2, 0.7, 0.5]  # cluster 3
    ]]
    scores_list = [[0.9, 0.3, 0.7, 0.75, 0.2, 0.5], [0.4, 0.8, 0.3]]
    labels_list = [[1, 1, 1, 1, 1, 1], [1, 1, 1]]
    yield boxes_list, scores_list, labels_list


def test_wbf(boxes_data):
    boxes_list, scores_list, labels_list = boxes_data
    # boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.5, conf_type='box_and_model_avg')
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.5, allows_overflow=False)
    b1 = np.asarray(boxes_list[0])
    b2 = np.asarray(boxes_list[1])
    n1 = (scores_list[0][0] + scores_list[0][1] + scores_list[1][0]) / 3
    n2 = (scores_list[0][2] + scores_list[0][3] + scores_list[1][1]) / 3
    n3 = (scores_list[0][4] + scores_list[0][5] + scores_list[1][2]) / 3
    c1 = (b1[0]*scores_list[0][0] + b1[1]*scores_list[0][1] + b2[0]*scores_list[1][0]) / (scores_list[0][0] + scores_list[0][1] + scores_list[1][0])
    c2 = (b1[2]*scores_list[0][2] + b1[3]*scores_list[0][3] + b2[1]*scores_list[1][1]) / (scores_list[0][2] + scores_list[0][3] + scores_list[1][1])
    c3 = (b1[4]*scores_list[0][4] + b1[5]*scores_list[0][5] + b2[2]*scores_list[1][2]) / (scores_list[0][4] + scores_list[0][5] + scores_list[1][2])
    result_labels = [1, 1, 1]
    result_scores = np.asarray([n1, n2, n3])
    indexes = np.argsort(-result_scores)
    result_scores = result_scores[indexes]
    result_boxes = np.array([c1, c2, c3])[indexes]
    np.testing.assert_allclose(boxes, result_boxes)
    np.testing.assert_allclose(scores, result_scores)
    np.testing.assert_allclose(labels, result_labels)



def test_nms(boxes_data):
    boxes_list, scores_list, labels_list = boxes_data
    boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr=0.5)
    result_boxes = [[0.0, 0.0, 0.5, 0.3],
                    [0.6, 0.0, 1, 0.3],
                    [0.2, 0.2, 0.6, 0.5]]
    result_score = [0.9, 0.8, 0.5]
    result_labels = [1, 1, 1]
    np.testing.assert_allclose(boxes, result_boxes)
    np.testing.assert_allclose(scores, result_score)
    np.testing.assert_allclose(labels, result_labels)
