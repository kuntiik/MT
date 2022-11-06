from mt.evaluation.boxes_fusion.ensemble_boxes_wbf import weighted_boxes_fusion


def test_weighted_boxes_fusion():

    boxes_list = [[
        [0.00, 0.51, 0.81, 0.91],
        [0.10, 0.31, 0.71, 0.61],
        [0.01, 0.32, 0.83, 0.93],
        [0.02, 0.53, 0.11, 0.94],
        [0.03, 0.24, 0.12, 0.35],
    ], [
        [0.04, 0.56, 0.84, 0.92],
        [0.12, 0.33, 0.72, 0.64],
        [0.38, 0.66, 0.79, 0.95],
        [0.08, 0.49, 0.21, 0.89],
    ]]
    scores_list = [[0.9, 0.8, 0.2, 0.4, 0.7], [0.5, 0.8, 0.7, 0.3]]
    labels_list = [[0, 1, 0, 1, 1], [1, 1, 1, 0]]
    weights = [1, 1]

    iou_thr = 0.5
    skip_box_thr = 0.0001
    sigma = 0.1

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                  iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    weights = [[2, 1, 1], [1, 1, 1]]
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                  iou_thr=iou_thr, skip_box_thr=skip_box_thr)
