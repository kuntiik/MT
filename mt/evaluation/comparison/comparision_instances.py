from .core import Comparison


def generate_silver_dataset(num, ids, model1, model2):
    results = []
    comparison = Comparison()

    for id in ids:
        if id < 10:
            comparison.parse_coco_data('/datagrid/personal/kuntluka/dental_rtg_test/test_annotations.json')
            comparison.load_coco_data(id, False)
        elif id == 10:
            comparison.load_json_data(model1, False)
        elif id == 11:
            comparison.load_json_data(model2, False)
        comparison.parse_coco_data('silver_' + str(num) + '.json')
        comparison.load_coco_data([1, 2], True)
        _, fp, _ = comparison.pairwise_evaluate()
        comparison.load_coco_data(1, True)
        iou, _, fn = comparison.pairwise_evaluate()
        res = [iou, fp, fn]
        results.append(res)

    return results
