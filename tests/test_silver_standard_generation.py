import json
from src.evaluation.comparison.silver_standard_generation import generate_silver_dataset

def test_compose_three():
    with open('/datagrid/personal/kuntluka/dental_rtg_test/test_annotations.json', 'r') as f:
        data = json.load(f)
    expert_ids = [2,8,9]
    id = 2
    data = generate_silver_dataset(data, id, expert_ids)
    two_plus = 0
    single = 0
    for ann in data['annotations']:
        if ann['category_id'] == 1:
            two_plus += 1
        elif ann['category_id'] == 2:
            single += 1
    print(f"Number of two matches is {two_plus} and number of single matches is {single}")