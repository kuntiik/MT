from pathlib import Path
from src.datamodules.dental.dental_restorations import COCOSegmentation

def test_coco_segmentation_dataset():
        annotations_path = Path('/datagrid/personal/kuntluka/dental_rtg1/annotations.json')
        imgs_path = Path('/datagrid/personal/kuntluka/dental_rtg1/images')
        dataset = COCOSegmentation(imgs_path, annotations_path, None)
        for i in range(len(dataset)):
            print(i)
            dataset[i]


