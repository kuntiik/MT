__all__ = [
    "darknet53",
    "darknet53s"
]

from src.models.mmdet.common.backbone_config import MMDetBackboneConfig
from src.models.mmdet.common.download_config import mmdet_configs_path


class MMDetYOLO3BackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="yolo", **kwargs)


base_config_path = mmdet_configs_path / "yolo"
base_weights_url = "https://download.openmmlab.com/mmdetection/v2.0/yolo"

darknet53 = MMDetYOLO3BackboneConfig(
    config_path=base_config_path / "yolov3_d53_mstrain-608_273e_coco.py",
    weights_url=f"{base_weights_url}/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth"
)

darknet53s = MMDetYOLO3BackboneConfig(
    config_path =base_config_path / "yolov3_d53_320_273e_coco.py",
    weights_url=f"{base_weights_url}/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth"
)
