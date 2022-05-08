__all__ = [
    "resnet50_sb",
]

from src.models.mmdet.common.backbone_config import MMDetBackboneConfig
# import src.models.mmdet.common.backbone_config
from src.models.mmdet.common.download_config import mmdet_configs_path


class MMDetFasterRCNNBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="faster_rcnn", **kwargs)


base_config_path = mmdet_configs_path / "resnet_strikes_back"
base_weights_url = "http://download.openmmlab.com/mmdetection/v2.0/resnet_strikes_back"

resnet50_sb = MMDetFasterRCNNBackboneConfig(
    config_path=base_config_path / "faster_rcnn_r50_fpn_rsb-pretrain_1x_coco.py",
    weights_url=""
)

