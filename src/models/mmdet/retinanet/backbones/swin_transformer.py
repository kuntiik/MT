__all__ = [
    "swin_t_p4_w7_fpn_1x_coco",
    "swin_s_p4_w7_fpn_1x_coco",
    "swin_b_p4_w7_fpn_1x_coco",
]

from src.models.mmdet.common.backbone_config import MMDetBackboneConfig
from src.models.mmdet.common.download_config import mmdet_configs_path


class MMDetRetinanetBackboneConfig(MMDetBackboneConfig):
    def __init__(self, **kwargs):
        super().__init__(model_name="retinanet", **kwargs)


base_config_path = mmdet_configs_path.parent / "custom_configs" / "retinanet"

swin_t_p4_w7_fpn_1x_coco = MMDetRetinanetBackboneConfig(
    config_path=base_config_path / "retinanet_swin-t-p4-w7_fpn_1x_coco.py",
    weights_url="",  # There are no pretrained weights available for this model
)

swin_s_p4_w7_fpn_1x_coco = MMDetRetinanetBackboneConfig(
    config_path=base_config_path / "retinanet_swin-s-p4-w7_fpn_1x_coco.py",
    weights_url="",  # There are no pretrained weights available for this model
)

swin_b_p4_w7_fpn_1x_coco = MMDetRetinanetBackboneConfig(
    config_path=base_config_path / "retinanet_swin-b-p4-w7_fpn_1x_coco.py",
    weights_url="",  # There are no pretrained weights available for this model
)
