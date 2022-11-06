__all__ = ["model"]

import torch.nn as nn
from mt.models.mmdet.common.backbone_config import MMDetBackboneConfig
from typing import Optional, Union
from pathlib import Path

from mt.models.mmdet.common.utils import build_model


def model(
        backbone: MMDetBackboneConfig,
        num_classes: int,
        checkpoints_path: Optional[Union[str, Path]] = "checkpoints",
        force_download=False,
        cfg_options=None,
) -> nn.Module:
    return build_model(
        model_type="two_stage_detector_bbox",
        backbone=backbone,
        num_classes=num_classes,
        pretrained=backbone.pretrained,
        checkpoints_path=checkpoints_path,
        force_download=force_download,
        cfg_options=cfg_options,
    )
