import pytest
from mt.models.yolov5.backbones import *
import mt.models.yolov5
from mt.utils.path_utils import get_root_dir


@pytest.mark.parametrize(
    "backbone",
    # fmt: off
    [
        small,    medium,    large,    extra_large,
        small_p6, medium_p6, large_p6, extra_large_p6,
    ],
    # fmt: on
)
def test_yolo_model(backbone):
    model = mt.models.yolov5.model(
        num_classes=5, img_size=320, backbone=backbone(pretrained=True)
    )
    weights_path = get_root_dir() / "yolo" / f"{backbone.model_name}.pt"

    assert weights_path.is_file() == True
    assert len(list(model.param_groups())) == 3
    assert model.nc == 4


@pytest.mark.parametrize(
    "backbone",
    [small, medium, large, extra_large],
)
def test_yolo_model_notpretrained(backbone):
    model = mt.models.yolov5.model(
        num_classes=5, img_size=320, backbone=backbone(pretrained=False)
    )

    assert len(list(model.param_groups())) == 3
    assert model.nc == 4
