import pytest
import torch
import torch.nn as nn

from src.models import efficientdet
from src.models.efficientdet.backbones import tf_lite0


@pytest.fixture()
def fridge_efficientdet_model() -> nn.Module:
    WEIGHTS_URL = "https://github.com/airctic/model_zoo/releases/download/m2/fridge_tf_efficientdet_lite0.pt"
    # TODO: HACK 5+1 in num_classes (becaues of change in two_stage_model.py)
    backbone = tf_lite0(pretrained=False)
    model = efficientdet.model(backbone=backbone, num_classes=5, img_size=384)

    state_dict = torch.hub.load_state_dict_from_url(
        WEIGHTS_URL, map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)

    return model

@pytest.fixture
def fridge_efficientdet_records(fridge_ds):
    for i, record in enumerate(fridge_ds[0].records):
        if record.filepath.stem == "10":
            return [fridge_ds[0][i]]


