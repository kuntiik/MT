import pytest

from src.datamodules.dental_caries.dental_caries import DentalCaries
from hydra.utils import instantiate


def test_efficient_det_module(cfg):
    module = instantiate(cfg.module)


