import hydra
import pytest
from albumentations import Normalize, PadIfNeeded, LongestMaxSize, HorizontalFlip

from mt.transforms.transforms_composer import TransformsComposer


def test_default_composition(cfg):
    assert cfg.seed == 42
    assert 'trainer' in cfg
    assert 'module' in cfg
    assert 'transforms' in cfg


def test_overrides_composition():
    with hydra.initialize(config_path='../configs'):
        cfg = hydra.compose(config_name='train', overrides=['experiment=yolov5', 'datamodule.batch_size=1'])

    assert 'experiment' in cfg


