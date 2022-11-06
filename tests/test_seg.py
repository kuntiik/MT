import hydra
from mt.modules import SegmentationModuleTorch
from mt.training_pipeline import train

def test_model():
    with hydra.initialize(config_path='../configs'):
        cfg = hydra.compose(config_name='segmentation', overrides=[])
    model = hydra.utils.instantiate(cfg.module.model)
    module = hydra.utils.instantiate(cfg.module)

def test_model():
    with hydra.initialize(config_path='../configs'):
        cfg = hydra.compose(config_name='train', overrides=['++trainer.fast_dev_run=True', 'experiment=unet', 'trainer.gpus=0', 'datamodule.num_workers=0'])
    train(cfg)

def test_inference():
    with hydra.initialize(config_path='../configs'):
        cfg = hydra.compose(config_name='train', overrides=['experiment=unet', 'trainer.gpus=1', 'datamodule.num_workers=0'])
    model = hydra.utils.instantiate(cfg.module.model)
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup(stage='predict')
    trainer = hydra.utils.instantiate(cfg.trainer)
    val_data = trainer.predict(model, dm.predict_dataloader('val'))




