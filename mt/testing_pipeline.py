import hydra
import os
import torch
from typing import List
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from loguru import logger as log
from pytorch_lightning.strategies import DDPStrategy

from mt.datamodules.dental import DentalCaries
from mt.transforms import TransformsComposer
from mt.utils.logger_utils import log_hyperparameters, finish

import logging



def test(config: DictConfig):

    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    log.info(f"Instantiating model <{config.module.model._target_}>")

    logging.getLogger('yolov5.models.yolo').disabled = True
    logging.getLogger('mmcv').disabled = True
    model: LightningModule = hydra.utils.instantiate(config.module.model)

    if config.module.get("pretrained"):
        pretrained_path = config.module.pretrained
        if config.module.pretrained and not os.path.isabs(config.module.pretrained):
            pretrained_path = os.path.join(hydra.utils.get_original_cwd(), config.module.pretrained)
        pretrained_model = torch.load(pretrained_path)
        model.load_state_dict(pretrained_model["state_dict"])
        log.info("loaded pretrained weights")

    # TODO this is experimental setup
    # bn2gn(model)

    transforms_composer: TransformsComposer = hydra.utils.instantiate(config.transforms, _recursive_=False)
    train_transforms, val_transforms = transforms_composer.train_val_transforms()
    dm: DentalCaries = hydra.utils.instantiate(config.datamodule, train_transforms=train_transforms,
                                               val_transforms=val_transforms)
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        # strategy=DDPStrategy(find_unused_parameters=False),
    )

    log.info("Starting testing")
    trainer.test(model=model, datamodule=dm)
