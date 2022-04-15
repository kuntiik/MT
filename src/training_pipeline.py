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

from src.datamodules.dental_caries import DentalCaries
from src.transforms import TransformsComposer
from src.utils.logger_utils import log_hyperparameters, finish

import logging

def train(config: DictConfig):
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # convert relative path to absolute (hydra requires absolute path)
    ckpt_path = config.trainer.get("resume_from_checkpoint")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.trainer.resume_from_checkpoint = os.path.join(
            hydra.utils.get_original_cwd(), ckpt_path
        )

    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

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
        if config.module.pretrained and not os.path.isabs(config.module.pretrained):
            pretrained_path = os.path.join(hydra.utils.get_original_cwd(), config.module.petrained)
            pretrained_model = torch.load(pretrained_path)
            model.load_state_dict(pretrained_model["state_dict"])

    transforms_composer : TransformsComposer = hydra.utils.instantiate(config.transforms, _recursive_=False)
    train_transforms, val_transforms = transforms_composer.train_val_transforms()
    dm: DentalCaries = hydra.utils.instantiate(config.datamodule, train_transforms=train_transforms, val_transforms=val_transforms)
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False),
    )

    log_hyperparameters(
        config=config,
        model=model,
        datamodule=dm,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    log.info("Starting training")
    trainer.fit(model=model, datamodule=dm)

    # finish(config, model, dm, trainer, callbacks, logger)
    # optimized_metric = config.get("optimized_metric")
    # if optimized_metric and optimized_metric not in trainer.callback_metrics:
    #     raise Exception(
    #         "Metric for hyperparameter optimization not found! "
    #         "Make sure the `optimized_metric` in `hparams_search` config is correct!"
    #     )
    # score = trainer.callback_metrics.get(optimized_metric)
    # return score
