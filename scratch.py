import hydra
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

if __name__ == '__main__':

    with hydra.initialize(config_path="mt/configs"):
        cfg = hydra.compose(
            config_name="train",
            overrides=['experiment=segmentation']
        )
        logger = []
        if "logger" in cfg:
            for _, lg_conf in cfg.logger.items():
                if "_target_" in lg_conf:
                    logger.append(hydra.utils.instantiate(lg_conf))
        trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
        statistics = dict(mean=0.3669, std=0.2768)
        v = A.Compose([A.Resize(896, 1024), A.Normalize(mean=statistics["mean"], std=statistics["std"]), ToTensorV2()])
        dm = hydra.utils.instantiate(cfg.datamodule, train_transforms=v, val_transforms=v)
        dm.setup()


    ckpt_path = '/datagrid/personal/kuntluka/weights/restoration_segmentation/unet/unet_0.662.ckpt'
    model = hydra.utils.instantiate(cfg.module.model)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
    foo = trainer.predict(model, dm.val_dataloader())

    print('done')
