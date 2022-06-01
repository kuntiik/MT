import hydra
import pytorch_lightning as pl
import torch
from pathlib import Path
import albumentations as A
from src.transforms.albumentations_adapter import Adapter
from src.utils.bbox_inverse_transform import predictions_to_fiftyone
import json
from src.utils.conver_to_coco import to_coco
from src.evaluation.prediction_evaluation import PredictionEval


def create_overrides_and_path(ckpt_name):
    overrides = []
    target_path = ""
    if '/retinanet/' in ckpt_name:
        overrides.append('module=retinanet')
        target_path = Path('retinanet')

    elif '/faster_rcnn/' in ckpt_name:
        overrides.append('module=faster_rcnn')
        target_path = Path('faster_rcnn')

    elif '/yolov5/' in ckpt_name:
        overrides.append('module=yolov5')
        target_path = Path('yolov5')

    elif '/effdet/' in ckpt_name:
        overrides.append('module=efficientdet')
        target_path = Path('effdet')

    if '/medium_p6/' in ckpt_name:
        overrides.append('module.backbone=medium_p6')
        target_path = target_path / "medium"

    if '/small_p6/' in ckpt_name:
        overrides.append('module.backbone=small_p6')
        target_path = target_path / "small"

    elif '/large_p6/' in ckpt_name:
        overrides.append('module.backbone=large_p6')
        target_path = target_path / "large"

    elif '/resnet101/' in ckpt_name:
        overrides.append('module.backbone=resnet101_fpn_1x')
        target_path = target_path / "resnet101"

    elif '/resnet50/' in ckpt_name:
        overrides.append('module.backbone=resnet50_fpn_1x')
        target_path = target_path / "resnet50"

    elif '/swint/' in ckpt_name:
        overrides.append('module.backbone=swin_t_p4_w7_fpn_1x_coco')
        target_path = target_path / "swint"

    elif '/d0/' in ckpt_name:
        overrides.append('module.backbone=d0')
        target_path = target_path / "d0"
    elif '/d1/' in ckpt_name:
        overrides.append('module.backbone=d1')
        target_path = target_path / "d1"
    elif '/d2/' in ckpt_name:
        overrides.append('module.backbone=d2')
        target_path = target_path / "d2"
    elif '/tf_d0/' in ckpt_name:
        overrides.append('module.backbone=tf_d0')
        target_path = target_path / "tf_d0"
    elif '/tf_d3/' in ckpt_name:
        overrides.append('module.backbone=tf_d3')
        target_path = target_path / "tf_d3"

    return overrides, target_path


def predict_and_save(ckpt, ann_path, path=Path('predictions')):
    overrides, name = create_overrides_and_path(str(ckpt))
    overrides.append('logger.wandb.project=inference')
    with hydra.initialize(config_path="configs"):
        cfg = hydra.compose(
            config_name="train",
            overrides=overrides
        )
    model = hydra.utils.instantiate(cfg.module.model)
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])
    logger = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                logger.append(hydra.utils.instantiate(lg_conf))

    statistics = dict(mean=0.3669, std=0.2768)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    composer = hydra.utils.instantiate(cfg.transforms, _recursive_=False)
    t, v = composer.train_val_transforms()
    v = Adapter([A.Resize(896, 1024), A.Normalize(mean=statistics["mean"], std=statistics["std"])])
    dm = hydra.utils.instantiate(cfg.datamodule, train_transforms=v, val_transforms=v)
    dm.setup()

    test_prediction = trainer.predict(model, dm.predict_dataloader('test'))
    val_prediction = trainer.predict(model, dm.predict_dataloader('val'))
    train_prediction = trainer.predict(model, dm.predict_dataloader('train'))

    test_data = predictions_to_fiftyone(test_prediction, stage='test')
    val_data = predictions_to_fiftyone(val_prediction, stage='val')
    train_data = predictions_to_fiftyone(train_prediction, stage='train')
    data = {**test_data, **val_data, **train_data}

    with open(ann_path, 'r') as f:
        ann_file = json.load(f)

    preds_coco, names = to_coco(data, ann_file)
    pred_eval = PredictionEval()
    pred_eval.load_data_coco_files(ann_path, preds_coco, names)

    map50 = round(pred_eval.map_query(stage='test'), 3)
    dir_name = path / name
    dir_name.mkdir(parents=True, exist_ok=True)
    save_name = dir_name / (str(map50) + '.json')

    with open(save_name, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    ann_path = '/datagrid/personal/kuntluka/dental_rtg3/annotations.json'
    weights_root = Path('/datagrid/personal/kuntluka/weights').rglob("**/*.ckpt")
    ckpt_path = list(weights_root)
    for ckpt in ckpt_path:
        predict_and_save(ckpt, ann_path)

