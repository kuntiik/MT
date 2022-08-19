import yaml
from pathlib import Path
import shutil


def get_backbone(b: str):
    if 'swin_t' in b:
        return 'swin_t'
    if 'resnet50' in b:
        return 'resnset50'
    if 'resnet101' in b:
        return 'resnset101'
    if 'small' in b:
        return 'small'
    if 'medium' in b:
        return 'medium'
    if 'large' in b:
        return 'large'
    if 'd0' in b:
        return 'd0'
    if 'd1' in b:
        return 'd1'
    if 'd2' in b:
        return 'd2'
    if 'd3' in b:
        return 'd3'
    if 'd4' in b:
        return 'd4'
    if 'd5' in b:
        return 'd5'


def get_architecture(a: str):
    if 'yolov5' in a:
        return 'yolov5'
    if 'faster_rcnn' in a:
        return 'faster_rcnn'
    if 'retinanet' in a:
        return 'retinanet'
    if 'efficientdet' in a:
        return 'efficientdet'


def arch_bb(cfg):
    architecture_key = 'model/model/model/_target_'
    backbone_key = 'model/model/model/backbone/_target_'
    if not architecture_key in cfg.keys() or not architecture_key in cfg.keys():
        return None
    else:
        architecture = get_architecture(cfg[architecture_key]['value'])
        backbone = get_backbone(cfg[backbone_key]['value'])
        return architecture, backbone


def load_cfg(path):
    cfg_path = path / 'wandb/latest-run/files/config.yaml'
    if cfg_path.exists():
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        return None


def load_ckpt(path):
    ckpt_path = path / 'checkpoints'
    if not ckpt_path.exists():
        return None
    else:
        ckpts = list(ckpt_path.glob('epoch*.ckpt'))
        if len(ckpts) == 1:
            return ckpts[0]
        else:
            return None


def set_name(ckpt_path):
    name = ckpt_path.stem
    name = name.split('_')[-1] + ckpt_path.suffix
    while ckpt_path.with_name(name).exists():
        name = name.replace(ckpt_path.suffix, '') + '0' + ckpt_path.suffix
    return ckpt_path.with_name(name)


def get_cfgs_ckpt(ckpt_folder):
    ckpt_folder = Path(ckpt_folder) if type(ckpt_folder) is str else ckpt_folder
    configs = []
    checkpoints = []
    for path in ckpt_folder.iterdir():
        if (path / 'multirun.yaml').exists():
            for p in path.iterdir():
                if not p.is_dir():
                    continue
                configs.append(load_cfg(p))
                checkpoints.append(load_ckpt(p))
        else:
            configs.append(load_cfg(path))
            checkpoints.append(load_ckpt(path))
    return configs, checkpoints


def organize_ckpts(ckpt_folder: Path, target_folder):
    configs, checkpoints = get_cfgs_ckpt(ckpt_folder)

    for cfg, ckpt in zip(configs, checkpoints):
        if cfg is None or ckpt is None:
            continue
        else:
            arch, bb = arch_bb(cfg)
            target = target_folder / arch / bb
            target.mkdir(parents=True, exist_ok=True)
            name = set_name(target / ckpt.name)

            shutil.copyfile(str(ckpt), name)
